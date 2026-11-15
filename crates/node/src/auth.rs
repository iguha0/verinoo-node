//! API-key authentication and per-address daily quotas.
//!
//! A key is bound to one funded address: it may only submit transactions whose
//! `from` matches that address. That gives tenant isolation without changing
//! consensus state, and lets the existing per-address metering answer "what did
//! this tenant use?". Only the SHA-256 of a key is written to disk; the raw key
//! is printed or returned once when it is created. Operator keys live in a JSON
//! file; with `VNN_SELF_SERVE_KEYS=1`, self-serve keys live in RocksDB meta.
//!
//! Quotas are enforced at the gateway, not by consensus: they are a billing and
//! abuse control, not a security invariant. Reload the key file with SIGHUP:
//! `systemctl kill -s HUP verinoo-node`.

use axum::http::{header, HeaderMap};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use verinoo_state::StateStore;

const KEY_PREFIX: &str = "vnn_";
const KEY_BYTES: usize = 32;
const QUOTA_DAY_MS: u64 = 86_400_000;
const DEFAULT_KEYS_FILE: &str = "/etc/verinoo/api-keys.json";
const SELF_SERVE_TENANT: &str = "self-serve";
const SELF_SERVE_KEY_PREFIX: &str = "selfserve:key:";
const SELF_SERVE_ADDR_PREFIX: &str = "selfserve:addr:";
const SELF_SERVE_DEFAULT_QUOTA: u64 = 1000;
const ADDRESS_HEX_LEN: usize = 40;

fn default_enabled() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyEntry {
    #[serde(rename = "keyHash")]
    pub key_hash: String,
    pub tenant: String,
    pub address: String,
    /// Tasks per UTC day. `0` means unlimited.
    #[serde(rename = "dailyQuota", default)]
    pub daily_quota: u64,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct KeyFile {
    #[serde(default)]
    pub keys: Vec<KeyEntry>,
}

#[derive(Debug, Default)]
pub struct ApiKeys {
    by_hash: HashMap<String, KeyEntry>,
}

impl ApiKeys {
    pub fn load(path: &Path) -> Result<Self, String> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| format!("reading API keys {}: {e}", path.display()))?;
        let file: KeyFile = serde_json::from_str(&raw)
            .map_err(|e| format!("parsing API keys {}: {e}", path.display()))?;
        let mut by_hash = HashMap::new();
        for entry in file.keys {
            validate_entry(&entry)?;
            if by_hash.insert(entry.key_hash.clone(), entry).is_some() {
                return Err("duplicate keyHash in API key file".into());
            }
        }
        Ok(Self { by_hash })
    }

    pub fn len(&self) -> usize {
        self.by_hash.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_hash.is_empty()
    }

    /// Look up a raw key. Timing is not constant, but keys are high-entropy and
    /// this is a gateway lookup, not a password check.
    pub fn authorize(&self, raw_key: &str) -> Option<&KeyEntry> {
        let hash = hash_key(raw_key);
        self.by_hash.get(&hash).filter(|e| e.enabled)
    }
}

fn validate_entry(entry: &KeyEntry) -> Result<(), String> {
    if entry.key_hash.len() != 64 || !entry.key_hash.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!(
            "tenant `{}`: keyHash must be 64 hex chars",
            entry.tenant
        ));
    }
    if !entry.address.starts_with("ai_") || entry.address.len() != 43 {
        return Err(format!(
            "tenant `{}`: address `{}` is not a verinoo address",
            entry.tenant, entry.address
        ));
    }
    if entry.tenant.trim().is_empty() {
        return Err("tenant name must not be empty".into());
    }
    Ok(())
}

/// A self-serve key persisted in RocksDB. Unlike `KeyEntry` it has no
/// `keyHash` field: the hash is already the meta key.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SelfServeKey {
    tenant: String,
    address: String,
    #[serde(rename = "dailyQuota", default)]
    daily_quota: u64,
    #[serde(default = "default_enabled")]
    enabled: bool,
    #[serde(rename = "createdAt", default)]
    created_at: u64,
}

/// A newly issued self-serve key. `api_key` is the only time the raw value
/// exists; callers must return it to the client immediately.
#[derive(Debug, Clone)]
pub struct IssuedSelfServeKey {
    pub api_key: String,
    pub tenant: String,
    pub address: String,
    pub daily_quota: u64,
    pub created_at: u64,
}

#[derive(Debug, Clone)]
pub enum SelfServeError {
    InvalidAddress,
    AlreadyIssued { tenant: String, address: String },
}

pub fn self_serve_enabled() -> bool {
    std::env::var("VNN_SELF_SERVE_KEYS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

pub fn self_serve_default_quota() -> u64 {
    std::env::var("VNN_SELF_SERVE_QUOTA")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(SELF_SERVE_DEFAULT_QUOTA)
}

fn valid_self_serve_address(address: &str) -> bool {
    address.len() == "ai_".len() + ADDRESS_HEX_LEN
        && address.starts_with("ai_")
        && address["ai_".len()..]
            .chars()
            .all(|c| c.is_ascii_hexdigit())
}

/// Issue a key bound to `address`, storing only its SHA-256 hash. One active
/// key per address: a second call returns the existing tenant/address, never
/// the raw key.
pub fn issue_self_serve_key(
    store: &StateStore,
    address: &str,
    daily_quota: u64,
    now_ms: u64,
) -> Result<IssuedSelfServeKey, SelfServeError> {
    let address = address.trim();
    if !valid_self_serve_address(address) {
        return Err(SelfServeError::InvalidAddress);
    }
    let addr_key = format!("{SELF_SERVE_ADDR_PREFIX}{address}");
    if let Some(hash) = store
        .get_meta(&addr_key)
        .and_then(|v| v.as_str().map(String::from))
    {
        let tenant = store
            .get_meta(&format!("{SELF_SERVE_KEY_PREFIX}{hash}"))
            .and_then(|v| serde_json::from_value::<SelfServeKey>(v).ok())
            .map(|k| k.tenant)
            .unwrap_or_else(|| SELF_SERVE_TENANT.to_string());
        return Err(SelfServeError::AlreadyIssued {
            tenant,
            address: address.to_string(),
        });
    }

    let raw_key = generate_key();
    let hash = hash_key(&raw_key);
    let record = SelfServeKey {
        tenant: SELF_SERVE_TENANT.to_string(),
        address: address.to_string(),
        daily_quota,
        enabled: true,
        created_at: now_ms,
    };
    store.set_meta(
        &format!("{SELF_SERVE_KEY_PREFIX}{hash}"),
        &serde_json::to_value(&record).expect("serialize self-serve key"),
    );
    store.set_meta(&addr_key, &serde_json::json!(hash));
    Ok(IssuedSelfServeKey {
        api_key: raw_key,
        tenant: record.tenant,
        address: record.address,
        daily_quota: record.daily_quota,
        created_at: record.created_at,
    })
}

fn lookup_self_serve(store: &StateStore, raw_key: &str) -> Option<Caller> {
    let hash = hash_key(raw_key);
    let record: SelfServeKey =
        serde_json::from_value(store.get_meta(&format!("{SELF_SERVE_KEY_PREFIX}{hash}"))?).ok()?;
    if !record.enabled {
        return None;
    }
    Some(Caller {
        tenant: record.tenant,
        address: record.address,
        daily_quota: record.daily_quota,
    })
}

pub fn hash_key(raw_key: &str) -> String {
    let mut h = Sha256::new();
    h.update(raw_key.as_bytes());
    hex::encode(h.finalize())
}

fn generate_key() -> String {
    use rand::RngCore as _;
    let mut bytes = [0u8; KEY_BYTES];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    format!("{KEY_PREFIX}{}", hex::encode(bytes))
}

fn write_key_file(path: &Path, file: &KeyFile) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("creating {}: {e}", parent.display()))?;
        }
    }
    let mut text = serde_json::to_string_pretty(file).map_err(|e| e.to_string())?;
    text.push('\n');
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, text).map_err(|e| format!("writing {}: {e}", tmp.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600))
            .map_err(|e| format!("chmod {}: {e}", tmp.display()))?;
    }
    std::fs::rename(&tmp, path).map_err(|e| format!("installing {}: {e}", path.display()))
}

/// Load or create the key file at `path` and append one key.
pub fn add_key(
    path: &Path,
    tenant: &str,
    address: &str,
    daily_quota: u64,
) -> Result<String, String> {
    let mut file = if path.exists() {
        let raw = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        serde_json::from_str::<KeyFile>(&raw)
            .map_err(|e| format!("parsing {}: {e}", path.display()))?
    } else {
        KeyFile::default()
    };
    let raw_key = generate_key();
    let entry = KeyEntry {
        key_hash: hash_key(&raw_key),
        tenant: tenant.trim().to_string(),
        address: address.trim().to_string(),
        daily_quota,
        enabled: true,
    };
    validate_entry(&entry)?;
    if file.keys.iter().any(|k| k.key_hash == entry.key_hash) {
        return Err("generated key collided; try again".into());
    }
    file.keys.push(entry);
    write_key_file(path, &file)?;
    Ok(raw_key)
}

pub fn revoke(
    path: &Path,
    tenant: Option<&str>,
    hash_prefix: Option<&str>,
) -> Result<usize, String> {
    let raw = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut file: KeyFile =
        serde_json::from_str(&raw).map_err(|e| format!("parsing {}: {e}", path.display()))?;
    let before = file.keys.len();
    file.keys.retain(|k| {
        let tenant_match = tenant.map(|t| k.tenant == t).unwrap_or(false);
        let hash_match = hash_prefix
            .map(|p| k.key_hash.starts_with(p))
            .unwrap_or(false);
        !(tenant_match || hash_match)
    });
    let removed = before - file.keys.len();
    write_key_file(path, &file)?;
    Ok(removed)
}

/// A successful key lookup: who is calling and what they are allowed to do.
#[derive(Debug, Clone)]
pub struct Caller {
    pub tenant: String,
    pub address: String,
    pub daily_quota: u64,
}

#[derive(Debug)]
pub enum AuthError {
    Missing,
    Invalid,
    AddressMismatch,
    QuotaExceeded { used: u64, quota: u64 },
}

impl AuthError {
    pub fn status(&self) -> axum::http::StatusCode {
        use axum::http::StatusCode;
        match self {
            AuthError::Missing | AuthError::Invalid => StatusCode::UNAUTHORIZED,
            AuthError::AddressMismatch => StatusCode::FORBIDDEN,
            AuthError::QuotaExceeded { .. } => StatusCode::TOO_MANY_REQUESTS,
        }
    }

    pub fn message(&self) -> String {
        match self {
            AuthError::Missing => "missing Authorization: Bearer <api-key>".into(),
            AuthError::Invalid => "invalid or disabled API key".into(),
            AuthError::AddressMismatch => {
                "API key is not authorized for this transaction sender".into()
            }
            AuthError::QuotaExceeded { used, quota } => {
                format!("daily quota exceeded ({used}/{quota})")
            }
        }
    }
}

/// Shared authentication state. `inner == None` means authentication is
/// disabled (local demo / single-operator mode).
#[derive(Clone)]
pub struct Auth {
    inner: Option<Arc<RwLock<ApiKeys>>>,
    path: Option<PathBuf>,
    quota_lock: Arc<std::sync::Mutex<()>>,
    issue_lock: Arc<std::sync::Mutex<()>>,
    self_serve: bool,
    self_serve_quota: u64,
}

impl Auth {
    pub fn disabled() -> Self {
        Self {
            inner: None,
            path: None,
            quota_lock: Arc::new(std::sync::Mutex::new(())),
            issue_lock: Arc::new(std::sync::Mutex::new(())),
            self_serve: false,
            self_serve_quota: SELF_SERVE_DEFAULT_QUOTA,
        }
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let keys = ApiKeys::load(path)?;
        Ok(Self {
            inner: Some(Arc::new(RwLock::new(keys))),
            path: Some(path.to_path_buf()),
            quota_lock: Arc::new(std::sync::Mutex::new(())),
            issue_lock: Arc::new(std::sync::Mutex::new(())),
            self_serve: false,
            self_serve_quota: SELF_SERVE_DEFAULT_QUOTA,
        })
    }

    /// `VNN_API_KEYS_FILE` if set; otherwise the default file if it exists;
    /// otherwise authentication is disabled. Self-serve issuance is gated
    /// separately by `VNN_SELF_SERVE_KEYS`.
    pub fn from_env() -> Result<Self, String> {
        let mut auth = match std::env::var("VNN_API_KEYS_FILE") {
            Ok(raw) if !raw.trim().is_empty() => Self::load(Path::new(raw.trim()))?,
            _ => {
                let default = Path::new(DEFAULT_KEYS_FILE);
                if default.exists() {
                    Self::load(default)?
                } else {
                    Self::disabled()
                }
            }
        };
        auth.self_serve = self_serve_enabled();
        auth.self_serve_quota = self_serve_default_quota();
        Ok(auth)
    }

    #[cfg(test)]
    pub fn with_self_serve(mut self, enabled: bool) -> Self {
        self.self_serve = enabled;
        self
    }

    #[cfg(test)]
    pub fn with_self_serve_quota(mut self, quota: u64) -> Self {
        self.self_serve_quota = quota;
        self
    }

    pub fn self_serve_enabled(&self) -> bool {
        self.self_serve
    }

    pub fn is_enabled(&self) -> bool {
        self.inner.is_some() || self.self_serve
    }

    pub fn len(&self) -> usize {
        self.inner
            .as_ref()
            .map(|k| k.read().expect("api keys lock").len())
            .unwrap_or(0)
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    #[cfg(test)]
    pub fn authorize(&self, headers: &HeaderMap) -> Result<Caller, AuthError> {
        let Some(inner) = &self.inner else {
            return Err(AuthError::Invalid);
        };
        let raw = bearer(headers).ok_or(AuthError::Missing)?;
        let keys = inner.read().expect("api keys lock");
        let entry = keys.authorize(&raw).ok_or(AuthError::Invalid)?;
        Ok(Caller {
            tenant: entry.tenant.clone(),
            address: entry.address.clone(),
            daily_quota: entry.daily_quota,
        })
    }

    /// Authorize against file-based keys first, then self-serve keys stored in
    /// RocksDB. Self-serve lookups are only attempted when the feature is on.
    pub fn authorize_with_store(
        &self,
        headers: &HeaderMap,
        store: &StateStore,
    ) -> Result<Caller, AuthError> {
        let raw = bearer(headers).ok_or(AuthError::Missing)?;
        if let Some(inner) = &self.inner {
            let keys = inner.read().expect("api keys lock");
            if let Some(entry) = keys.authorize(&raw) {
                return Ok(Caller {
                    tenant: entry.tenant.clone(),
                    address: entry.address.clone(),
                    daily_quota: entry.daily_quota,
                });
            }
        }
        if self.self_serve {
            if let Some(caller) = lookup_self_serve(store, &raw) {
                return Ok(caller);
            }
        }
        Err(AuthError::Invalid)
    }

    pub fn issue_self_serve_key(
        &self,
        store: &StateStore,
        address: &str,
        now_ms: u64,
    ) -> Result<IssuedSelfServeKey, SelfServeError> {
        let _guard = self.issue_lock.lock().expect("self-serve issue lock");
        self::issue_self_serve_key(store, address, self.self_serve_quota, now_ms)
    }

    /// Check and consume one inference from today's allowance.
    pub fn consume_quota(
        &self,
        store: &StateStore,
        caller: &Caller,
        now_ms: u64,
    ) -> Result<u64, AuthError> {
        if caller.daily_quota == 0 {
            return Ok(0);
        }
        let _guard = self.quota_lock.lock().expect("quota lock");
        let key = quota_key(&caller.address, now_ms);
        let used = store.meta_u64_or(&key, 0);
        if used >= caller.daily_quota {
            return Err(AuthError::QuotaExceeded {
                used,
                quota: caller.daily_quota,
            });
        }
        let next = used + 1;
        store.set_meta_u64(&key, next);
        Ok(next)
    }

    pub fn reload(&self) -> Result<usize, String> {
        let (Some(inner), Some(path)) = (&self.inner, &self.path) else {
            return Ok(0);
        };
        let keys = ApiKeys::load(path)?;
        let n = keys.len();
        *inner.write().expect("api keys lock") = keys;
        Ok(n)
    }

    /// Reload the key file on SIGHUP so keys can be rotated without downtime.
    pub fn spawn_reload_on_sighup(&self) {
        if self.path.is_none() {
            return;
        }
        let this = self.clone();
        tokio::spawn(async move {
            #[cfg(unix)]
            {
                use tokio::signal::unix::{signal, SignalKind};
                let mut sig = match signal(SignalKind::hangup()) {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::warn!("cannot install SIGHUP handler: {e}");
                        return;
                    }
                };
                while sig.recv().await.is_some() {
                    match this.reload() {
                        Ok(n) => tracing::info!("api keys reloaded ({n} entries)"),
                        Err(e) => tracing::error!("api key reload failed: {e}"),
                    }
                }
            }
        });
    }
}

fn bearer(headers: &HeaderMap) -> Option<String> {
    let value = headers.get(header::AUTHORIZATION)?.to_str().ok()?;
    let token = value
        .strip_prefix("Bearer ")
        .or_else(|| value.strip_prefix("bearer "))?;
    let token = token.trim();
    if token.is_empty() {
        None
    } else {
        Some(token.to_string())
    }
}

pub fn quota_key(address: &str, now_ms: u64) -> String {
    // Per address, not per key: issuing a second key for the same address must
    // not multiply its daily allowance.
    format!("quota:{address}:{}", now_ms / QUOTA_DAY_MS)
}

// ============================== CLI ==============================

pub fn cli(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        print_usage();
        return Ok(());
    }
    let cmd = args[0].as_str();
    let mut file = std::env::var("VNN_API_KEYS_FILE")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_KEYS_FILE.to_string());
    let mut tenant: Option<String> = None;
    let mut address: Option<String> = None;
    let mut quota: u64 = 1000;
    let mut hash_prefix: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        let next = |i: &mut usize| -> Result<String, String> {
            *i += 1;
            args.get(*i)
                .cloned()
                .ok_or_else(|| format!("{} needs a value", args[*i - 1]))
        };
        match args[i].as_str() {
            "--file" => file = next(&mut i)?,
            "--tenant" => tenant = Some(next(&mut i)?),
            "--address" => address = Some(next(&mut i)?),
            "--quota" => {
                let raw = next(&mut i)?;
                quota = raw
                    .parse()
                    .map_err(|_| format!("--quota expects a number, got `{raw}`"))?;
            }
            "--hash" => hash_prefix = Some(next(&mut i)?),
            "--help" | "-h" => {
                print_usage();
                return Ok(());
            }
            other => return Err(format!("unknown argument `{other}` (try --help)")),
        }
        i += 1;
    }

    let path = Path::new(&file);
    match cmd {
        "new" => {
            let tenant = tenant.ok_or("--tenant is required")?;
            let address = address.ok_or("--address is required")?;
            let raw = add_key(path, &tenant, &address, quota)?;
            println!("key:      {raw}");
            println!("tenant:   {tenant}");
            println!("address:  {address}");
            println!(
                "quota:    {}",
                if quota == 0 {
                    "unlimited".to_string()
                } else {
                    format!("{quota}/day")
                }
            );
            println!("file:     {}", path.display());
            println!();
            println!("The key is shown once; only its hash is stored. Reload the node with:");
            println!("  systemctl kill -s HUP verinoo-node");
        }
        "list" => {
            if !path.exists() {
                return Err(format!("no key file at {}", path.display()));
            }
            let keys = ApiKeys::load(path)?;
            if keys.is_empty() {
                println!("no keys");
                return Ok(());
            }
            println!(
                "{:<16} {:<46} {:>10}  {:<7} hash",
                "tenant", "address", "quota/day", "enabled"
            );
            let mut entries: Vec<&KeyEntry> = keys.by_hash.values().collect();
            entries.sort_by(|a, b| a.tenant.cmp(&b.tenant));
            for e in entries {
                println!(
                    "{:<16} {:<46} {:>10}  {:<7} {}…",
                    e.tenant,
                    e.address,
                    if e.daily_quota == 0 {
                        "unlimited".to_string()
                    } else {
                        e.daily_quota.to_string()
                    },
                    e.enabled,
                    &e.key_hash[..12]
                );
            }
        }
        "revoke" => {
            if tenant.is_none() && hash_prefix.is_none() {
                return Err("--tenant or --hash is required".into());
            }
            let removed = revoke(path, tenant.as_deref(), hash_prefix.as_deref())?;
            println!("revoked {removed} key(s) in {}", path.display());
            println!("reload the node with: systemctl kill -s HUP verinoo-node");
        }
        other => return Err(format!("unknown apikey command `{other}` (try --help)")),
    }
    Ok(())
}

fn print_usage() {
    println!(
        "Usage: verinoo-node apikey <command> [options]\n\
         \n\
         Commands:\n\
           new     --tenant <name> --address <ai_...> [--quota N, default 1000] [--file path]\n\
           list    [--file path]\n\
           revoke  (--tenant <name> | --hash <prefix>) [--file path]\n\
         \n\
         --quota 0 means unlimited. Default file: {DEFAULT_KEYS_FILE}\n\
         or $VNN_API_KEYS_FILE."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_file(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("verinoo-auth-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        let _ = std::fs::remove_file(&path);
        path
    }

    const ADDR: &str = "ai_0123456789abcdef0123456789abcdef01234567";

    #[test]
    fn keys_are_stored_as_hashes_and_authorize_the_raw_value() {
        let path = temp_file("keys.json");
        let raw = add_key(&path, "acme", ADDR, 100).unwrap();
        assert!(raw.starts_with(KEY_PREFIX));
        let disk = std::fs::read_to_string(&path).unwrap();
        assert!(
            !disk.contains(&raw),
            "raw key must never be written to disk"
        );
        assert!(disk.contains(&hash_key(&raw)));

        let auth = Auth::load(&path).unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            format!("Bearer {raw}").parse().unwrap(),
        );
        let caller = auth.authorize(&headers).unwrap();
        assert_eq!(caller.tenant, "acme");
        assert_eq!(caller.address, ADDR);
        assert_eq!(caller.daily_quota, 100);

        headers.insert(header::AUTHORIZATION, "Bearer wrong".parse().unwrap());
        assert!(matches!(auth.authorize(&headers), Err(AuthError::Invalid)));
    }

    #[test]
    fn missing_header_and_disabled_keys_are_rejected() {
        let path = temp_file("disabled.json");
        let raw = add_key(&path, "acme", ADDR, 0).unwrap();
        let mut file: KeyFile =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        file.keys[0].enabled = false;
        write_key_file(&path, &file).unwrap();

        let auth = Auth::load(&path).unwrap();
        let mut headers = HeaderMap::new();
        assert!(matches!(auth.authorize(&headers), Err(AuthError::Missing)));
        headers.insert(
            header::AUTHORIZATION,
            format!("Bearer {raw}").parse().unwrap(),
        );
        assert!(matches!(auth.authorize(&headers), Err(AuthError::Invalid)));
    }

    #[test]
    fn quota_is_per_address_per_utc_day() {
        let store = StateStore::open_temp().unwrap();
        let path = temp_file("quota.json");
        add_key(&path, "acme", ADDR, 2).unwrap();
        let auth = Auth::load(&path).unwrap();
        let caller = Caller {
            tenant: "acme".into(),
            address: ADDR.into(),
            daily_quota: 2,
        };
        let now = 1_700_000_000_000u64; // fixed day
        assert_eq!(auth.consume_quota(&store, &caller, now).unwrap(), 1);
        assert_eq!(auth.consume_quota(&store, &caller, now).unwrap(), 2);
        assert!(matches!(
            auth.consume_quota(&store, &caller, now),
            Err(AuthError::QuotaExceeded { used: 2, quota: 2 })
        ));
        // Next UTC day is a fresh allowance.
        assert_eq!(
            auth.consume_quota(&store, &caller, now + QUOTA_DAY_MS)
                .unwrap(),
            1
        );
        // Unlimited keys are never counted.
        let unlimited = Caller {
            daily_quota: 0,
            ..caller
        };
        assert_eq!(auth.consume_quota(&store, &unlimited, now).unwrap(), 0);
    }

    #[test]
    fn revoke_removes_by_tenant() {
        let path = temp_file("revoke.json");
        add_key(&path, "acme", ADDR, 1).unwrap();
        add_key(&path, "other", ADDR, 1).unwrap();
        assert_eq!(revoke(&path, Some("acme"), None).unwrap(), 1);
        let keys = ApiKeys::load(&path).unwrap();
        assert_eq!(keys.len(), 1);
        assert!(keys.by_hash.values().all(|k| k.tenant == "other"));
    }

    #[test]
    fn invalid_addresses_are_refused() {
        let path = temp_file("bad-address.json");
        let err = add_key(&path, "acme", "0xdeadbeef", 1).unwrap_err();
        assert!(err.contains("not a verinoo address"), "{err}");
    }

    #[test]
    fn self_serve_issue_stores_only_hash_and_is_one_per_address() {
        let store = StateStore::open_temp().unwrap();
        let now = 1_700_000_000_000u64;
        let issued = issue_self_serve_key(&store, ADDR, 25, now).unwrap();
        assert!(issued.api_key.starts_with(KEY_PREFIX));
        assert_eq!(issued.api_key.len(), KEY_PREFIX.len() + 64);
        assert_eq!(issued.tenant, SELF_SERVE_TENANT);
        assert_eq!(issued.address, ADDR);
        assert_eq!(issued.daily_quota, 25);
        assert_eq!(issued.created_at, now);

        let hash = hash_key(&issued.api_key);
        let record = store
            .get_meta(&format!("{SELF_SERVE_KEY_PREFIX}{hash}"))
            .expect("key record stored");
        assert_eq!(record["tenant"], SELF_SERVE_TENANT);
        assert_eq!(record["address"], ADDR);
        assert_eq!(record["dailyQuota"], 25);
        assert_eq!(record["enabled"], true);
        assert_eq!(record["createdAt"], now);
        assert_eq!(
            store
                .get_meta(&format!("{SELF_SERVE_ADDR_PREFIX}{ADDR}"))
                .unwrap(),
            serde_json::json!(hash)
        );
        for value in store.iter_prefix::<serde_json::Value>("meta:selfserve:key:") {
            assert!(
                !value.to_string().contains(&issued.api_key),
                "raw key must never be stored"
            );
        }

        match issue_self_serve_key(&store, ADDR, 25, now).unwrap_err() {
            SelfServeError::AlreadyIssued { tenant, address } => {
                assert_eq!(tenant, SELF_SERVE_TENANT);
                assert_eq!(address, ADDR);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn self_serve_rejects_invalid_addresses() {
        let store = StateStore::open_temp().unwrap();
        let bad = [
            "",
            "0xdeadbeef",
            "ai_1234",
            "ai_0123456789abcdef0123456789abcdef0123456z",
        ];
        for address in bad {
            assert!(
                matches!(
                    issue_self_serve_key(&store, address, 1, 0),
                    Err(SelfServeError::InvalidAddress)
                ),
                "expected {address:?} to be rejected"
            );
        }
    }

    #[test]
    fn self_serve_authorize_with_store_and_reject_wrong_or_disabled() {
        let store = StateStore::open_temp().unwrap();
        let issued = issue_self_serve_key(&store, ADDR, 7, 1).unwrap();
        let auth = Auth::disabled().with_self_serve(true);
        let mut headers = HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            format!("Bearer {}", issued.api_key).parse().unwrap(),
        );
        let caller = auth.authorize_with_store(&headers, &store).unwrap();
        assert_eq!(caller.tenant, SELF_SERVE_TENANT);
        assert_eq!(caller.address, ADDR);
        assert_eq!(caller.daily_quota, 7);

        headers.insert(
            header::AUTHORIZATION,
            "Bearer vnn_deadbeef".parse().unwrap(),
        );
        assert!(matches!(
            auth.authorize_with_store(&headers, &store),
            Err(AuthError::Invalid)
        ));

        // Feature off: DB keys are ignored.
        headers.insert(
            header::AUTHORIZATION,
            format!("Bearer {}", issued.api_key).parse().unwrap(),
        );
        assert!(matches!(
            Auth::disabled().authorize_with_store(&headers, &store),
            Err(AuthError::Invalid)
        ));

        // Disabled record: rejected even when the feature is on.
        let key = format!("{SELF_SERVE_KEY_PREFIX}{}", hash_key(&issued.api_key));
        let mut record = store.get_meta(&key).unwrap();
        record["enabled"] = serde_json::json!(false);
        store.set_meta(&key, &record);
        assert!(matches!(
            auth.authorize_with_store(&headers, &store),
            Err(AuthError::Invalid)
        ));
    }

    #[test]
    fn authorize_with_store_still_checks_file_keys_first() {
        let store = StateStore::open_temp().unwrap();
        let path = temp_file("store-file.json");
        let raw = add_key(&path, "acme", ADDR, 3).unwrap();
        let auth = Auth::load(&path).unwrap().with_self_serve(true);
        let mut headers = HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            format!("Bearer {raw}").parse().unwrap(),
        );
        let caller = auth.authorize_with_store(&headers, &store).unwrap();
        assert_eq!(caller.tenant, "acme");
        assert_eq!(caller.address, ADDR);
        assert_eq!(caller.daily_quota, 3);
    }

    #[test]
    fn self_serve_quota_is_still_per_address() {
        let store = StateStore::open_temp().unwrap();
        let issued = issue_self_serve_key(&store, ADDR, 2, 1).unwrap();
        let auth = Auth::disabled().with_self_serve(true);
        let mut headers = HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            format!("Bearer {}", issued.api_key).parse().unwrap(),
        );
        let caller = auth.authorize_with_store(&headers, &store).unwrap();
        let now = 1_700_000_000_000u64;
        assert_eq!(auth.consume_quota(&store, &caller, now).unwrap(), 1);
        assert_eq!(auth.consume_quota(&store, &caller, now).unwrap(), 2);
        assert!(matches!(
            auth.consume_quota(&store, &caller, now),
            Err(AuthError::QuotaExceeded { used: 2, quota: 2 })
        ));
        // Another key for the same address would share the same allowance.
        let same_address = Caller {
            tenant: "self-serve-2".into(),
            address: ADDR.into(),
            daily_quota: 2,
        };
        assert!(matches!(
            auth.consume_quota(&store, &same_address, now),
            Err(AuthError::QuotaExceeded { used: 2, quota: 2 })
        ));
    }
}
