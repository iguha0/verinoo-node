//! Read-only RPC surface for the Verinoo node (axum).
//!
//! Mutations (`POST /tx`) validate + enqueue to mempool; they never touch
//! state directly. Block/tx payloads are canonical JSON.

use axum::{
    body::{Body, Bytes},
    extract::{ConnectInfo, Path as AxumPath, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::sse::{Event, KeepAlive, Sse},
    response::{IntoResponse, Response},
    routing::{get, post},
    Extension, Json, Router,
};
use futures::stream::{self, Stream};
use std::collections::{BTreeSet, HashMap, VecDeque};
use std::convert::Infallible;
use std::io::{self, SeekFrom};
use std::net::SocketAddr;
use std::path::{Component, Path as FsPath, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncSeekExt};

use crate::auth::{Auth, AuthError, Caller, SelfServeError};
use crate::policy_gate;
use crate::readiness::{self, ReadinessInput};
use crate::Shared;
use verinoo_consensus::{verify_certificate, CommitCertificate};

pub async fn serve(
    shared: Arc<Shared>,
    net: verinoo_network::NetworkNode,
    port: u16,
    auth: Auth,
) -> Result<(), String> {
    let app = app(shared, net, auth);

    let addr = format!("0.0.0.0:{port}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| e.to_string())?;
    tracing::info!("rpc listening on {addr}");
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await
    .map_err(|e| e.to_string())
}

fn app(shared: Arc<Shared>, net: verinoo_network::NetworkNode, auth: Auth) -> Router {
    Router::new()
        .route("/status", get(status))
        .route("/health", get(health))
        .route("/ready", get(ready))
        .route("/blocks", get(blocks))
        .route("/blocks/since/:index", get(blocks_since))
        .route("/finalized/since/:index", get(finalized_since))
        .route("/mempool", get(mempool))
        .route("/account/:address", get(account))
        .route("/validators", get(validators))
        .route("/faucet", post(faucet))
        .route("/metrics", get(metrics))
        .route("/snapshot/manifest", get(snapshot_manifest))
        .route("/snapshot/files/*path", get(snapshot_file))
        .route("/tx", post(submit_tx))
        // ---- v1 product surface ----
        .route("/v1/models", get(v1_models))
        .route("/v1/input-key", get(v1_input_key))
        .route("/v1/keys", post(v1_issue_key))
        .route("/v1/infer", post(submit_tx))
        .route("/v1/infer/batch", post(v1_submit_batch))
        .route("/v1/tasks/:task_id", get(v1_task))
        .route("/v1/tasks/:task_id/events", get(v1_task_events))
        .route("/v1/verify/:task_id", get(v1_verify))
        .route("/v1/usage/:address", get(v1_usage))
        .route("/v1/usage/:address/csv", get(v1_usage_csv))
        .layer(axum::middleware::from_fn(cors))
        .with_state(shared)
        .layer(Extension(net))
        .layer(Extension(auth))
        .layer(Extension(KeyRateLimit::default()))
}

const KEY_RATE_LIMIT: usize = 5;
const KEY_RATE_WINDOW: Duration = Duration::from_secs(3_600);

/// Best-effort per-IP limiter for self-serve key issuance. State is process
/// local, so a restart clears it and replicas each keep their own window.
#[derive(Clone, Default)]
struct KeyRateLimit {
    inner: Arc<Mutex<HashMap<String, Vec<Instant>>>>,
}

impl KeyRateLimit {
    fn check(&self, key: &str) -> Result<(), u64> {
        let now = Instant::now();
        let mut map = self.inner.lock().expect("key rate limit lock");
        let entries = map.entry(key.to_string()).or_default();
        entries.retain(|t| now.duration_since(*t) < KEY_RATE_WINDOW);
        if entries.len() >= KEY_RATE_LIMIT {
            let wait = KEY_RATE_WINDOW.saturating_sub(now.duration_since(entries[0]));
            return Err(wait.as_secs().max(1));
        }
        entries.push(now);
        Ok(())
    }
}

fn request_ip(headers: &HeaderMap, peer: SocketAddr) -> String {
    if let Some(forwarded) = headers.get("x-forwarded-for").and_then(|v| v.to_str().ok()) {
        if let Some(first) = forwarded
            .split(',')
            .next()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            return first.to_string();
        }
    }
    if let Some(real) = headers.get("x-real-ip").and_then(|v| v.to_str().ok()) {
        let real = real.trim();
        if !real.is_empty() {
            return real.to_string();
        }
    }
    peer.ip().to_string()
}

/// Liveness probe. `ok` only means a head exists; use `/ready` for traffic.
async fn health(State(shared): State<Arc<Shared>>) -> Json<serde_json::Value> {
    let head = shared.head();
    let (height, ts) = match &head {
        Some(b) => (b.header.index, b.header.timestamp),
        None => (0, 0),
    };
    Json(serde_json::json!({
        "ok": head.is_some(),
        "chainId": shared.params.chain_id,
        "height": height,
        "headAgeMs": shared.now_ms().saturating_sub(ts),
        "validators": shared.params.validators.len(),
    }))
}

/// Readiness probe for load balancers and promotion checks. 200 only when the
/// node has a fresh head, is not applying catch-up blocks, and is not behind
/// the best peer observed by `VNN_SYNC_URLS`.
async fn ready(
    State(shared): State<Arc<Shared>>,
    Extension(net): Extension<verinoo_network::NetworkNode>,
) -> (StatusCode, Json<serde_json::Value>) {
    ready_report(&shared, shared.now_ms(), net.peer_count())
}

fn ready_report(
    shared: &Shared,
    now_ms: u64,
    peer_count: usize,
) -> (StatusCode, Json<serde_json::Value>) {
    let head = shared.head();
    let (serving, height, head_age_ms) = match &head {
        Some(block) => (
            true,
            block.header.index,
            now_ms.saturating_sub(block.header.timestamp),
        ),
        // No head means no age to report, not "as old as the epoch".
        None => (false, 0, 0),
    };
    let syncing = shared.syncing.load(Ordering::Relaxed);
    let validator_set_consistent = shared.validator_set_consistent();
    let decision = readiness::evaluate(ReadinessInput {
        serving,
        syncing,
        height,
        head_age_ms,
        block_time_ms: shared.block_time_ms,
        best_peer_height: shared.best_peer_height(),
        peer_lag_blocks: shared.peer_lag_blocks,
        validator_set_consistent,
    });
    let status = if decision.ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (
        status,
        Json(serde_json::json!({
            "ready": decision.ready,
            "serving": serving,
            "height": height,
            "headAgeMs": head_age_ms,
            "peerCount": peer_count,
            "syncing": syncing,
            "validatorSetHash": shared.validator_set_hash(),
            "validatorSetConsistent": validator_set_consistent,
            "reason": decision.reason,
        })),
    )
}

/// Permissive CORS so browser-based testnet clients (faucet, explorer) can
/// call the public read API directly.
async fn cors(
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::http::{header, HeaderValue, Method};
    let preflight = req.method() == Method::OPTIONS;
    let mut res = if preflight {
        axum::response::Response::new(axum::body::Body::empty())
    } else {
        next.run(req).await
    };
    let headers = res.headers_mut();
    headers.insert(
        header::ACCESS_CONTROL_ALLOW_ORIGIN,
        HeaderValue::from_static("*"),
    );
    headers.insert(
        header::ACCESS_CONTROL_ALLOW_METHODS,
        HeaderValue::from_static("GET, POST, OPTIONS"),
    );
    headers.insert(
        header::ACCESS_CONTROL_ALLOW_HEADERS,
        HeaderValue::from_static("content-type, authorization"),
    );
    res
}

async fn status(
    State(shared): State<Arc<Shared>>,
    Extension(net): Extension<verinoo_network::NetworkNode>,
) -> Json<serde_json::Value> {
    let head = shared.head();
    let now_ms = shared.now_ms();
    let (height, latest_hash, validator, head_age_ms) = match head {
        Some(b) => (
            b.header.index,
            b.header.hash,
            b.header.validator,
            now_ms.saturating_sub(b.header.timestamp),
        ),
        None => (0, String::new(), String::new(), 0),
    };
    let (finalized_height, finalized_hash) = if let Some(manager) = &shared.finality {
        let manager = manager.lock().await;
        (
            manager.finalized_height(),
            manager.finalized_hash().to_string(),
        )
    } else {
        (
            shared.store.meta_u64_or("finalizedHeight", 0),
            shared
                .store
                .get_meta("finalizedHash")
                .and_then(|value| value.as_str().map(str::to_owned))
                .unwrap_or_default(),
        )
    };
    Json(serde_json::json!({
        "chainId": shared.params.chain_id,
        "height": height,
        "latestHash": latest_hash,
        "validator": validator,
        "headAgeMs": head_age_ms,
        "peerCount": net.peer_count(),
        "syncing": shared.syncing.load(Ordering::Relaxed),
        "validatorSetHash": shared.validator_set_hash(),
        "validatorSetConsistent": shared.validator_set_consistent(),
        "finalizedHeight": finalized_height,
        "finalizedHash": finalized_hash,
        "consensus": shared.consensus_mode,
        "mempool": shared.mempool.lock().await.len(),
        "validators": shared.params.validators,
    }))
}

/// Blocks from `index + 1` up to the head, capped at 200 per request. A node
/// that fell behind replays these through the same validation and state-root
/// checks as gossip, so catch-up cannot bypass consensus.
async fn blocks_since(
    State(shared): State<Arc<Shared>>,
    axum::extract::Path(from): axum::extract::Path<u64>,
) -> Json<serde_json::Value> {
    let Some(head) = shared.head() else {
        return Json(serde_json::json!([]));
    };
    let start = from.saturating_add(1);
    if start > head.header.index {
        return Json(serde_json::json!([]));
    }
    let end = start.saturating_add(199).min(head.header.index);
    let mut out = Vec::new();
    for height in start..=end {
        let Some(hash) = shared
            .store
            .get_meta(&format!("blockheight:{height}"))
            .and_then(|v| v.as_str().map(String::from))
        else {
            break;
        };
        let Some(block) = shared.store.get_block_by_hash(&hash) else {
            break;
        };
        out.push(block);
    }
    Json(serde_json::json!(out))
}

/// Finalized `{block, certificate}` pairs strictly above `index`, ascending
/// and capped at 200. Stops at the first height without a stored block or
/// commit certificate, so a catch-up client never has to guess across a gap.
/// Each certificate is verified against the configured validator set and bound
/// to the served block before it leaves this node; an invalid entry ends the
/// range instead of exposing an unfinalized or forged block.
async fn finalized_since(
    State(shared): State<Arc<Shared>>,
    axum::extract::Path(from): axum::extract::Path<u64>,
) -> Json<serde_json::Value> {
    Json(serde_json::json!(finalized_pairs(&shared, from)))
}

fn finalized_pairs(shared: &Shared, from: u64) -> Vec<serde_json::Value> {
    let Some(head) = shared.head() else {
        return Vec::new();
    };
    let start = from.saturating_add(1);
    let end = start.saturating_add(199).min(head.header.index);
    if start > end {
        return Vec::new();
    }
    let mut out = Vec::new();
    for height in start..=end {
        let Some(hash) = shared
            .store
            .get_meta(&format!("blockheight:{height}"))
            .and_then(|v| v.as_str().map(String::from))
        else {
            break;
        };
        let Some(block_value) = shared.store.get_block_by_hash(&hash) else {
            break;
        };
        let block = match serde_json::from_value::<verinoo_exec::engine::Block>(block_value) {
            Ok(block) => block,
            Err(e) => {
                tracing::warn!(height = height, error = %e, "finalized block is malformed");
                break;
            }
        };
        if block.header.index != height || block.header.hash != hash {
            tracing::warn!(
                height = height,
                "finalized block does not match its height index"
            );
            break;
        }
        let Some(cert_value) = shared.store.get_meta(&format!("commit:{height}")) else {
            break;
        };
        let certificate = match serde_json::from_value::<CommitCertificate>(cert_value) {
            Ok(certificate) => certificate,
            Err(e) => {
                tracing::warn!(height = height, error = %e, "finalized certificate is malformed");
                break;
            }
        };
        let invalid = if certificate.height != height {
            Some(format!(
                "certificate height {} does not match {height}",
                certificate.height
            ))
        } else if certificate.block_hash != block.header.hash {
            Some("certificate does not bind the served block".to_string())
        } else if certificate.chain_id != shared.params.chain_id {
            Some("certificate chain id mismatch".to_string())
        } else if let Err(e) = verify_certificate(&certificate, &shared.params.validators) {
            Some(e.to_string())
        } else {
            None
        };
        if let Some(reason) = invalid {
            tracing::warn!(
                height = height,
                reason = %reason,
                "dropping invalid finalized certificate"
            );
            break;
        }
        out.push(serde_json::json!({
            "block": block,
            "certificate": certificate,
        }));
    }
    out
}

async fn blocks(State(shared): State<Arc<Shared>>) -> Json<serde_json::Value> {
    // Walk back from head via previousHash.
    let mut out = Vec::new();
    if let Some(cur) = shared.head() {
        out.push(serde_json::to_value(&cur).expect("block serialize"));
        let mut hash = cur.header.previous_hash.clone();
        for _ in 0..19 {
            match shared.store.get_block_by_hash(&hash) {
                Some(prev_value) => {
                    let prev_header_hash = prev_value["header"]["hash"]
                        .as_str()
                        .unwrap_or_default()
                        .to_string();
                    out.push(prev_value);
                    hash = prev_header_hash;
                }
                None => break,
            }
        }
    }
    Json(serde_json::json!(out))
}

async fn account(
    State(shared): State<Arc<Shared>>,
    axum::extract::Path(address): axum::extract::Path<String>,
) -> Json<serde_json::Value> {
    match shared.store.get_account(&address) {
        Some(a) => Json(serde_json::to_value(&a).unwrap()),
        None => Json(serde_json::json!({ "error": "not found" })),
    }
}

async fn validators(State(shared): State<Arc<Shared>>) -> Json<serde_json::Value> {
    let vals = shared.store.active_validators();
    Json(serde_json::json!(vals
        .iter()
        .map(|v| serde_json::json!({
            "address": v.address,
            "stakedAmount": v.staked_amount,
            "genesis": v.genesis,
        }))
        .collect::<Vec<_>>()))
}

/// Testnet faucet: drip tokens to any address. Rate-limited per address.
/// The drip is a signed transfer from the genesis-funded faucet account, so
/// it replicates through the block gossip like any other transaction.
async fn faucet(
    State(shared): State<Arc<Shared>>,
    Extension(net): Extension<verinoo_network::NetworkNode>,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, Json<serde_json::Value>)> {
    const DRIP_AMOUNT: i64 = 10_000;
    const COOLDOWN_MS: u64 = 60_000;

    let addr = body["address"].as_str().unwrap_or("").to_string();
    if addr.is_empty() || !addr.starts_with("ai_") {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "missing or invalid address" })),
        ));
    }

    // Rate limit: one drip per address per cooldown period.
    {
        let key = format!("faucet:{}", addr);
        let now_ms = shared.now_ms();
        if let Some(last) = shared.store.get_meta(&key) {
            if let Some(ts) = last.as_u64() {
                if now_ms.saturating_sub(ts) < COOLDOWN_MS {
                    return Err((
                        axum::http::StatusCode::TOO_MANY_REQUESTS,
                        Json(serde_json::json!({
                            "error": "cooldown active",
                            "retryAfterMs": COOLDOWN_MS - (now_ms - ts)
                        })),
                    ));
                }
            }
        }
        shared.store.set_meta(&key, &serde_json::json!(now_ms));
    }

    let balance = shared.store.account_or_default(&addr).balance;
    let nonce = shared
        .store
        .account_or_default(&shared.faucet.address)
        .nonce
        + 1;
    let value = crate::wallet::build_signed_transfer(
        &shared.faucet.signing,
        &shared.faucet.address,
        &addr,
        DRIP_AMOUNT as u64,
        nonce,
    )
    .map_err(|e| {
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e })),
        )
    })?;
    let tx: verinoo_types::tx::TxEnvelope = serde_json::from_value(value).map_err(|e| {
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
    })?;

    let base_fee = shared
        .store
        .meta_u64_or("baseFee", shared.params.initial_base_fee);
    let validator = verinoo_exec::Validator {
        store: &shared.store,
        base_fee,
    };
    if let Err(e) = validator.validate_transaction(&tx) {
        return Err((
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("faucet tx invalid: {e}") })),
        ));
    }

    let payload = serde_json::to_string(&tx).expect("faucet tx serialize");
    shared.mempool.lock().await.push(tx.clone());
    net.publish_tx(&payload);

    Ok(Json(serde_json::json!({
        "queued": true,
        "dripped": DRIP_AMOUNT,
        "address": addr,
        "balanceBefore": balance,
        "txId": tx.tx_id
    })))
}

async fn mempool(State(shared): State<Arc<Shared>>) -> Json<serde_json::Value> {
    let mp = shared.mempool.lock().await;
    Json(serde_json::json!({ "count": mp.len() }))
}

async fn metrics(State(shared): State<Arc<Shared>>) -> Json<serde_json::Value> {
    let head = shared.head();
    let (height, latest_hash, ts) = match &head {
        Some(b) => (b.header.index, b.header.hash.clone(), b.header.timestamp),
        None => (0, String::new(), 0),
    };
    Json(serde_json::json!({
        "height": height,
        "latestHash": latest_hash,
        "headAgeMs": shared.now_ms().saturating_sub(ts),
        "mempoolSize": shared.mempool.lock().await.len(),
        "validatorCount": shared.store.validator_addresses().len(),
        "selfServeKeys": shared
            .store
            .iter_prefix::<serde_json::Value>("meta:selfserve:key:")
            .len(),
        "chainId": shared.params.chain_id,
        "blockTimeMs": shared.params.block_time_ms,
    }))
}

fn auth_error(e: AuthError) -> (StatusCode, Json<serde_json::Value>) {
    (
        e.status(),
        Json(serde_json::json!({ "error": e.message() })),
    )
}

#[derive(Debug)]
struct AcceptedTx {
    tx_id: String,
    task_id: Option<String>,
}

#[derive(Debug)]
enum SubmitError {
    Auth(AuthError),
    Invalid(String),
    Policy(policy_gate::GateError),
    Conflict(String),
    Full(String),
    Internal(String),
}

impl SubmitError {
    fn status(&self) -> StatusCode {
        match self {
            Self::Auth(error) => error.status(),
            Self::Invalid(_) => StatusCode::BAD_REQUEST,
            Self::Policy(_) => StatusCode::UNPROCESSABLE_ENTITY,
            Self::Conflict(_) => StatusCode::CONFLICT,
            Self::Full(_) => StatusCode::TOO_MANY_REQUESTS,
            Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    fn message(&self) -> String {
        match self {
            Self::Auth(error) => error.message(),
            Self::Invalid(message)
            | Self::Conflict(message)
            | Self::Full(message)
            | Self::Internal(message) => message.clone(),
            Self::Policy(error) => error.message(),
        }
    }

    fn policy_code(&self) -> Option<&'static str> {
        match self {
            Self::Policy(error) => Some(error.code()),
            _ => None,
        }
    }

    fn into_rejection(self) -> (StatusCode, Json<serde_json::Value>) {
        let status = self.status();
        let message = self.message();
        let policy_code = self.policy_code();
        let mut body = serde_json::json!({ "error": message });
        if let Some(code) = policy_code {
            body["reason"] = serde_json::json!(policy_gate::POLICY_REASON);
            body["policyReason"] = serde_json::json!(code);
        }
        (status, Json(body))
    }
}

#[derive(Debug)]
struct BatchAccount {
    next_nonce: u64,
    available_balance: i128,
}

#[derive(Debug, Default)]
struct BatchOverlay {
    accounts: HashMap<String, BatchAccount>,
}

impl BatchOverlay {
    fn account(&mut self, store: &verinoo_state::StateStore, address: &str) -> &mut BatchAccount {
        self.accounts.entry(address.to_string()).or_insert_with(|| {
            let account = store.account_or_default(address);
            BatchAccount {
                next_nonce: account.nonce + 1,
                available_balance: account.balance as i128,
            }
        })
    }

    fn record(&mut self, address: &str, value: u64, fee: u64) {
        if let Some(account) = self.accounts.get_mut(address) {
            account.next_nonce += 1;
            account.available_balance -= value as i128 + fee as i128;
        }
    }
}

fn is_stateful_validation_error(error: &str) -> bool {
    error.starts_with("bad nonce") || error == "insufficient balance for value + gas"
}

fn transaction_fee(tx: &verinoo_types::tx::TxEnvelope, base_fee: u64) -> Result<u64, String> {
    let data = serde_json::to_value(&tx.data).unwrap();
    let multiplier = match verinoo_exec::gas::policy_of_tx(&data) {
        Some(policy) => verinoo_exec::gas::policy_multiplier(Some(policy))?,
        None => 1,
    };
    Ok(verinoo_exec::gas::gas_cost_for(&data) * base_fee * multiplier)
}

fn transaction_gas(tx: &verinoo_types::tx::TxEnvelope) -> Result<u64, String> {
    let data = serde_json::to_value(&tx.data).map_err(|e| e.to_string())?;
    let multiplier = match verinoo_exec::gas::policy_of_tx(&data) {
        Some(policy) => verinoo_exec::gas::policy_multiplier(Some(policy))?,
        None => 1,
    };
    verinoo_exec::gas::gas_cost_for(&data)
        .checked_mul(multiplier)
        .ok_or_else(|| "transaction gas overflow".to_string())
}

fn validate_batch_item(
    store: &verinoo_state::StateStore,
    base_fee: u64,
    overlay: &mut BatchOverlay,
    tx: &verinoo_types::tx::TxEnvelope,
) -> Result<u64, String> {
    // Consensus validation runs stateful checks last, so a nonce/balance
    // rejection means the stateless checks already passed. Re-evaluate only
    // the stateful rules against the batch overlay.
    let validator = verinoo_exec::Validator { store, base_fee };
    match validator.validate_transaction(tx) {
        Ok(()) => {}
        Err(error) if is_stateful_validation_error(&error) => {}
        Err(error) => return Err(error),
    }

    let account = overlay.account(store, &tx.from);
    if tx.nonce != account.next_nonce {
        return Err(format!("bad nonce (have {})", account.next_nonce - 1));
    }
    let fee = transaction_fee(tx, base_fee)?;
    let total = tx.value as i128 + fee as i128;
    if account.available_balance < total {
        return Err("insufficient balance for value + gas".into());
    }
    Ok(fee)
}

fn check_caller(
    caller: Option<&Caller>,
    tx: &verinoo_types::tx::TxEnvelope,
) -> Result<(), SubmitError> {
    if let Some(caller) = caller {
        if caller.address != tx.from {
            return Err(SubmitError::Auth(AuthError::AddressMismatch));
        }
    }
    Ok(())
}

fn check_inference_submission(
    shared: &Shared,
    tx: &verinoo_types::tx::TxEnvelope,
) -> Result<(), SubmitError> {
    if tx.data.r#type != verinoo_types::tx::TxKind::SubmitInference {
        return Ok(());
    }

    let has_encrypted = tx
        .data
        .data
        .get("encryptedInput")
        .map(|v| !v.is_null())
        .unwrap_or(false);
    let has_plain = tx
        .data
        .data
        .get("input")
        .and_then(|v| v.as_str())
        .map(|s| !s.is_empty())
        .unwrap_or(false);
    let bad = |msg: &str| SubmitError::Invalid(msg.to_string());
    if has_encrypted && has_plain {
        return Err(bad("submit either input or encryptedInput, not both"));
    }
    match verinoo_exec::engine::parse_sampling(tx.data.data.get("sampling")) {
        Err(e) => return Err(SubmitError::Invalid(e)),
        Ok(Some(spec)) => {
            let target = tx.data.data["targetModel"].as_str().unwrap_or("");
            if let Some(model) = verinoo_exec::engine::ExecModel::resolve(target) {
                if let Err(e) = model.validate_sampling(&spec) {
                    return Err(SubmitError::Invalid(e));
                }
            }
        }
        Ok(None) => {}
    }
    // Commitment-only submissions are allowed for the manual-claim path.
    if encrypted_input_required() && !has_encrypted {
        return Err(bad(
            "this node requires encrypted input; fetch /v1/input-key and send encryptedInput",
        ));
    }
    if has_encrypted && verinoo_crypto::input_key().is_none() {
        return Err(bad("this node has no input key configured"));
    }
    policy_gate::check_tx(&shared.request_policy, tx).map_err(SubmitError::Policy)?;
    Ok(())
}

async fn admit_tx(
    shared: &Shared,
    net: &verinoo_network::NetworkNode,
    auth: &Auth,
    caller: Option<&Caller>,
    tx: verinoo_types::tx::TxEnvelope,
) -> Result<AcceptedTx, SubmitError> {
    let mut mp = shared.mempool.lock().await;
    if mp.iter().any(|t| t.tx_id == tx.tx_id) {
        return Err(SubmitError::Conflict("already in mempool".into()));
    }
    if mp.len() >= crate::MAX_MEMPOOL_TRANSACTIONS {
        return Err(SubmitError::Full(format!(
            "mempool is full ({} transactions); retry later",
            crate::MAX_MEMPOOL_TRANSACTIONS
        )));
    }
    if let Some(caller) = caller {
        if tx.data.r#type == verinoo_types::tx::TxKind::SubmitInference {
            auth.consume_quota(&shared.store, caller, shared.now_ms())
                .map_err(SubmitError::Auth)?;
        }
    }
    let payload = serde_json::to_string(&tx).map_err(|e| SubmitError::Internal(e.to_string()))?;
    let task_id = verinoo_exec::engine::task_id_for(&tx);
    mp.push(tx.clone());
    drop(mp);

    // Without this, a transaction submitted to a non-validator node never
    // leaves the process and is silently lost.
    net.publish_tx(&payload);

    Ok(AcceptedTx {
        tx_id: tx.tx_id,
        task_id,
    })
}

async fn submit_one(
    shared: &Shared,
    net: &verinoo_network::NetworkNode,
    auth: &Auth,
    caller: Option<&Caller>,
    tx: verinoo_types::tx::TxEnvelope,
) -> Result<AcceptedTx, SubmitError> {
    check_caller(caller, &tx)?;

    // Full validation at submission time — fail fast with the exact reason.
    // (Stateful rules like nonce/balance are evaluated against current state;
    // they are re-checked at block production per reference semantics.)
    let base_fee = shared
        .store
        .meta_u64_or("baseFee", shared.params.initial_base_fee);
    let validator = verinoo_exec::Validator {
        store: &shared.store,
        base_fee,
    };
    validator
        .validate_transaction(&tx)
        .map_err(SubmitError::Invalid)?;

    check_inference_submission(shared, &tx)?;

    admit_tx(shared, net, auth, caller, tx).await
}

async fn submit_batch_item(
    shared: &Shared,
    net: &verinoo_network::NetworkNode,
    auth: &Auth,
    caller: Option<&Caller>,
    base_fee: u64,
    overlay: &mut BatchOverlay,
    tx: verinoo_types::tx::TxEnvelope,
) -> Result<(AcceptedTx, u64), SubmitError> {
    check_caller(caller, &tx)?;
    let fee =
        validate_batch_item(&shared.store, base_fee, overlay, &tx).map_err(SubmitError::Invalid)?;
    check_inference_submission(shared, &tx)?;
    let accepted = admit_tx(shared, net, auth, caller, tx).await?;
    Ok((accepted, fee))
}

async fn submit_tx(
    State(shared): State<Arc<Shared>>,
    Extension(net): Extension<verinoo_network::NetworkNode>,
    Extension(auth): Extension<Auth>,
    headers: HeaderMap,
    Json(tx): Json<verinoo_types::tx::TxEnvelope>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, Json<serde_json::Value>)> {
    // Authentication is opt-in: a node with no key file accepts any funded
    // sender (public demo). When keys are configured, a key may only submit for
    // the address it was issued to.
    let caller = if auth.is_enabled() {
        let caller = auth
            .authorize_with_store(&headers, &shared.store)
            .map_err(auth_error)?;
        tracing::debug!(
            tenant = %caller.tenant,
            address = %caller.address,
            "api key authorized"
        );
        Some(caller)
    } else {
        None
    };

    let accepted = submit_one(&shared, &net, &auth, caller.as_ref(), tx)
        .await
        .map_err(SubmitError::into_rejection)?;

    Ok(Json(serde_json::json!({
        "accepted": true,
        "txId": accepted.tx_id,
        "taskId": accepted.task_id,
    })))
}

/// Transport cap for batch size. The real capacity limit is gas: a block has
/// `BLOCK_GAS_LIMIT` (1000) gas and `submitInference` costs 60 (sampled 120,
/// full-zk 240), so at most 16 optimistic inference items fit in one batch.
const MAX_BATCH_TRANSACTIONS: usize = 32;

fn invalid_request(message: &str) -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({ "error": message })),
    )
}

async fn v1_submit_batch(
    State(shared): State<Arc<Shared>>,
    Extension(net): Extension<verinoo_network::NetworkNode>,
    Extension(auth): Extension<Auth>,
    headers: HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let caller = if auth.is_enabled() {
        Some(
            auth.authorize_with_store(&headers, &shared.store)
                .map_err(auth_error)?,
        )
    } else {
        None
    };

    let Some(transactions) = body.get("transactions").and_then(|value| value.as_array()) else {
        return Err(invalid_request("missing transactions array"));
    };
    if transactions.is_empty() {
        return Err(invalid_request("transactions must not be empty"));
    }
    if transactions.len() > MAX_BATCH_TRANSACTIONS {
        return Err(invalid_request("transactions exceeds the 32 item limit"));
    }

    // Reject the whole batch before admitting anything when its gas cannot fit
    // in one block. Otherwise the producer silently drops gas-exhausted items
    // and the caller sees "accepted" for work that never executes.
    let mut total_gas = 0u64;
    for value in transactions {
        if let Ok(tx) = serde_json::from_value::<verinoo_types::tx::TxEnvelope>(value.clone()) {
            if let Ok(gas) = transaction_gas(&tx) {
                total_gas = total_gas.saturating_add(gas);
            }
        }
    }
    let block_gas_limit = verinoo_exec::gas::block_gas_limit();
    if total_gas > block_gas_limit {
        return Err(invalid_request(&format!(
            "batch gas {total_gas} exceeds the per-block gas limit of {block_gas_limit}; split into smaller batches"
        )));
    }

    let mut results = Vec::with_capacity(transactions.len());
    let mut accepted_count = 0usize;
    let base_fee = shared
        .store
        .meta_u64_or("baseFee", shared.params.initial_base_fee);
    let mut overlay = BatchOverlay::default();
    for (index, value) in transactions.iter().enumerate() {
        let hinted_tx_id = value
            .get("txId")
            .and_then(|tx_id| tx_id.as_str())
            .unwrap_or("")
            .to_string();
        let tx = match serde_json::from_value::<verinoo_types::tx::TxEnvelope>(value.clone()) {
            Ok(tx) => tx,
            Err(_) => {
                results.push(serde_json::json!({
                    "index": index,
                    "accepted": false,
                    "error": "malformed transaction",
                    "txId": hinted_tx_id,
                }));
                continue;
            }
        };
        let tx_id = tx.tx_id.clone();
        let from = tx.from.clone();
        let value = tx.value;
        match submit_batch_item(
            &shared,
            &net,
            &auth,
            caller.as_ref(),
            base_fee,
            &mut overlay,
            tx,
        )
        .await
        {
            Ok((accepted, fee)) => {
                overlay.record(&from, value, fee);
                accepted_count += 1;
                results.push(serde_json::json!({
                    "index": index,
                    "accepted": true,
                    "txId": accepted.tx_id,
                    "taskId": accepted.task_id,
                }));
            }
            Err(error) => {
                let mut result = serde_json::json!({
                    "index": index,
                    "accepted": false,
                    "error": error.message(),
                    "txId": tx_id,
                });
                if let Some(code) = error.policy_code() {
                    result["reason"] = serde_json::json!(policy_gate::POLICY_REASON);
                    result["policyReason"] = serde_json::json!(code);
                }
                results.push(result);
            }
        }
    }

    Ok(Json(serde_json::json!({
        "accepted": accepted_count,
        "rejected": results.len() - accepted_count,
        "results": results,
    })))
}

// ============================ v1 product API ============================

fn not_found(msg: &str) -> (axum::http::StatusCode, Json<serde_json::Value>) {
    (
        axum::http::StatusCode::NOT_FOUND,
        Json(serde_json::json!({ "error": msg })),
    )
}

/// Models this node can actually execute, with the exact weights identity.
async fn v1_models() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "models": verinoo_exec::engine::execute_models(),
        "modelSetHash": verinoo_runtime::global().hash(),
        "transformerSetHash": verinoo_runtime::transformer::transformer_global().hash(),
        "gptSetHash": verinoo_runtime::gpt::gpt_global().hash(),
        "runtime": "deterministic-q16.16",
        "verification": "recompute",
    }))
}

fn encrypted_input_required() -> bool {
    std::env::var("VNN_REQUIRE_ENCRYPTED_INPUT")
        .map(|v| v == "1" || v == "true")
        .unwrap_or(false)
}

/// Public key clients use to encrypt inference inputs. When `publicKey` is
/// present, clients should send `encryptedInput` instead of plaintext `input`.
async fn v1_input_key() -> Json<serde_json::Value> {
    let key = verinoo_crypto::input_public_key_hex();
    Json(serde_json::json!({
        "publicKey": key,
        "required": encrypted_input_required(),
        "algorithm": verinoo_crypto::ALGORITHM,
    }))
}

/// Self-serve API key issuance. Disabled unless `VNN_SELF_SERVE_KEYS=1`.
/// One active key per address; the raw key is returned exactly once.
async fn v1_issue_key(
    State(shared): State<Arc<Shared>>,
    Extension(auth): Extension<Auth>,
    Extension(rate): Extension<KeyRateLimit>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    if !auth.self_serve_enabled() {
        return Err(not_found("self-serve key issuance is disabled"));
    }
    let ip = request_ip(&headers, peer);
    if let Err(retry_after) = rate.check(&ip) {
        return Err((
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::json!({
                "error": "self-serve key rate limit exceeded (5 per hour per IP)",
                "retryAfterSecs": retry_after,
            })),
        ));
    }
    let address = body["address"].as_str().unwrap_or("").trim();
    if address.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "missing address" })),
        ));
    }
    match auth.issue_self_serve_key(&shared.store, address, shared.now_ms()) {
        Ok(issued) => Ok(Json(serde_json::json!({
            "apiKey": issued.api_key,
            "tenant": issued.tenant,
            "address": issued.address,
            "dailyQuota": issued.daily_quota,
            "createdAt": issued.created_at,
            "note": "store this key; it is shown only once",
        }))),
        Err(SelfServeError::InvalidAddress) => Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "address must be `ai_` followed by 40 hex characters"
            })),
        )),
        Err(SelfServeError::AlreadyIssued { tenant, address }) => Err((
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "error": "address already has a self-serve key",
                "tenant": tenant,
                "address": address,
            })),
        )),
    }
}

fn redacted_task_json(
    task: &verinoo_state::models::InferenceTask,
) -> Result<serde_json::Value, serde_json::Error> {
    let input_commitment = task.input_commitment.clone();
    let encrypted = task.encrypted_input.is_some();
    let mut task_json = serde_json::to_value(task)?;
    if let Some(obj) = task_json.as_object_mut() {
        // Inputs and ciphertext are not part of the public task view. The
        // commitment is enough to match a receipt; verification recomputes
        // server-side.
        obj.remove("input");
        obj.remove("encryptedInput");
        obj.insert(
            "inputHash".into(),
            serde_json::Value::String(input_commitment),
        );
        obj.insert("encrypted".into(), serde_json::Value::Bool(encrypted));
    }
    Ok(task_json)
}

async fn v1_task(
    State(shared): State<Arc<Shared>>,
    axum::extract::Path(task_id): axum::extract::Path<String>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, Json<serde_json::Value>)> {
    let task = shared
        .store
        .get_task(&task_id)
        .ok_or_else(|| not_found("task not found"))?;
    let receipt = verinoo_exec::engine::receipt_for(&shared.store, &task_id);
    let task_json = redacted_task_json(&task).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
    })?;
    Ok(Json(serde_json::json!({
        "task": task_json,
        "receipt": receipt,
        "anchored": receipt.is_some(),
    })))
}

const TASK_EVENT_POLL: Duration = Duration::from_secs(1);
const TASK_EVENT_TIMEOUT: Duration = Duration::from_secs(60);

struct TaskEventState {
    shared: Arc<Shared>,
    task_id: String,
    ticker: tokio::time::Interval,
    deadline: Instant,
    last_status: Option<(String, Option<String>, u64)>,
    queue: VecDeque<Event>,
    finished: bool,
    /// Held for the stream lifetime so the global stream cap cannot be
    /// exceeded by long-lived connections.
    _permit: Option<tokio::sync::OwnedSemaphorePermit>,
}

fn task_status_event(shared: &Shared, task: &verinoo_state::models::InferenceTask) -> Event {
    let height = if task.executed_at > 0 {
        task.executed_at
    } else {
        shared.head().map(|block| block.header.index).unwrap_or(0)
    };
    Event::default().event("status").data(
        serde_json::json!({
            "taskId": task.task_id,
            "status": task.status,
            "height": height,
            "resultHash": task.result_hash,
            "updatedAt": shared.now_ms(),
        })
        .to_string(),
    )
}

fn task_result_event(state: &mut TaskEventState, task: &verinoo_state::models::InferenceTask) {
    let receipt = verinoo_exec::engine::receipt_for(&state.shared.store, &state.task_id);
    let task_json =
        redacted_task_json(task).unwrap_or_else(|_| serde_json::json!({ "taskId": state.task_id }));
    state.queue.push_back(
        Event::default().event("result").data(
            serde_json::json!({
                "taskId": state.task_id,
                "task": task_json,
                "receipt": receipt,
            })
            .to_string(),
        ),
    );
    state.queue.push_back(
        Event::default()
            .event("done")
            .data(serde_json::json!({ "taskId": state.task_id }).to_string()),
    );
    state.finished = true;
}

fn task_event_stream(
    shared: Arc<Shared>,
    task_id: String,
    poll: Duration,
    timeout: Duration,
    permit: Option<tokio::sync::OwnedSemaphorePermit>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let state = TaskEventState {
        shared,
        task_id,
        ticker: tokio::time::interval(poll),
        deadline: Instant::now() + timeout,
        last_status: None,
        queue: VecDeque::new(),
        finished: false,
        _permit: permit,
    };
    let stream = stream::unfold(state, |mut state| async move {
        loop {
            if let Some(event) = state.queue.pop_front() {
                return Some((Ok(event), state));
            }
            if state.finished {
                return None;
            }
            if Instant::now() >= state.deadline {
                state.queue.push_back(
                    Event::default().event("timeout").data(
                        serde_json::json!({
                            "taskId": state.task_id,
                            "reason": "timeout",
                        })
                        .to_string(),
                    ),
                );
                state.queue.push_back(
                    Event::default()
                        .event("done")
                        .data(serde_json::json!({ "taskId": state.task_id }).to_string()),
                );
                state.finished = true;
                continue;
            }

            state.ticker.tick().await;

            if let Some(task) = state.shared.store.get_task(&state.task_id) {
                let status_key = (
                    task.status.clone(),
                    task.result_hash.clone(),
                    task.executed_at,
                );
                if state.last_status.as_ref() != Some(&status_key) {
                    state.last_status = Some(status_key);
                    state
                        .queue
                        .push_back(task_status_event(&state.shared, &task));
                }
                if task.status == "completed" || task.status == "failed" {
                    task_result_event(&mut state, &task);
                }
            }
        }
    });
    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keep-alive"),
    )
}

async fn v1_task_events(
    State(shared): State<Arc<Shared>>,
    axum::extract::Path(task_id): axum::extract::Path<String>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, (StatusCode, Json<serde_json::Value>)>
{
    if shared.store.get_task(&task_id).is_none() {
        return Err(not_found("task not found"));
    }
    let permit = shared
        .task_streams
        .clone()
        .try_acquire_owned()
        .map_err(|_| {
            (
                StatusCode::TOO_MANY_REQUESTS,
                Json(serde_json::json!({
                    "error": format!(
                        "too many active task event streams (limit {})",
                        crate::MAX_ACTIVE_TASK_STREAMS
                    )
                })),
            )
        })?;
    Ok(task_event_stream(
        shared,
        task_id,
        TASK_EVENT_POLL,
        TASK_EVENT_TIMEOUT,
        Some(permit),
    ))
}

/// Independent verification: re-run the model and compare against the value
/// the chain committed for this task.
async fn v1_verify(
    State(shared): State<Arc<Shared>>,
    axum::extract::Path(task_id): axum::extract::Path<String>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, Json<serde_json::Value>)> {
    let task = shared
        .store
        .get_task(&task_id)
        .ok_or_else(|| not_found("task not found"))?;

    let Some(model) = verinoo_exec::engine::ExecModel::resolve(&task.target_model) else {
        return Ok(Json(serde_json::json!({
            "taskId": task_id,
            "verified": false,
            "reason": "this node has no deterministic runtime for this model",
            "modelId": task.target_model,
        })));
    };
    if task.status != "completed" {
        return Ok(Json(serde_json::json!({
            "taskId": task_id,
            "verified": false,
            "reason": format!("task is `{}`, not completed", task.status),
        })));
    }

    let plaintext = match verinoo_exec::engine::plaintext_input(&task) {
        Ok(text) => text,
        Err(reason) => {
            return Ok(Json(serde_json::json!({
                "taskId": task_id,
                "verified": false,
                "reason": reason,
                "modelId": model.model_id(),
                "encrypted": task.encrypted_input.is_some(),
            })))
        }
    };
    let effective_sampling = model.effective_sampling_json(task.sampling.as_ref());
    let recomputed = match model.result_json_with_sampling(&plaintext, task.sampling.as_ref()) {
        Ok(output) => output,
        Err(reason) => {
            return Ok(Json(serde_json::json!({
                "taskId": task_id,
                "verified": false,
                "reason": reason,
                "modelId": model.model_id(),
            })))
        }
    };
    let recomputed_hash = verinoo_model::sha256_hex(&recomputed);
    let on_chain = task.result_hash.clone().unwrap_or_default();
    let receipt = verinoo_exec::engine::receipt_for(&shared.store, &task_id);
    let matches = recomputed_hash == on_chain;
    let receipt_output_matches = receipt
        .as_ref()
        .and_then(|r| r.get("outputHash"))
        .and_then(|v| v.as_str())
        == Some(recomputed_hash.as_str());
    let expected_sampling_hash = effective_sampling
        .as_ref()
        .and_then(|value| verinoo_types::canonical::canonical_json(value).ok())
        .map(|canonical| verinoo_model::sha256_hex(&canonical));
    let receipt_sampling_hash = receipt
        .as_ref()
        .and_then(|r| r.get("samplingHash"))
        .and_then(|v| v.as_str());
    let sampling_matches = match receipt_sampling_hash {
        None => true,
        Some(actual) => expected_sampling_hash.as_deref() == Some(actual),
    };
    let receipt_matches = receipt_output_matches && sampling_matches;

    Ok(Json(serde_json::json!({
        "taskId": task_id,
        "verified": matches && receipt_matches,
        "outputMatches": matches,
        "anchored": receipt.is_some(),
        "recomputedHash": recomputed_hash,
        "onChainHash": on_chain,
        "output": recomputed,
        "modelId": model.model_id(),
        "weightsHash": model.weights_hash(),
        "sampling": effective_sampling,
        "samplingHash": expected_sampling_hash,
        "receipt": receipt,
    })))
}

fn usage_snapshot(shared: &Shared, address: &str) -> serde_json::Value {
    let tasks: Vec<verinoo_state::models::InferenceTask> = shared.store.iter_prefix("task:");
    let mine: Vec<_> = tasks
        .into_iter()
        .filter(|t| t.requester == address)
        .collect();
    let completed = mine.iter().filter(|t| t.status == "completed").count();
    let pending = mine
        .iter()
        .filter(|t| t.status == "pending" || t.status == "assigned")
        .count();
    let failed = mine.iter().filter(|t| t.status == "failed").count();
    let fees: u64 = mine
        .iter()
        .filter(|t| t.status == "completed")
        .map(|t| t.max_fee)
        .sum();
    serde_json::json!({
        "address": address,
        "tasks": mine.len(),
        "completed": completed,
        "pending": pending,
        "failed": failed,
        "feesCommitted": fees,
        "balance": shared.store.account_or_default(address).balance,
    })
}

async fn v1_usage(
    State(shared): State<Arc<Shared>>,
    Extension(auth): Extension<Auth>,
    headers: HeaderMap,
    axum::extract::Path(address): axum::extract::Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    if auth.is_enabled() {
        let caller = auth
            .authorize_with_store(&headers, &shared.store)
            .map_err(auth_error)?;
        if caller.address != address {
            return Err(auth_error(AuthError::AddressMismatch));
        }
    }
    Ok(Json(usage_snapshot(&shared, &address)))
}

/// Metering export for invoicing.
async fn v1_usage_csv(
    State(shared): State<Arc<Shared>>,
    Extension(auth): Extension<Auth>,
    headers: HeaderMap,
    axum::extract::Path(address): axum::extract::Path<String>,
) -> Result<impl axum::response::IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    if auth.is_enabled() {
        let caller = auth
            .authorize_with_store(&headers, &shared.store)
            .map_err(auth_error)?;
        if caller.address != address {
            return Err(auth_error(AuthError::AddressMismatch));
        }
    }
    let u = usage_snapshot(&shared, &address);
    let body = format!(
        "address,tasks,completed,pending,failed,feesCommitted,balance\n{},{},{},{},{},{},{}\n",
        u["address"].as_str().unwrap_or(""),
        u["tasks"],
        u["completed"],
        u["pending"],
        u["failed"],
        u["feesCommitted"],
        u["balance"],
    );
    Ok(([(axum::http::header::CONTENT_TYPE, "text/csv")], body))
}

const SNAPSHOT_READ_CHUNK: usize = 64 * 1024;
const MAX_SNAPSHOT_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;

/// Canonicalize the configured snapshot root once. A symlinked
/// `VNN_SNAPSHOT_DIR` is resolved here so the manifest and file endpoints
/// behave the same; every component below the resolved root is still checked
/// for symlinks.
async fn resolve_snapshot_root(root: &FsPath) -> Result<PathBuf, SnapshotPathError> {
    let resolved = tokio::fs::canonicalize(root)
        .await
        .map_err(|_| SnapshotPathError::NotFound)?;
    let meta = tokio::fs::metadata(&resolved)
        .await
        .map_err(|_| SnapshotPathError::NotFound)?;
    if !meta.is_dir() {
        return Err(SnapshotPathError::NotFound);
    }
    Ok(resolved)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SnapshotManifestError {
    NotFound,
    TooLarge,
    Invalid,
}

async fn load_snapshot_manifest(
    root: &FsPath,
) -> Result<verinoo_state::snapshot::SnapshotManifest, SnapshotManifestError> {
    let path = root.join("manifest.json");
    let meta = tokio::fs::symlink_metadata(&path)
        .await
        .map_err(|_| SnapshotManifestError::NotFound)?;
    if meta.file_type().is_symlink() || !meta.is_file() {
        return Err(SnapshotManifestError::NotFound);
    }
    if meta.len() > MAX_SNAPSHOT_MANIFEST_BYTES {
        return Err(SnapshotManifestError::TooLarge);
    }
    let bytes = tokio::fs::read(&path)
        .await
        .map_err(|_| SnapshotManifestError::NotFound)?;
    let manifest: verinoo_state::snapshot::SnapshotManifest =
        serde_json::from_slice(&bytes).map_err(|_| SnapshotManifestError::Invalid)?;
    for file in &manifest.files {
        verinoo_state::snapshot::validate_snapshot_path(&file.path)
            .map_err(|_| SnapshotManifestError::Invalid)?;
    }
    Ok(manifest)
}

/// Public snapshot reads: the manifest is the signed JSON document produced by
/// `verinoo-node snapshot create`. No API key is required; snapshots are public
/// chain state. The root is canonicalized once and the manifest itself must be
/// a regular file under it.
async fn snapshot_manifest(State(shared): State<Arc<Shared>>) -> Response {
    let Some(dir) = &shared.snapshot_dir else {
        return snapshot_not_found("snapshot serving is not configured");
    };
    let root = match resolve_snapshot_root(dir).await {
        Ok(root) => root,
        Err(_) => return snapshot_not_found("snapshot manifest not found"),
    };
    let path = root.join("manifest.json");
    let meta = match tokio::fs::symlink_metadata(&path).await {
        Ok(meta) => meta,
        Err(_) => return snapshot_not_found("snapshot manifest not found"),
    };
    if meta.file_type().is_symlink() || !meta.is_file() {
        return snapshot_not_found("snapshot manifest not found");
    }
    if meta.len() > MAX_SNAPSHOT_MANIFEST_BYTES {
        return snapshot_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            "snapshot manifest is too large",
        );
    }
    let bytes = match tokio::fs::read(&path).await {
        Ok(bytes) => bytes,
        Err(_) => return snapshot_not_found("snapshot manifest not found"),
    };
    match serde_json::from_slice::<serde_json::Value>(&bytes) {
        Ok(value) => {
            let mut response = Json(value).into_response();
            response
                .headers_mut()
                .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
            response
        }
        Err(_) => snapshot_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "snapshot manifest is not valid JSON",
        ),
    }
}

async fn snapshot_file(
    State(shared): State<Arc<Shared>>,
    AxumPath(relative): AxumPath<String>,
    headers: HeaderMap,
) -> Response {
    let Some(dir) = &shared.snapshot_dir else {
        return snapshot_not_found("snapshot serving is not configured");
    };
    let root = match resolve_snapshot_root(dir).await {
        Ok(root) => root,
        Err(_) => return snapshot_not_found("snapshot file not found"),
    };
    let path = match resolve_snapshot_file(&root, &relative).await {
        Ok(path) => path,
        Err(SnapshotPathError::Invalid) => {
            return snapshot_error(StatusCode::BAD_REQUEST, "invalid snapshot path")
        }
        Err(SnapshotPathError::Forbidden) => {
            return snapshot_error(StatusCode::FORBIDDEN, "snapshot path is a symlink")
        }
        Err(SnapshotPathError::NotFound) => return snapshot_not_found("snapshot file not found"),
    };

    let manifest = match load_snapshot_manifest(&root).await {
        Ok(manifest) => manifest,
        Err(SnapshotManifestError::NotFound) => {
            return snapshot_not_found("snapshot manifest not found")
        }
        Err(SnapshotManifestError::TooLarge) => {
            return snapshot_error(
                StatusCode::PAYLOAD_TOO_LARGE,
                "snapshot manifest is too large",
            )
        }
        Err(SnapshotManifestError::Invalid) => {
            return snapshot_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "snapshot manifest is not valid",
            )
        }
    };
    let listed: BTreeSet<&str> = manifest
        .files
        .iter()
        .map(|file| file.path.as_str())
        .collect();
    if !listed.contains(relative.as_str()) {
        return snapshot_not_found("snapshot file not found");
    }

    let mut file = match tokio::fs::File::open(&path).await {
        Ok(file) => file,
        Err(_) => return snapshot_not_found("snapshot file not found"),
    };
    let len = match file.metadata().await {
        Ok(meta) if meta.is_file() => meta.len(),
        _ => return snapshot_not_found("snapshot file not found"),
    };
    let Some(declared) = manifest.files.iter().find(|file| file.path == relative) else {
        return snapshot_not_found("snapshot file not found");
    };
    if declared.size != len {
        return snapshot_not_found("snapshot file not found");
    }

    let requested = match headers
        .get(header::RANGE)
        .and_then(|value| value.to_str().ok())
    {
        Some(value) => match parse_snapshot_range(value, len) {
            Ok(range) => Some(range),
            Err(()) => return snapshot_range_not_satisfiable(len),
        },
        None => None,
    };

    let (status, start, end) = match requested {
        Some((start, end)) => (StatusCode::PARTIAL_CONTENT, start, end),
        None => (StatusCode::OK, 0, len.saturating_sub(1)),
    };
    let response_len = if len == 0 { 0 } else { end - start + 1 };
    if start > 0 && file.seek(SeekFrom::Start(start)).await.is_err() {
        return snapshot_not_found("snapshot file not found");
    }

    let mut response = Response::new(snapshot_body(file, response_len));
    *response.status_mut() = status;
    let response_headers = response.headers_mut();
    response_headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/octet-stream"),
    );
    response_headers.insert(header::ACCEPT_RANGES, HeaderValue::from_static("bytes"));
    response_headers.insert(
        header::CONTENT_LENGTH,
        HeaderValue::from_str(&response_len.to_string()).expect("length is a valid header"),
    );
    if status == StatusCode::PARTIAL_CONTENT {
        response_headers.insert(
            header::CONTENT_RANGE,
            HeaderValue::from_str(&format!("bytes {start}-{end}/{len}"))
                .expect("range is a valid header"),
        );
    }
    response
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SnapshotPathError {
    Invalid,
    Forbidden,
    NotFound,
}

async fn resolve_snapshot_file(
    root: &FsPath,
    relative: &str,
) -> Result<PathBuf, SnapshotPathError> {
    if verinoo_state::snapshot::validate_snapshot_path(relative).is_err() {
        return Err(SnapshotPathError::Invalid);
    }
    let root_meta = tokio::fs::symlink_metadata(root)
        .await
        .map_err(|_| SnapshotPathError::NotFound)?;
    if root_meta.file_type().is_symlink() || !root_meta.is_dir() {
        return Err(SnapshotPathError::NotFound);
    }

    let components: Vec<_> = FsPath::new(relative).components().collect();
    let mut current = root.to_path_buf();
    for (index, component) in components.iter().enumerate() {
        let Component::Normal(name) = component else {
            return Err(SnapshotPathError::Invalid);
        };
        current.push(name);
        let meta = tokio::fs::symlink_metadata(&current)
            .await
            .map_err(|_| SnapshotPathError::NotFound)?;
        if meta.file_type().is_symlink() {
            return Err(SnapshotPathError::Forbidden);
        }
        let last = index + 1 == components.len();
        if last {
            if !meta.is_file() {
                return Err(SnapshotPathError::NotFound);
            }
        } else if !meta.is_dir() {
            return Err(SnapshotPathError::NotFound);
        }
    }
    Ok(current)
}

/// Parse a single `bytes=` range. Multiple ranges are rejected (416) so the
/// response is always one contiguous stream.
fn parse_snapshot_range(value: &str, len: u64) -> Result<(u64, u64), ()> {
    let (unit, spec) = value.trim().split_once('=').ok_or(())?;
    if !unit.eq_ignore_ascii_case("bytes") || spec.contains(',') {
        return Err(());
    }
    let (start_raw, end_raw) = spec.trim().split_once('-').ok_or(())?;
    if start_raw.trim().is_empty() {
        let suffix: u64 = end_raw.trim().parse().map_err(|_| ())?;
        if suffix == 0 || len == 0 {
            return Err(());
        }
        return Ok((len.saturating_sub(suffix), len - 1));
    }
    let start: u64 = start_raw.trim().parse().map_err(|_| ())?;
    if len == 0 || start >= len {
        return Err(());
    }
    if end_raw.trim().is_empty() {
        return Ok((start, len - 1));
    }
    let end: u64 = end_raw.trim().parse().map_err(|_| ())?;
    if end < start {
        return Err(());
    }
    Ok((start, end.min(len - 1)))
}

fn snapshot_body(file: tokio::fs::File, remaining: u64) -> Body {
    struct ReadState {
        file: tokio::fs::File,
        remaining: u64,
    }
    let stream = stream::unfold(ReadState { file, remaining }, |mut state| async move {
        if state.remaining == 0 {
            return None;
        }
        let wanted = state.remaining.min(SNAPSHOT_READ_CHUNK as u64) as usize;
        let mut buffer = vec![0u8; wanted];
        match state.file.read(&mut buffer).await {
            Ok(0) => {
                state.remaining = 0;
                Some((
                    Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "snapshot file ended before its declared length",
                    )),
                    state,
                ))
            }
            Ok(read) => {
                state.remaining -= read as u64;
                buffer.truncate(read);
                Some((Ok(Bytes::from(buffer)), state))
            }
            Err(error) => {
                state.remaining = 0;
                Some((Err(error), state))
            }
        }
    });
    Body::from_stream(stream)
}

fn snapshot_range_not_satisfiable(len: u64) -> Response {
    let mut response = Response::new(Body::empty());
    *response.status_mut() = StatusCode::RANGE_NOT_SATISFIABLE;
    let headers = response.headers_mut();
    headers.insert(header::ACCEPT_RANGES, HeaderValue::from_static("bytes"));
    headers.insert(
        header::CONTENT_RANGE,
        HeaderValue::from_str(&format!("bytes */{len}")).expect("range is a valid header"),
    );
    response
}

fn snapshot_not_found(message: &str) -> Response {
    snapshot_error(StatusCode::NOT_FOUND, message)
}

fn snapshot_error(status: StatusCode, message: &str) -> Response {
    (status, Json(serde_json::json!({ "error": message }))).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use verinoo_exec::engine::ProducerKey;
    use verinoo_state::genesis::GenesisParams;
    use verinoo_state::StateStore;

    const ADDR: &str = "ai_0123456789abcdef0123456789abcdef01234567";

    fn test_shared() -> Arc<Shared> {
        test_shared_with(GenesisParams::default())
    }

    fn test_shared_with(params: GenesisParams) -> Arc<Shared> {
        test_shared_with_policy(params, verinoo_policy::Policy::default())
    }

    fn test_shared_with_policy(
        params: GenesisParams,
        request_policy: verinoo_policy::Policy,
    ) -> Arc<Shared> {
        test_shared_full(params, request_policy, None)
    }

    fn test_shared_with_snapshot(snapshot_dir: Option<PathBuf>) -> Arc<Shared> {
        test_shared_full(
            GenesisParams::default(),
            verinoo_policy::Policy::default(),
            snapshot_dir,
        )
    }

    fn test_shared_full(
        params: GenesisParams,
        request_policy: verinoo_policy::Policy,
        snapshot_dir: Option<PathBuf>,
    ) -> Arc<Shared> {
        let store = StateStore::open_temp().unwrap();
        let signing = ed25519_dalek::SigningKey::from_bytes(&[9u8; 32]);
        let address = verinoo_exec::engine::public_key_to_address(&hex::encode(
            signing.verifying_key().to_bytes(),
        ))
        .unwrap();
        let faucet = ProducerKey::from_parts(address, signing);
        let block_time_ms = params.block_time_ms.max(500);
        Arc::new(Shared {
            store,
            params,
            mempool: tokio::sync::Mutex::new(Vec::new()),
            faucet,
            chain_lock: tokio::sync::Mutex::new(()),
            syncing: std::sync::atomic::AtomicBool::new(false),
            best_peer_height: std::sync::atomic::AtomicU64::new(0),
            block_time_ms,
            peer_lag_blocks: 2,
            request_policy,
            task_streams: Arc::new(tokio::sync::Semaphore::new(crate::MAX_ACTIVE_TASK_STREAMS)),
            snapshot_dir,
            finality: None,
            consensus_mode: "legacy".into(),
            policy: std::sync::Mutex::new(crate::Policy::default()),
        })
    }

    static INSTALL_DECODER: std::sync::Once = std::sync::Once::new();

    fn install_decoder() {
        INSTALL_DECODER.call_once(|| {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../models/tiny-gpt-v1.json");
            let bytes = std::fs::read(path).expect("committed decoder model");
            let model = verinoo_runtime::GptModel::from_bytes(&bytes).expect("valid decoder");
            let mut decoders = verinoo_runtime::GptRegistry::empty();
            decoders.insert(model).unwrap();
            verinoo_runtime::gpt::install_gpt_once(decoders);
        });
    }

    fn completed_decoder_task(
        task_id: &str,
        prompt: &str,
        sampling: Option<verinoo_types::SamplingSpec>,
    ) -> verinoo_state::models::InferenceTask {
        let model = verinoo_exec::engine::ExecModel::resolve("tiny-gpt-v1").unwrap();
        let output = model
            .result_json_with_sampling(prompt, sampling.as_ref())
            .unwrap();
        verinoo_state::models::InferenceTask {
            task_id: task_id.into(),
            requester: ADDR.into(),
            target_model: "tiny-gpt-v1".into(),
            input_commitment: verinoo_model::sha256_hex(prompt),
            input: prompt.into(),
            encrypted_input: None,
            max_fee: 100,
            deadline: 1_000,
            sampling,
            verification_type: "optimistic".into(),
            status: "completed".into(),
            assigned_to: None,
            collateral: 0,
            result_hash: Some(verinoo_model::sha256_hex(&output)),
            result_output: Some(output),
            proof_data: None,
            challenge_window_end: None,
            game_id: None,
            executed_at: 1,
        }
    }

    fn stamp_test_receipts(shared: &Shared) {
        let block = verinoo_exec::engine::Block {
            header: verinoo_exec::engine::BlockHeader {
                hash: "ab".repeat(32),
                version: verinoo_exec::engine::HEADER_VERSION,
                index: 1,
                timestamp: 1_000,
                previous_hash: "00".repeat(32),
                validator: "validator".into(),
                validator_pub_key: String::new(),
                validator_signature: "sig".into(),
                state_root: "00".repeat(32),
                tx_root: "00".repeat(32),
                inference_tasks_root: "00".repeat(32),
                compute_root: "00".repeat(32),
            },
            transactions: vec![],
        };
        verinoo_exec::engine::stamp_receipts(&shared.store, &block);
    }

    fn set_test_head(shared: &Shared, index: u64, timestamp: u64) {
        let block = verinoo_exec::engine::Block {
            header: verinoo_exec::engine::BlockHeader {
                hash: "aa".repeat(32),
                version: 1,
                index,
                timestamp,
                previous_hash: "00".repeat(32),
                validator: "genesis".into(),
                validator_pub_key: String::new(),
                validator_signature: String::new(),
                state_root: "00".repeat(32),
                tx_root: "00".repeat(32),
                inference_tasks_root: "00".repeat(32),
                compute_root: "00".repeat(32),
            },
            transactions: vec![],
        };
        verinoo_exec::engine::set_head(&shared.store, &block);
    }

    #[test]
    fn ready_requires_a_head() {
        let shared = test_shared();
        let (status, body) = ready_report(&shared, shared.now_ms(), 0);
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body.0["ready"], false);
        assert_eq!(body.0["serving"], false);
        assert_eq!(body.0["height"], 0);
        assert_eq!(body.0["reason"], "no_head");
        assert_eq!(body.0["headAgeMs"], 0);
        assert_eq!(body.0["peerCount"], 0);
    }

    #[test]
    fn ready_accepts_a_fresh_head() {
        let shared = test_shared();
        set_test_head(&shared, 42, 1_000);
        let (status, body) = ready_report(&shared, 1_500, 3);
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body.0["ready"], true);
        assert_eq!(body.0["serving"], true);
        assert_eq!(body.0["height"], 42);
        assert_eq!(body.0["headAgeMs"], 500);
        assert_eq!(body.0["peerCount"], 3);
        assert_eq!(body.0["syncing"], false);
        assert_eq!(body.0["reason"], serde_json::Value::Null);
        assert_eq!(body.0["validatorSetHash"], shared.validator_set_hash());
        assert_eq!(body.0["validatorSetConsistent"], true);
    }

    #[test]
    fn ready_rejects_a_stale_head() {
        let shared = test_shared();
        set_test_head(&shared, 42, 1_000);
        let stale_at = 1_000 + shared.block_time_ms * readiness::STALE_BLOCK_MULTIPLIER;
        let (status, body) = ready_report(&shared, stale_at, 0);
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body.0["ready"], false);
        assert_eq!(body.0["reason"], "stale_head");
    }

    #[test]
    fn ready_rejects_while_syncing() {
        let shared = test_shared();
        set_test_head(&shared, 42, 1_000);
        shared.syncing.store(true, Ordering::Relaxed);
        let (status, body) = ready_report(&shared, 1_100, 1);
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body.0["ready"], false);
        assert_eq!(body.0["syncing"], true);
        assert_eq!(body.0["reason"], "syncing");
    }

    #[test]
    fn ready_rejects_a_divergent_state_validator_set() {
        let params = GenesisParams {
            validators: vec!["ai_a".into()],
            ..GenesisParams::default()
        };
        let shared = test_shared_with(params);
        shared.store.seed_genesis_validators(&["ai_b".into()], 0);
        set_test_head(&shared, 42, 1_000);
        let (status, body) = ready_report(&shared, 1_100, 1);
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body.0["ready"], false);
        assert_eq!(body.0["validatorSetConsistent"], false);
        assert_eq!(body.0["reason"], "validator_set_mismatch");
    }

    #[test]
    fn ready_rejects_a_lagging_node_when_peer_height_is_known() {
        let shared = test_shared();
        set_test_head(&shared, 42, 1_000);
        shared.best_peer_height.store(45, Ordering::Relaxed);
        let (status, body) = ready_report(&shared, 1_100, 1);
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body.0["reason"], "behind_peer");
    }

    #[test]
    fn validator_set_hash_falls_back_to_the_configured_set() {
        let params = GenesisParams {
            validators: vec!["ai_b".into(), "ai_a".into()],
            ..GenesisParams::default()
        };
        let shared = test_shared_with(params);
        assert_eq!(
            shared.validator_set_hash(),
            crate::readiness::validator_set_hash(&["ai_a".into(), "ai_b".into()])
        );
    }

    #[test]
    fn validator_set_hash_prefers_state_over_the_configured_set() {
        let params = GenesisParams {
            validators: vec!["ai_c".into()],
            ..GenesisParams::default()
        };
        let shared = test_shared_with(params);
        shared.store.seed_genesis_validators(&["ai_a".into()], 0);
        assert_eq!(
            shared.validator_set_hash(),
            crate::readiness::validator_set_hash(&["ai_a".into()])
        );
        assert!(!shared.validator_set_consistent());
    }

    #[test]
    fn best_peer_height_is_replaced_each_poll_not_retained_forever() {
        let shared = test_shared();
        shared.set_best_peer_height(Some(100));
        assert_eq!(shared.best_peer_height(), Some(100));
        shared.set_best_peer_height(Some(90));
        assert_eq!(shared.best_peer_height(), Some(90));
        shared.set_best_peer_height(None);
        assert_eq!(shared.best_peer_height(), None);
    }

    #[tokio::test]
    async fn ready_route_has_the_network_extension_and_returns_200_and_503() {
        let shared = test_shared();
        let (net, _events) =
            verinoo_network::NetworkNode::new(0, [7u8; 32], &[], &shared.params.chain_id)
                .await
                .unwrap();
        let app = app(shared.clone(), net, Auth::disabled());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app.into_make_service())
                .await
                .unwrap();
        });

        let url = format!("http://{addr}/ready");
        let response = reqwest::get(&url).await.unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body: serde_json::Value = response.json().await.unwrap();
        assert_eq!(body["reason"], "no_head");
        assert_eq!(body["headAgeMs"], 0);

        set_test_head(&shared, 42, shared.now_ms());
        let response = reqwest::get(&url).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body: serde_json::Value = response.json().await.unwrap();
        assert_eq!(body["ready"], true);
        assert_eq!(body["peerCount"], 0);
        assert_eq!(body["validatorSetConsistent"], true);
    }

    async fn call_issue(
        shared: &Arc<Shared>,
        auth: &Auth,
        rate: &KeyRateLimit,
        ip: &str,
        body: serde_json::Value,
    ) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-for", ip.parse().unwrap());
        let peer: SocketAddr = "127.0.0.1:12345".parse().unwrap();
        v1_issue_key(
            State(shared.clone()),
            Extension(auth.clone()),
            Extension(rate.clone()),
            ConnectInfo(peer),
            headers,
            Json(body),
        )
        .await
    }

    #[tokio::test]
    async fn self_serve_route_is_404_when_disabled() {
        let shared = test_shared();
        let auth = Auth::disabled();
        let rate = KeyRateLimit::default();
        let err = call_issue(
            &shared,
            &auth,
            &rate,
            "203.0.113.1",
            serde_json::json!({ "address": ADDR }),
        )
        .await
        .unwrap_err();
        assert_eq!(err.0, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn self_serve_issue_returns_key_once_and_conflicts_on_second_request() {
        let shared = test_shared();
        let auth = Auth::disabled()
            .with_self_serve(true)
            .with_self_serve_quota(5);
        let rate = KeyRateLimit::default();
        let ok = call_issue(
            &shared,
            &auth,
            &rate,
            "203.0.113.2",
            serde_json::json!({ "address": ADDR }),
        )
        .await
        .unwrap();
        let raw = ok.0["apiKey"].as_str().unwrap();
        assert!(raw.starts_with("vnn_"));
        assert_eq!(ok.0["tenant"], "self-serve");
        assert_eq!(ok.0["address"], ADDR);
        assert_eq!(ok.0["dailyQuota"], 5);

        let conflict = call_issue(
            &shared,
            &auth,
            &rate,
            "203.0.113.2",
            serde_json::json!({ "address": ADDR }),
        )
        .await
        .unwrap_err();
        assert_eq!(conflict.0, StatusCode::CONFLICT);
        assert_eq!(conflict.1 .0["tenant"], "self-serve");
        assert_eq!(conflict.1 .0["address"], ADDR);
        assert!(!conflict.1 .0.to_string().contains(raw));

        let bad = call_issue(
            &shared,
            &auth,
            &rate,
            "203.0.113.2",
            serde_json::json!({ "address": "ai_nothex" }),
        )
        .await
        .unwrap_err();
        assert_eq!(bad.0, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn self_serve_rate_limit_is_five_per_hour_per_ip() {
        let shared = test_shared();
        let auth = Auth::disabled().with_self_serve(true);
        let rate = KeyRateLimit::default();
        for i in 0..5 {
            let address = format!("ai_{i:040x}");
            let res = call_issue(
                &shared,
                &auth,
                &rate,
                "198.51.100.9",
                serde_json::json!({ "address": address }),
            )
            .await;
            assert!(res.is_ok(), "request {i} should pass: {res:?}");
        }
        let err = call_issue(
            &shared,
            &auth,
            &rate,
            "198.51.100.9",
            serde_json::json!({ "address": format!("ai_{:040x}", 99) }),
        )
        .await
        .unwrap_err();
        assert_eq!(err.0, StatusCode::TOO_MANY_REQUESTS);

        // A different IP has its own window.
        let other = call_issue(
            &shared,
            &auth,
            &rate,
            "198.51.100.10",
            serde_json::json!({ "address": format!("ai_{:040x}", 100) }),
        )
        .await;
        assert!(other.is_ok());
    }

    #[tokio::test]
    async fn verify_reports_effective_sampling_and_rejects_tampered_receipts() {
        install_decoder();
        let shared = test_shared();
        let spec = verinoo_types::SamplingSpec {
            temperature_milli: 800,
            top_k: 40,
            top_p_milli: 900,
            seed: 7,
        };
        let prompt = "please cancel";
        let override_task_id = "verify-sampling-override";
        let default_task_id = "verify-sampling-default";
        shared.store.set_task(&completed_decoder_task(
            override_task_id,
            prompt,
            Some(spec),
        ));
        shared
            .store
            .set_task(&completed_decoder_task(default_task_id, prompt, None));
        stamp_test_receipts(&shared);

        let Json(override_body) = v1_verify(
            State(shared.clone()),
            axum::extract::Path(override_task_id.to_string()),
        )
        .await
        .unwrap();
        assert_eq!(override_body["verified"], true);
        assert_eq!(
            override_body["sampling"],
            serde_json::to_value(spec).unwrap()
        );
        let receipt = verinoo_exec::engine::receipt_for(&shared.store, override_task_id).unwrap();
        assert_eq!(receipt["samplingHash"], override_body["samplingHash"]);

        let Json(default_body) = v1_verify(
            State(shared.clone()),
            axum::extract::Path(default_task_id.to_string()),
        )
        .await
        .unwrap();
        assert_eq!(default_body["verified"], true);
        let model = verinoo_exec::engine::ExecModel::resolve("tiny-gpt-v1").unwrap();
        assert_eq!(
            default_body["sampling"],
            model.effective_sampling_json(None).unwrap()
        );
        let default_receipt =
            verinoo_exec::engine::receipt_for(&shared.store, default_task_id).unwrap();
        assert!(default_receipt.get("samplingHash").is_none());

        let receipt_key = format!("receipt:{override_task_id}");
        let mut tampered = receipt.clone();
        tampered["outputHash"] = serde_json::json!("00".repeat(32));
        shared.store.set_meta(&receipt_key, &tampered);
        let Json(body) = v1_verify(
            State(shared.clone()),
            axum::extract::Path(override_task_id.to_string()),
        )
        .await
        .unwrap();
        assert_eq!(body["verified"], false);

        let mut tampered = receipt.clone();
        tampered["samplingHash"] = serde_json::json!("00".repeat(32));
        shared.store.set_meta(&receipt_key, &tampered);
        let Json(body) = v1_verify(
            State(shared.clone()),
            axum::extract::Path(override_task_id.to_string()),
        )
        .await
        .unwrap();
        assert_eq!(body["verified"], false);

        let mut legacy = receipt.clone();
        if let Some(object) = legacy.as_object_mut() {
            object.remove("sampling");
            object.remove("samplingHash");
            object.remove("samplingSource");
        }
        shared.store.set_meta(&receipt_key, &legacy);
        let Json(body) = v1_verify(
            State(shared.clone()),
            axum::extract::Path(override_task_id.to_string()),
        )
        .await
        .unwrap();
        assert_eq!(
            body["verified"], true,
            "old receipts without sampling fields stay compatible"
        );
    }

    fn test_key(seed: u8) -> ed25519_dalek::SigningKey {
        ed25519_dalek::SigningKey::from_bytes(&[seed; 32])
    }

    fn key_address(signing: &ed25519_dalek::SigningKey) -> String {
        verinoo_exec::engine::public_key_to_address(&hex::encode(
            signing.verifying_key().to_bytes(),
        ))
        .unwrap()
    }

    fn fund(shared: &Shared, address: &str) {
        let mut account = shared.store.account_or_default(address);
        account.balance = 10_000_000;
        shared.store.set_account(&account);
    }

    fn transaction_fee(shared: &Shared, tx: &verinoo_types::tx::TxEnvelope) -> u64 {
        let data = serde_json::to_value(&tx.data).unwrap();
        verinoo_exec::gas::gas_cost_for(&data)
            * shared
                .store
                .meta_u64_or("baseFee", shared.params.initial_base_fee)
            * verinoo_exec::gas::policy_multiplier(verinoo_exec::gas::policy_of_tx(&data))
                .unwrap_or(1)
    }

    fn signed_inference(
        signing: &ed25519_dalek::SigningKey,
        from: &str,
        data: serde_json::Value,
        nonce: u64,
    ) -> verinoo_types::tx::TxEnvelope {
        use ed25519_dalek::Signer as _;
        let data = serde_json::json!({ "type": "submitInference", "data": data });
        let payload = serde_json::json!({
            "data": data,
            "from": from,
            "nonce": nonce,
            "to": "",
            "value": 0,
        });
        let tx_id =
            verinoo_model::sha256_hex(&verinoo_types::canonical::canonical_json(&payload).unwrap());
        let signature = signing.sign(tx_id.as_bytes());
        serde_json::from_value(serde_json::json!({
            "txId": tx_id,
            "from": from,
            "to": "",
            "value": 0,
            "nonce": nonce,
            "data": data,
            "publicKey": hex::encode(signing.verifying_key().to_bytes()),
            "signature": hex::encode(signature.to_bytes()),
        }))
        .unwrap()
    }

    fn inference_data(requester: &str, model: &str, input: &str) -> serde_json::Value {
        serde_json::json!({
            "requester": requester,
            "targetModel": model,
            "input": input,
            "inputCommitment": verinoo_crypto::sha256_hex(input),
            "maxFee": 100,
            "deadline": 1000,
            "verificationType": "optimistic",
        })
    }

    async fn test_network(shared: &Shared) -> verinoo_network::NetworkNode {
        let (net, _events) =
            verinoo_network::NetworkNode::new(0, [7u8; 32], &[], &shared.params.chain_id)
                .await
                .unwrap();
        net
    }

    async fn call_batch(
        shared: &Arc<Shared>,
        net: &verinoo_network::NetworkNode,
        auth: &Auth,
        headers: HeaderMap,
        transactions: Vec<serde_json::Value>,
    ) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
        v1_submit_batch(
            State(shared.clone()),
            Extension(net.clone()),
            Extension(auth.clone()),
            headers,
            Json(serde_json::json!({ "transactions": transactions })),
        )
        .await
    }

    #[tokio::test]
    async fn batch_rejects_empty_and_oversized_requests() {
        let shared = test_shared();
        let net = test_network(&shared).await;
        let auth = Auth::disabled();

        let error = call_batch(&shared, &net, &auth, HeaderMap::new(), vec![])
            .await
            .unwrap_err();
        assert_eq!(error.0, StatusCode::BAD_REQUEST);

        let signing = test_key(4);
        let address = key_address(&signing);
        fund(&shared, &address);
        let tx = serde_json::to_value(signed_inference(
            &signing,
            &address,
            inference_data(&address, "tiny-gpt-v1", "hello"),
            1,
        ))
        .unwrap();
        let error = call_batch(&shared, &net, &auth, HeaderMap::new(), vec![tx; 33])
            .await
            .unwrap_err();
        assert_eq!(error.0, StatusCode::BAD_REQUEST);
        assert!(shared.mempool.lock().await.is_empty());
    }

    #[tokio::test]
    async fn batch_rejects_a_request_that_exceeds_one_block_of_gas() {
        let shared = test_shared();
        let net = test_network(&shared).await;
        let auth = Auth::disabled();
        let signing = test_key(20);
        let address = key_address(&signing);
        fund(&shared, &address);

        let items: Vec<_> = (1..=17)
            .map(|nonce| {
                let input = format!("prompt {nonce}");
                serde_json::to_value(signed_inference(
                    &signing,
                    &address,
                    inference_data(&address, "tiny-gpt-v1", &input),
                    nonce,
                ))
                .unwrap()
            })
            .collect();

        // 17 optimistic submitInference txs cost 17 * 60 = 1020 gas, above the
        // 1000 block limit, so the request is rejected before admission.
        let error = call_batch(&shared, &net, &auth, HeaderMap::new(), items.clone())
            .await
            .unwrap_err();
        assert_eq!(error.0, StatusCode::BAD_REQUEST);
        let message = error.1 .0["error"].as_str().unwrap();
        assert!(message.contains("1020"), "{message}");
        assert!(message.contains("1000"), "{message}");
        assert!(shared.mempool.lock().await.is_empty());

        // 16 items cost 960 gas and fit.
        let body = call_batch(&shared, &net, &auth, HeaderMap::new(), items[..16].to_vec())
            .await
            .unwrap()
            .0;
        assert_eq!(body["accepted"], 16);
        assert_eq!(body["rejected"], 0);
        assert_eq!(shared.mempool.lock().await.len(), 16);
    }

    #[tokio::test]
    async fn admit_tx_refuses_when_the_shared_mempool_cap_is_reached() {
        let shared = test_shared();
        let net = test_network(&shared).await;
        let auth = Auth::disabled();
        let signing = test_key(21);
        let address = key_address(&signing);
        fund(&shared, &address);
        let tx = signed_inference(
            &signing,
            &address,
            inference_data(&address, "tiny-gpt-v1", "cap prompt"),
            1,
        );

        {
            let mut mp = shared.mempool.lock().await;
            let mut filler = tx.clone();
            for index in 0..crate::MAX_MEMPOOL_TRANSACTIONS {
                filler.tx_id = format!("filler-{index}");
                mp.push(filler.clone());
            }
        }

        let error = admit_tx(&shared, &net, &auth, None, tx.clone())
            .await
            .unwrap_err();
        assert_eq!(error.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(error.message().contains("mempool is full"));
        assert_eq!(
            shared.mempool.lock().await.len(),
            crate::MAX_MEMPOOL_TRANSACTIONS
        );

        shared.mempool.lock().await.pop();
        admit_tx(&shared, &net, &auth, None, tx).await.unwrap();
        assert_eq!(
            shared.mempool.lock().await.len(),
            crate::MAX_MEMPOOL_TRANSACTIONS
        );
    }

    #[tokio::test]
    async fn batch_accepts_valid_items_and_reports_per_item_errors() {
        let shared = test_shared();
        let net = test_network(&shared).await;
        let auth = Auth::disabled();
        let signing = test_key(5);
        let address = key_address(&signing);
        fund(&shared, &address);

        let good = signed_inference(
            &signing,
            &address,
            inference_data(&address, "tiny-gpt-v1", "first prompt"),
            1,
        );
        let expected_task = verinoo_exec::engine::task_id_for(&good).unwrap();
        let mut bad = signed_inference(
            &signing,
            &address,
            inference_data(&address, "tiny-gpt-v1", "second prompt"),
            1,
        );
        bad.signature = "00".repeat(64);

        let body = call_batch(
            &shared,
            &net,
            &auth,
            HeaderMap::new(),
            vec![
                serde_json::to_value(&good).unwrap(),
                serde_json::to_value(&bad).unwrap(),
            ],
        )
        .await
        .unwrap()
        .0;

        assert_eq!(body["accepted"], 1);
        assert_eq!(body["rejected"], 1);
        assert_eq!(body["results"][0]["index"], 0);
        assert_eq!(body["results"][0]["accepted"], true);
        assert_eq!(body["results"][0]["txId"], good.tx_id);
        assert_eq!(body["results"][0]["taskId"], expected_task);
        assert_eq!(body["results"][1]["index"], 1);
        assert_eq!(body["results"][1]["accepted"], false);
        assert_eq!(body["results"][1]["txId"], bad.tx_id);
        assert!(body["results"][1]["error"]
            .as_str()
            .unwrap()
            .contains("sig"));
        assert_eq!(shared.mempool.lock().await.len(), 1);
    }

    #[tokio::test]
    async fn batch_auth_mismatch_is_per_item_and_missing_auth_fails_the_request() {
        let shared = test_shared();
        let net = test_network(&shared).await;

        let allowed_signing = test_key(6);
        let allowed = key_address(&allowed_signing);
        let other_signing = test_key(7);
        let other = key_address(&other_signing);
        fund(&shared, &allowed);
        fund(&shared, &other);

        let path = std::env::temp_dir().join(format!(
            "verinoo-batch-auth-{}-keys.json",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let raw_key = crate::auth::add_key(&path, "acme", &allowed, 0).unwrap();
        let auth = Auth::load(&path).unwrap();

        let allowed_tx = signed_inference(
            &allowed_signing,
            &allowed,
            inference_data(&allowed, "tiny-gpt-v1", "allowed"),
            1,
        );
        let other_tx = signed_inference(
            &other_signing,
            &other,
            inference_data(&other, "tiny-gpt-v1", "other"),
            1,
        );
        let items = vec![
            serde_json::to_value(&allowed_tx).unwrap(),
            serde_json::to_value(&other_tx).unwrap(),
        ];

        let mut headers = HeaderMap::new();
        headers.insert(
            "authorization",
            format!("Bearer {raw_key}").parse().unwrap(),
        );
        let body = call_batch(&shared, &net, &auth, headers.clone(), items.clone())
            .await
            .unwrap()
            .0;
        assert_eq!(body["accepted"], 1);
        assert_eq!(body["rejected"], 1);
        assert!(body["results"][1]["error"]
            .as_str()
            .unwrap()
            .contains("not authorized"));
        assert_eq!(shared.mempool.lock().await.len(), 1);

        let fresh = test_shared();
        let fresh_net = test_network(&fresh).await;
        let error = call_batch(&fresh, &fresh_net, &auth, HeaderMap::new(), items)
            .await
            .unwrap_err();
        assert_eq!(error.0, StatusCode::UNAUTHORIZED);
        assert!(fresh.mempool.lock().await.is_empty());

        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn batch_still_enforces_sampling_and_encryption_rules() {
        let shared = test_shared();
        let net = test_network(&shared).await;
        let auth = Auth::disabled();
        let signing = test_key(8);
        let address = key_address(&signing);
        fund(&shared, &address);

        let mut sampling_data = inference_data(&address, "intent-classifier-v1", "hello");
        sampling_data["sampling"] = serde_json::json!({
            "temperatureMilli": 800,
            "topK": 40,
            "topPMilli": 900,
            "seed": 7,
        });
        let sampling_tx = signed_inference(&signing, &address, sampling_data, 1);

        let mut both_data = inference_data(&address, "tiny-gpt-v1", "hello");
        both_data["encryptedInput"] = serde_json::json!({
            "ephemeralPublicKey": "ab".repeat(32),
            "nonce": "00".repeat(12),
            "ciphertext": "AAAA",
        });
        let both_tx = signed_inference(&signing, &address, both_data, 1);

        let body = call_batch(
            &shared,
            &net,
            &auth,
            HeaderMap::new(),
            vec![
                serde_json::to_value(sampling_tx).unwrap(),
                serde_json::to_value(both_tx).unwrap(),
            ],
        )
        .await
        .unwrap()
        .0;
        assert_eq!(body["accepted"], 0);
        assert_eq!(body["rejected"], 2);
        assert!(body["results"][0]["error"]
            .as_str()
            .unwrap()
            .contains("sampling"));
        assert!(body["results"][1]["error"]
            .as_str()
            .unwrap()
            .contains("not both"));
        assert!(shared.mempool.lock().await.is_empty());
    }

    #[tokio::test]
    async fn batch_policy_rejection_is_per_item_and_redacts_terms() {
        let policy =
            verinoo_policy::Policy::from_json(r#"{ "blockedTerms": ["secretproject"] }"#).unwrap();
        let shared = test_shared_with_policy(GenesisParams::default(), policy);
        let net = test_network(&shared).await;
        let auth = Auth::disabled();
        let signing = test_key(10);
        let address = key_address(&signing);
        fund(&shared, &address);

        let blocked = signed_inference(
            &signing,
            &address,
            inference_data(&address, "tiny-gpt-v1", "summarize secretPROJECT notes"),
            1,
        );
        let clean = signed_inference(
            &signing,
            &address,
            inference_data(&address, "tiny-gpt-v1", "summarize release notes"),
            1,
        );

        let body = call_batch(
            &shared,
            &net,
            &auth,
            HeaderMap::new(),
            vec![
                serde_json::to_value(blocked).unwrap(),
                serde_json::to_value(clean).unwrap(),
            ],
        )
        .await
        .unwrap()
        .0;
        assert_eq!(body["accepted"], 1);
        assert_eq!(body["rejected"], 1);
        assert_eq!(body["results"][0]["reason"], "policy");
        assert_eq!(body["results"][0]["policyReason"], "blocked_term");
        assert!(!body["results"][0]["error"]
            .as_str()
            .unwrap()
            .to_lowercase()
            .contains("secretproject"));
        assert_eq!(body["results"][1]["accepted"], true);
        assert_eq!(shared.mempool.lock().await.len(), 1);
    }

    #[tokio::test]
    async fn batch_accepts_consecutive_nonces_from_one_sender() {
        let shared = test_shared();
        let net = test_network(&shared).await;
        let auth = Auth::disabled();
        let signing = test_key(11);
        let address = key_address(&signing);
        fund(&shared, &address);

        let first = signed_inference(
            &signing,
            &address,
            inference_data(&address, "tiny-gpt-v1", "first prompt"),
            1,
        );
        let second = signed_inference(
            &signing,
            &address,
            inference_data(&address, "tiny-gpt-v1", "second prompt"),
            2,
        );
        let first_task = verinoo_exec::engine::task_id_for(&first).unwrap();
        let second_task = verinoo_exec::engine::task_id_for(&second).unwrap();

        let body = call_batch(
            &shared,
            &net,
            &auth,
            HeaderMap::new(),
            vec![
                serde_json::to_value(&first).unwrap(),
                serde_json::to_value(&second).unwrap(),
            ],
        )
        .await
        .unwrap()
        .0;

        assert_eq!(body["accepted"], 2);
        assert_eq!(body["rejected"], 0);
        assert_eq!(body["results"][0]["index"], 0);
        assert_eq!(body["results"][0]["accepted"], true);
        assert_eq!(body["results"][0]["txId"], first.tx_id);
        assert_eq!(body["results"][0]["taskId"], first_task);
        assert_eq!(body["results"][1]["index"], 1);
        assert_eq!(body["results"][1]["accepted"], true);
        assert_eq!(body["results"][1]["txId"], second.tx_id);
        assert_eq!(body["results"][1]["taskId"], second_task);
        assert_ne!(first_task, second_task);
        assert_eq!(shared.store.account_or_default(&address).nonce, 0);

        let mp = shared.mempool.lock().await;
        assert_eq!(mp.len(), 2);
        assert_eq!(mp[0].tx_id, first.tx_id);
        assert_eq!(mp[1].tx_id, second.tx_id);
    }

    #[tokio::test]
    async fn batch_nonce_gap_rejects_only_that_item_without_advancing_the_overlay() {
        let shared = test_shared();
        let net = test_network(&shared).await;
        let auth = Auth::disabled();
        let signing = test_key(12);
        let address = key_address(&signing);
        fund(&shared, &address);

        let first = signed_inference(
            &signing,
            &address,
            inference_data(&address, "tiny-gpt-v1", "first prompt"),
            1,
        );
        let gap = signed_inference(
            &signing,
            &address,
            inference_data(&address, "tiny-gpt-v1", "gap prompt"),
            3,
        );
        let after_gap = signed_inference(
            &signing,
            &address,
            inference_data(&address, "tiny-gpt-v1", "after gap"),
            2,
        );

        let body = call_batch(
            &shared,
            &net,
            &auth,
            HeaderMap::new(),
            vec![
                serde_json::to_value(&first).unwrap(),
                serde_json::to_value(&gap).unwrap(),
                serde_json::to_value(&after_gap).unwrap(),
            ],
        )
        .await
        .unwrap()
        .0;

        assert_eq!(body["accepted"], 2);
        assert_eq!(body["rejected"], 1);
        assert_eq!(body["results"][0]["accepted"], true);
        assert_eq!(body["results"][1]["index"], 1);
        assert_eq!(body["results"][1]["accepted"], false);
        assert_eq!(body["results"][1]["txId"], gap.tx_id);
        assert!(body["results"][1]["error"]
            .as_str()
            .unwrap()
            .contains("bad nonce"));
        assert_eq!(body["results"][2]["index"], 2);
        assert_eq!(body["results"][2]["accepted"], true);
        assert_eq!(body["results"][2]["txId"], after_gap.tx_id);

        let mp = shared.mempool.lock().await;
        assert_eq!(mp.len(), 2);
        assert_eq!(mp[0].tx_id, first.tx_id);
        assert_eq!(mp[1].tx_id, after_gap.tx_id);
    }

    #[tokio::test]
    async fn batch_rejects_only_the_item_that_exceeds_the_overlay_balance() {
        let shared = test_shared();
        let net = test_network(&shared).await;
        let auth = Auth::disabled();
        let signing = test_key(13);
        let address = key_address(&signing);

        let first = signed_inference(
            &signing,
            &address,
            inference_data(&address, "tiny-gpt-v1", "first prompt"),
            1,
        );
        let second = signed_inference(
            &signing,
            &address,
            inference_data(&address, "tiny-gpt-v1", "second prompt"),
            2,
        );
        let mut account = shared.store.account_or_default(&address);
        account.balance = transaction_fee(&shared, &first) as i64;
        shared.store.set_account(&account);

        let body = call_batch(
            &shared,
            &net,
            &auth,
            HeaderMap::new(),
            vec![
                serde_json::to_value(&first).unwrap(),
                serde_json::to_value(&second).unwrap(),
            ],
        )
        .await
        .unwrap()
        .0;

        assert_eq!(body["accepted"], 1);
        assert_eq!(body["rejected"], 1);
        assert_eq!(body["results"][0]["txId"], first.tx_id);
        assert_eq!(body["results"][0]["accepted"], true);
        assert_eq!(body["results"][1]["index"], 1);
        assert_eq!(body["results"][1]["accepted"], false);
        assert_eq!(body["results"][1]["txId"], second.tx_id);
        assert_eq!(
            body["results"][1]["error"],
            "insufficient balance for value + gas"
        );

        let mp = shared.mempool.lock().await;
        assert_eq!(mp.len(), 1);
        assert_eq!(mp[0].tx_id, first.tx_id);
    }

    #[tokio::test]
    async fn batch_overlays_are_independent_per_sender() {
        let shared = test_shared();
        let net = test_network(&shared).await;
        let auth = Auth::disabled();
        let first_signing = test_key(14);
        let first_address = key_address(&first_signing);
        let second_signing = test_key(15);
        let second_address = key_address(&second_signing);
        fund(&shared, &first_address);
        fund(&shared, &second_address);

        let first = signed_inference(
            &first_signing,
            &first_address,
            inference_data(&first_address, "tiny-gpt-v1", "first sender"),
            1,
        );
        let second = signed_inference(
            &second_signing,
            &second_address,
            inference_data(&second_address, "tiny-gpt-v1", "second sender"),
            1,
        );

        let body = call_batch(
            &shared,
            &net,
            &auth,
            HeaderMap::new(),
            vec![
                serde_json::to_value(&first).unwrap(),
                serde_json::to_value(&second).unwrap(),
            ],
        )
        .await
        .unwrap()
        .0;

        assert_eq!(body["accepted"], 2);
        assert_eq!(body["rejected"], 0);
        assert_eq!(shared.mempool.lock().await.len(), 2);
    }

    #[tokio::test]
    async fn single_tx_submission_still_validates_against_committed_nonce() {
        let shared = test_shared();
        let net = test_network(&shared).await;
        let auth = Auth::disabled();
        let signing = test_key(16);
        let address = key_address(&signing);
        fund(&shared, &address);

        let first = signed_inference(
            &signing,
            &address,
            inference_data(&address, "tiny-gpt-v1", "first prompt"),
            1,
        );
        let next_nonce = signed_inference(
            &signing,
            &address,
            inference_data(&address, "tiny-gpt-v1", "next prompt"),
            2,
        );
        submit_one(&shared, &net, &auth, None, first).await.unwrap();
        let error = submit_one(&shared, &net, &auth, None, next_nonce)
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            SubmitError::Invalid(message) if message.starts_with("bad nonce")
        ));
        assert_eq!(shared.mempool.lock().await.len(), 1);
    }

    fn test_task(task_id: &str, status: &str) -> verinoo_state::models::InferenceTask {
        let completed = status == "completed";
        verinoo_state::models::InferenceTask {
            task_id: task_id.into(),
            requester: ADDR.into(),
            target_model: "tiny-gpt-v1".into(),
            input_commitment: verinoo_model::sha256_hex("secret prompt"),
            input: "secret prompt".into(),
            encrypted_input: None,
            max_fee: 100,
            deadline: 1000,
            sampling: None,
            verification_type: "optimistic".into(),
            status: status.into(),
            assigned_to: None,
            collateral: 0,
            result_hash: completed.then(|| "ab".repeat(32)),
            result_output: completed.then(|| "final output".to_string()),
            proof_data: None,
            challenge_window_end: None,
            game_id: None,
            executed_at: 5,
        }
    }

    async fn collect_events<S>(stream: Sse<S>) -> String
    where
        S: Stream<Item = Result<Event, Infallible>> + Send + 'static,
    {
        use axum::response::IntoResponse as _;
        let response = stream.into_response();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    #[tokio::test]
    async fn task_events_returns_404_before_streaming_for_unknown_tasks() {
        let shared = test_shared();
        let error = v1_task_events(
            State(shared.clone()),
            axum::extract::Path("missing-task".to_string()),
        )
        .await
        .unwrap_err();
        assert_eq!(error.0, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn task_events_rejects_when_the_global_stream_cap_is_exhausted() {
        let shared = test_shared();
        let task_id = "task-stream-cap";
        shared.store.set_task(&test_task(task_id, "pending"));

        let permits: Vec<_> = (0..crate::MAX_ACTIVE_TASK_STREAMS)
            .map(|_| shared.task_streams.clone().try_acquire_owned().unwrap())
            .collect();
        let error = v1_task_events(
            State(shared.clone()),
            axum::extract::Path(task_id.to_string()),
        )
        .await
        .unwrap_err();
        assert_eq!(error.0, StatusCode::TOO_MANY_REQUESTS);

        drop(permits);
        assert!(v1_task_events(
            State(shared.clone()),
            axum::extract::Path(task_id.to_string()),
        )
        .await
        .is_ok());
    }

    #[tokio::test]
    async fn task_events_emits_status_result_and_done_for_completed_and_failed_tasks() {
        for status in ["completed", "failed"] {
            let shared = test_shared();
            let task_id = format!("task-events-{status}");
            shared.store.set_task(&test_task(&task_id, status));
            if status == "completed" {
                shared.store.set_meta(
                    &format!("receipt:{task_id}"),
                    &serde_json::json!({ "taskId": task_id, "outputHash": "cd".repeat(32) }),
                );
            }

            let body = collect_events(task_event_stream(
                shared.clone(),
                task_id.clone(),
                Duration::from_millis(5),
                Duration::from_millis(500),
                None,
            ))
            .await;

            assert!(body.contains("event: status"), "{body}");
            assert!(body.contains("event: result"), "{body}");
            assert!(body.contains("event: done"), "{body}");
            assert!(body.contains(&task_id));
            assert!(body.contains(&format!("\"status\":\"{status}\"")));
            assert!(!body.contains("secret prompt"));
            assert!(!body.contains("event: token"));
            assert!(!body.contains("event: preview"));
            if status == "completed" {
                assert!(body.contains("\"receipt\":{"));
            } else {
                assert!(body.contains("\"receipt\":null"));
            }
        }
    }

    struct SnapshotTempDir(std::path::PathBuf);

    impl SnapshotTempDir {
        fn new(tag: &str) -> Self {
            use std::sync::atomic::{AtomicU64, Ordering};
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let n = COUNTER.fetch_add(1, Ordering::SeqCst);
            let path = std::env::temp_dir().join(format!(
                "verinoo-rpc-snapshot-{tag}-{}-{n}",
                std::process::id()
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl Drop for SnapshotTempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    async fn response_parts(response: Response) -> (StatusCode, HeaderMap, Vec<u8>) {
        let status = response.status();
        let headers = response.headers().clone();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap()
            .to_vec();
        (status, headers, body)
    }

    fn range_headers(value: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(header::RANGE, value.parse().unwrap());
        headers
    }

    fn write_serving_manifest(root: &std::path::Path, files: &[(&str, u64)]) {
        let files: Vec<serde_json::Value> = files
            .iter()
            .map(|(path, size)| {
                serde_json::json!({
                    "path": path,
                    "size": size,
                    "sha256": "00".repeat(32)
                })
            })
            .collect();
        let manifest = serde_json::json!({
            "v": 1,
            "kind": "verinoo-state-snapshot",
            "chainId": "verinoo-test-1",
            "height": 1,
            "blockHash": "aa",
            "parentHash": "bb",
            "stateRoot": "cc",
            "modelSetHash": "dd",
            "modelHashes": {},
            "validatorSetHash": "ee",
            "createdAt": 1,
            "files": files
        });
        std::fs::write(
            root.join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
    }

    #[tokio::test]
    async fn snapshot_manifest_is_404_without_a_configured_or_present_snapshot() {
        let shared = test_shared_with_snapshot(None);
        let response = snapshot_manifest(State(shared)).await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);

        let temp = SnapshotTempDir::new("manifest-missing");
        let shared = test_shared_with_snapshot(Some(temp.path().join("missing")));
        let response = snapshot_manifest(State(shared)).await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);

        let snapshot_dir = temp.path().join("snapshot");
        std::fs::create_dir_all(&snapshot_dir).unwrap();
        std::fs::write(
            snapshot_dir.join("manifest.json"),
            br#"{"kind":"verinoo-state-snapshot","height":9}"#,
        )
        .unwrap();
        let shared = test_shared_with_snapshot(Some(snapshot_dir));
        let response = snapshot_manifest(State(shared)).await;
        let (status, _headers, body) = response_parts(response).await;
        assert_eq!(status, StatusCode::OK);
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["height"], 9);
    }

    #[tokio::test]
    async fn snapshot_file_serves_state_paths_and_ranges() {
        let temp = SnapshotTempDir::new("ranges");
        let snapshot_dir = temp.path().join("snapshot");
        std::fs::create_dir_all(snapshot_dir.join("state")).unwrap();
        let bytes: Vec<u8> = (0..=255u8).collect();
        std::fs::write(snapshot_dir.join("state/data.bin"), &bytes).unwrap();
        write_serving_manifest(&snapshot_dir, &[("state/data.bin", bytes.len() as u64)]);
        let shared = test_shared_with_snapshot(Some(snapshot_dir));

        let response = snapshot_file(
            State(shared.clone()),
            AxumPath("state/data.bin".into()),
            HeaderMap::new(),
        )
        .await;
        let (status, headers, body) = response_parts(response).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(headers[header::ACCEPT_RANGES], "bytes");
        assert_eq!(headers[header::CONTENT_LENGTH], "256");
        assert_eq!(body, bytes);

        let response = snapshot_file(
            State(shared.clone()),
            AxumPath("state/data.bin".into()),
            range_headers("bytes=10-19"),
        )
        .await;
        let (status, headers, body) = response_parts(response).await;
        assert_eq!(status, StatusCode::PARTIAL_CONTENT);
        assert_eq!(headers[header::CONTENT_RANGE], "bytes 10-19/256");
        assert_eq!(headers[header::CONTENT_LENGTH], "10");
        assert_eq!(body, &bytes[10..=19]);

        let response = snapshot_file(
            State(shared.clone()),
            AxumPath("state/data.bin".into()),
            range_headers("bytes=246-"),
        )
        .await;
        let (status, headers, body) = response_parts(response).await;
        assert_eq!(status, StatusCode::PARTIAL_CONTENT);
        assert_eq!(headers[header::CONTENT_RANGE], "bytes 246-255/256");
        assert_eq!(body, &bytes[246..]);

        let response = snapshot_file(
            State(shared.clone()),
            AxumPath("state/data.bin".into()),
            range_headers("bytes=-10"),
        )
        .await;
        let (status, headers, body) = response_parts(response).await;
        assert_eq!(status, StatusCode::PARTIAL_CONTENT);
        assert_eq!(headers[header::CONTENT_RANGE], "bytes 246-255/256");
        assert_eq!(body, &bytes[246..]);

        for value in ["bytes=999-", "bytes=9-3", "bytes=abc", "bytes=0-1,4-5"] {
            let response = snapshot_file(
                State(shared.clone()),
                AxumPath("state/data.bin".into()),
                range_headers(value),
            )
            .await;
            let (status, headers, _body) = response_parts(response).await;
            assert_eq!(status, StatusCode::RANGE_NOT_SATISFIABLE, "{value}");
            assert_eq!(headers[header::CONTENT_RANGE], "bytes */256");
        }
    }

    #[tokio::test]
    async fn snapshot_file_rejects_traversal_absolute_and_symlink_paths() {
        let temp = SnapshotTempDir::new("paths");
        let snapshot_dir = temp.path().join("snapshot");
        std::fs::create_dir_all(snapshot_dir.join("state")).unwrap();
        std::fs::write(snapshot_dir.join("state/ok.bin"), b"ok").unwrap();
        write_serving_manifest(&snapshot_dir, &[("state/ok.bin", 2)]);
        let shared = test_shared_with_snapshot(Some(snapshot_dir.clone()));

        for path in [
            "../outside",
            "/etc/passwd",
            "state/../../outside",
            "state\\..\\outside",
            "other/file",
            "state",
        ] {
            let response = snapshot_file(
                State(shared.clone()),
                AxumPath(path.into()),
                HeaderMap::new(),
            )
            .await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{path}");
        }

        let response = snapshot_file(
            State(shared.clone()),
            AxumPath("state/missing.sst".into()),
            HeaderMap::new(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);

        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            let outside = temp.path().join("outside.bin");
            std::fs::write(&outside, b"secret").unwrap();
            symlink(&outside, snapshot_dir.join("state/escape.bin")).unwrap();
            let response = snapshot_file(
                State(shared.clone()),
                AxumPath("state/escape.bin".into()),
                HeaderMap::new(),
            )
            .await;
            assert_eq!(response.status(), StatusCode::FORBIDDEN);
        }
    }

    #[tokio::test]
    async fn snapshot_file_serves_only_manifest_listed_files() {
        let temp = SnapshotTempDir::new("listed");
        let snapshot_dir = temp.path().join("snapshot");
        std::fs::create_dir_all(snapshot_dir.join("state")).unwrap();
        std::fs::write(snapshot_dir.join("state/listed.bin"), b"listed").unwrap();
        std::fs::write(snapshot_dir.join("state/unlisted.bin"), b"secret").unwrap();
        write_serving_manifest(&snapshot_dir, &[("state/listed.bin", 6)]);
        let shared = test_shared_with_snapshot(Some(snapshot_dir));

        let response = snapshot_file(
            State(shared.clone()),
            AxumPath("state/listed.bin".into()),
            HeaderMap::new(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);

        let response = snapshot_file(
            State(shared),
            AxumPath("state/unlisted.bin".into()),
            HeaderMap::new(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn snapshot_manifest_rejects_an_oversized_document() {
        let temp = SnapshotTempDir::new("manifest-large");
        let snapshot_dir = temp.path().join("snapshot");
        std::fs::create_dir_all(&snapshot_dir).unwrap();
        let oversized = vec![b' '; (MAX_SNAPSHOT_MANIFEST_BYTES + 1) as usize];
        std::fs::write(snapshot_dir.join("manifest.json"), oversized).unwrap();
        let shared = test_shared_with_snapshot(Some(snapshot_dir));

        let response = snapshot_manifest(State(shared)).await;
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn snapshot_endpoints_accept_a_symlinked_root() {
        use std::os::unix::fs::symlink;

        let temp = SnapshotTempDir::new("symlink-root");
        let real = temp.path().join("real-snapshot");
        std::fs::create_dir_all(real.join("state")).unwrap();
        std::fs::write(real.join("state/data.bin"), b"data").unwrap();
        write_serving_manifest(&real, &[("state/data.bin", 4)]);
        let link = temp.path().join("linked-snapshot");
        symlink(&real, &link).unwrap();
        let shared = test_shared_with_snapshot(Some(link));

        let response = snapshot_manifest(State(shared.clone())).await;
        assert_eq!(response.status(), StatusCode::OK);

        let response = snapshot_file(
            State(shared),
            AxumPath("state/data.bin".into()),
            HeaderMap::new(),
        )
        .await;
        let (status, _headers, body) = response_parts(response).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, b"data");
    }

    #[tokio::test]
    async fn task_events_times_out_without_token_output() {
        let shared = test_shared();
        let task_id = "task-events-timeout";
        shared.store.set_task(&test_task(task_id, "pending"));

        let body = collect_events(task_event_stream(
            shared.clone(),
            task_id.to_string(),
            Duration::from_millis(5),
            Duration::from_millis(30),
            None,
        ))
        .await;

        assert!(body.contains("event: status"));
        assert!(body.contains("event: timeout"));
        assert!(body.contains("event: done"));
        assert!(!body.contains("event: result"));
        assert!(!body.contains("event: token"));
    }

    fn test_bft_validators(count: u8) -> (Vec<ed25519_dalek::SigningKey>, Vec<String>) {
        let keys: Vec<_> = (0..count)
            .map(|index| ed25519_dalek::SigningKey::from_bytes(&[index.saturating_add(1); 32]))
            .collect();
        let addresses = keys
            .iter()
            .map(|key| {
                verinoo_consensus::public_key_to_address(&hex::encode(
                    key.verifying_key().to_bytes(),
                ))
                .unwrap()
            })
            .collect();
        (keys, addresses)
    }

    fn store_test_block(shared: &Shared, height: u64) -> String {
        let hash = format!("{height:064x}");
        let block = verinoo_exec::engine::Block {
            header: verinoo_exec::engine::BlockHeader {
                hash: hash.clone(),
                version: verinoo_exec::engine::HEADER_VERSION,
                index: height,
                timestamp: height,
                previous_hash: format!("{:064x}", height.saturating_sub(1)),
                validator: "genesis".into(),
                validator_pub_key: String::new(),
                validator_signature: String::new(),
                state_root: "00".repeat(32),
                tx_root: "00".repeat(32),
                inference_tasks_root: "00".repeat(32),
                compute_root: "00".repeat(32),
            },
            transactions: vec![],
        };
        shared.store.put_block_by_hash(
            &hash,
            &serde_json::to_value(&block).expect("block serialize"),
        );
        verinoo_exec::engine::set_head(&shared.store, &block);
        hash
    }

    fn test_certificate(
        chain_id: &str,
        height: u64,
        block_hash: &str,
        keys: &[ed25519_dalek::SigningKey],
        addresses: &[String],
    ) -> CommitCertificate {
        let votes = keys
            .iter()
            .zip(addresses)
            .map(|(key, address)| {
                verinoo_consensus::Vote::sign(
                    chain_id,
                    verinoo_consensus::VoteKind::Precommit,
                    height,
                    0,
                    block_hash,
                    address,
                    key,
                )
            })
            .collect();
        CommitCertificate {
            chain_id: chain_id.into(),
            height,
            round: 0,
            block_hash: block_hash.into(),
            votes,
        }
    }

    fn store_commit(
        shared: &Shared,
        height: u64,
        block_hash: &str,
        certificate: &CommitCertificate,
    ) {
        shared.store.set_meta(
            &format!("commit:{height}"),
            &serde_json::to_value(certificate).expect("certificate serialize"),
        );
        if height > shared.store.meta_u64_or("finalizedHeight", 0) {
            shared.store.set_meta_u64("finalizedHeight", height);
            shared
                .store
                .set_meta("finalizedHash", &serde_json::json!(block_hash));
        }
    }

    #[tokio::test]
    async fn finalized_since_returns_ordered_block_certificate_pairs() {
        let (keys, addresses) = test_bft_validators(4);
        let params = GenesisParams {
            validators: addresses.clone(),
            ..GenesisParams::default()
        };
        let shared = test_shared_with(params);
        let chain_id = shared.params.chain_id.clone();
        for height in 1..=3 {
            let hash = store_test_block(&shared, height);
            let certificate = test_certificate(&chain_id, height, &hash, &keys, &addresses);
            store_commit(&shared, height, &hash, &certificate);
        }

        let Json(body) = finalized_since(State(shared.clone()), AxumPath(0)).await;
        let pairs = body.as_array().unwrap();
        assert_eq!(pairs.len(), 3);
        for (offset, pair) in pairs.iter().enumerate() {
            let height = offset as u64 + 1;
            assert_eq!(pair["block"]["header"]["index"], height);
            assert_eq!(pair["certificate"]["height"], height);
            assert_eq!(
                pair["certificate"]["blockHash"],
                pair["block"]["header"]["hash"]
            );
        }

        let Json(body) = finalized_since(State(shared), AxumPath(1)).await;
        let pairs = body.as_array().unwrap();
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0]["block"]["header"]["index"], 2);
        assert_eq!(pairs[1]["block"]["header"]["index"], 3);
    }

    #[tokio::test]
    async fn finalized_since_stops_at_a_missing_certificate_or_block() {
        let (keys, addresses) = test_bft_validators(4);
        let params = GenesisParams {
            validators: addresses.clone(),
            ..GenesisParams::default()
        };
        let chain_id = params.chain_id.clone();

        let shared = test_shared_with(params.clone());
        for height in 1..=3 {
            let hash = store_test_block(&shared, height);
            if height == 2 {
                continue;
            }
            let certificate = test_certificate(&chain_id, height, &hash, &keys, &addresses);
            store_commit(&shared, height, &hash, &certificate);
        }
        let Json(body) = finalized_since(State(shared), AxumPath(0)).await;
        assert_eq!(body.as_array().unwrap().len(), 1);
        assert_eq!(body[0]["block"]["header"]["index"], 1);

        let shared = test_shared_with(params.clone());
        for height in 1..=3 {
            if height == 2 {
                continue;
            }
            let hash = store_test_block(&shared, height);
            let certificate = test_certificate(&chain_id, height, &hash, &keys, &addresses);
            store_commit(&shared, height, &hash, &certificate);
        }
        let Json(body) = finalized_since(State(shared), AxumPath(0)).await;
        assert_eq!(body.as_array().unwrap().len(), 1);
        assert_eq!(body[0]["block"]["header"]["index"], 1);
    }

    #[tokio::test]
    async fn finalized_since_drops_tampered_certificates() {
        let (keys, addresses) = test_bft_validators(4);
        let params = GenesisParams {
            validators: addresses.clone(),
            ..GenesisParams::default()
        };
        let chain_id = params.chain_id.clone();

        let shared = test_shared_with(params.clone());
        for height in 1..=2 {
            let hash = store_test_block(&shared, height);
            let mut certificate = test_certificate(&chain_id, height, &hash, &keys, &addresses);
            if height == 1 {
                certificate.votes[0].signature = "00".repeat(64);
            }
            store_commit(&shared, height, &hash, &certificate);
        }
        let Json(body) = finalized_since(State(shared), AxumPath(0)).await;
        assert!(body.as_array().unwrap().is_empty());

        let shared = test_shared_with(params);
        for height in 1..=2 {
            let hash = store_test_block(&shared, height);
            let mut certificate = test_certificate(&chain_id, height, &hash, &keys, &addresses);
            if height == 2 {
                certificate.block_hash = "ff".repeat(32);
            }
            store_commit(&shared, height, &hash, &certificate);
        }
        let Json(body) = finalized_since(State(shared), AxumPath(0)).await;
        let pairs = body.as_array().unwrap();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0]["block"]["header"]["index"], 1);
    }
}
