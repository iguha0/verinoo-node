//! Verinoo testnet node — wires state + execution + networking + RPC.
//!
//! Hostile-network rule: NOTHING from the wire touches state without
//! passing `validate_block` (header hash, producer signature +
//! authorization, chain linkage, timestamp sanity, txRoot, duplicate
//! rejection, per-tx canonical + signature checks).

mod auth;
mod finality;
mod policy_gate;
mod readiness;
mod rpc;
mod snapshot;
mod wallet;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use verinoo_exec::engine::{
    commit_finalized, current_head, produce_block_with_report, propose_block_with_report, set_head,
    validate_block, Block,
};
use verinoo_network::NetworkEvent;
use verinoo_state::genesis::GenesisParams;
use verinoo_state::StateStore;

#[derive(Debug, Clone)]
pub struct NodeConfig {
    pub db_path: PathBuf,
    pub api_port: u16,
    pub p2p_port: u16,
    pub key_seed: [u8; 32],
    pub seed_peers: Vec<String>,
    pub validator: bool,
    pub genesis: GenesisParams,
}

pub struct Shared {
    pub store: StateStore,
    pub params: GenesisParams,
    /// Pending transactions waiting for block inclusion.
    pub mempool: tokio::sync::Mutex<Vec<verinoo_types::tx::TxEnvelope>>,
    /// Testnet faucet identity. Fixed and public by design: every node must
    /// derive the same funded genesis account, otherwise faucet drips would
    /// only exist on the node that served them.
    pub faucet: verinoo_exec::engine::ProducerKey,
    /// Serialises chain advancement between block production, gossip
    /// application and catch-up sync. Two writers without this can apply
    /// different blocks at the same height.
    pub chain_lock: tokio::sync::Mutex<()>,
    /// True while `sync_once` is fetching or applying peer blocks.
    pub syncing: AtomicBool,
    /// Highest peer height seen by catch-up sync; 0 means unknown.
    pub best_peer_height: AtomicU64,
    /// Effective block interval (env override or genesis default, min 500 ms).
    pub block_time_ms: u64,
    /// Allowed lag behind the best observed peer before `/ready` fails.
    pub peer_lag_blocks: u64,
    /// Node-local plaintext policy applied at the inference submission
    /// boundary, before a transaction enters the mempool.
    pub request_policy: verinoo_policy::Policy,
    /// Global cap on concurrent task event streams, shared by every request.
    pub task_streams: Arc<tokio::sync::Semaphore>,
    /// Optional directory served by `GET /snapshot/manifest` and
    /// `GET /snapshot/files/*path`. Reads are public chain state.
    pub snapshot_dir: Option<PathBuf>,
    /// BFT finality certificate manager. `None` in legacy mode.
    pub finality: Option<tokio::sync::Mutex<finality::FinalityManager>>,
    /// `legacy`, `bft` or `bft-v2` (VNN_CONSENSUS).
    pub consensus_mode: String,
    /// Parent directory for the RocksDB checkpoint clones `bft-v2` leaders
    /// build proposals on. Created on demand; clones are removed after use.
    pub proposal_work_dir: PathBuf,
    policy: std::sync::Mutex<Policy>,
}

/// Shared admission cap for RPC and gossip, so either path cannot grow the
/// mempool without bound.
pub const MAX_MEMPOOL_TRANSACTIONS: usize = 5_000;
/// Concurrent `/v1/tasks/:id/events` streams allowed per process.
pub const MAX_ACTIVE_TASK_STREAMS: usize = 64;

/// Funded at genesis on every node, so a drip is a real transfer that
/// replicates through blocks instead of local-only state.
const FAUCET_BALANCE: i64 = 1_000_000_000;
const FAUCET_SEED_LABEL: &str = "verinoo-testnet-faucet-v1";

#[derive(Default)]
struct Policy {
    seen: std::collections::HashSet<[u8; 32]>,
    seen_order: Vec<[u8; 32]>,
    bans: std::collections::HashMap<String, std::time::Instant>,
}

const MAX_SEEN: usize = 50_000;
const BAN_SECS: u64 = 600;

impl Shared {
    /// Duplicate suppression: false => already processed.
    pub fn mark_seen(&self, payload: &str) -> bool {
        use sha2::Digest as _;
        let mut h = sha2::Sha256::new();
        h.update(payload.as_bytes());
        let digest: [u8; 32] = h.finalize().into();
        let mut p = self.policy.lock().expect("policy lock");
        if p.seen.contains(&digest) {
            return false;
        }
        p.seen.insert(digest);
        p.seen_order.push(digest);
        if p.seen_order.len() > MAX_SEEN {
            let old = p.seen_order.remove(0);
            p.seen.remove(&old);
        }
        true
    }

    pub fn is_banned(&self, peer_or_reason: &str) -> bool {
        self.policy
            .lock()
            .expect("policy lock")
            .bans
            .get(peer_or_reason)
            .map(|t| *t > std::time::Instant::now())
            .unwrap_or(false)
    }

    pub fn ban(&self, key: &str, reason: &str) {
        self.policy.lock().expect("policy lock").bans.insert(
            key.into(),
            std::time::Instant::now() + std::time::Duration::from_secs(BAN_SECS),
        );
        tracing::warn!("banned {key}: {reason}");
    }
}

impl Shared {
    pub fn head(&self) -> Option<Block> {
        current_head(&self.store)
    }
    pub fn allowed_validators(&self) -> Vec<String> {
        self.params.validators.clone()
    }
    pub fn now_ms(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }
    /// Genesis/configured validator set, canonicalised.
    pub fn configured_validator_set(&self) -> Vec<String> {
        readiness::canonical_validator_set(&self.params.validators)
    }
    /// State-derived active validator set, canonicalised.
    pub fn state_validator_set(&self) -> Vec<String> {
        readiness::canonical_validator_set(&self.store.validator_addresses())
    }
    /// Active set from state, falling back to the genesis/configured list when
    /// state has no `val:` entries.
    pub fn effective_validator_set(&self) -> Vec<String> {
        let addresses = self.state_validator_set();
        if addresses.is_empty() {
            self.configured_validator_set()
        } else {
            addresses
        }
    }
    /// Fail-closed comparison of the state-derived set with the configured set.
    /// An unseeded state only matches an empty configured set.
    pub fn validator_set_consistent(&self) -> bool {
        self.state_validator_set() == self.configured_validator_set()
    }
    /// Observability hash of the effective active set (not part of the state root).
    pub fn validator_set_hash(&self) -> String {
        readiness::validator_set_hash(&self.effective_validator_set())
    }
    pub async fn finality_observe(&self, net: &NetworkNodeAlias, block: &Block) {
        let Some(manager) = &self.finality else {
            return;
        };
        let output = {
            let mut manager = manager.lock().await;
            manager.observe_block(block.header.index, &block.header.hash)
        };
        self.apply_finality_output(net, output).await;
    }

    pub async fn finality_vote(&self, net: &NetworkNodeAlias, vote: verinoo_consensus::Vote) {
        let Some(manager) = &self.finality else {
            return;
        };
        let output = {
            let mut manager = manager.lock().await;
            manager.on_vote(&vote)
        };
        self.apply_finality_output(net, output).await;
    }

    async fn apply_finality_output(
        &self,
        net: &NetworkNodeAlias,
        output: finality::FinalityOutput,
    ) {
        for vote in output.votes {
            let kind = vote.kind.as_str();
            self.store.set_meta(
                &format!("consensusOwnVote:{}:{}:{}", vote.height, vote.round, kind),
                &serde_json::to_value(&vote).expect("vote serialize"),
            );
            net.publish_vote(&serde_json::to_string(&vote).expect("vote serialize"));
        }
        for evidence in output.evidence {
            let id = evidence.id();
            self.store.set_meta(
                &format!("evid:{id}"),
                &serde_json::to_value(&evidence).expect("evidence serialize"),
            );
            tracing::warn!(
                height = evidence.height,
                validator = %evidence.validator,
                "equivocation evidence recorded"
            );
        }
        if let Some((height, block_hash, certificate)) = output.finalized {
            if let Err(e) = shared_finality_commit(self, height, &block_hash, &certificate) {
                // Certificate-gated mode: without a committed block there is
                // nothing final to persist; the head stays where it was.
                tracing::error!(
                    "bft-v2 did not finalize #{} {block_hash}: {e} — head not advanced",
                    height
                );
                return;
            }
            persist_finalized(self, height, &block_hash, &certificate);
            tracing::info!(
                "finalized block #{height} {}",
                &block_hash[..16.min(block_hash.len())]
            );
        }
    }

    pub fn best_peer_height(&self) -> Option<u64> {
        match self.best_peer_height.load(Ordering::Relaxed) {
            0 => None,
            height => Some(height),
        }
    }
    /// Replace the best peer height with this poll's result. `None` clears it,
    /// so a peer that disappears cannot poison readiness indefinitely.
    pub fn set_best_peer_height(&self, height: Option<u64>) {
        self.best_peer_height
            .store(height.unwrap_or(0), Ordering::Relaxed);
    }
}

/// Commit the block a finality certificate binds to the canonical store.
///
/// `bft` (v1) applies blocks optimistically on arrival, so a finality output
/// only persists metadata. `bft-v2` never applies an uncertified block: the
/// head can only advance here, after the stored proposal is looked up by
/// hash and the certificate is re-verified against the configured set.
/// A certificate for an unknown, invalid, or non-extending block is an error
/// and finality is not persisted.
fn shared_finality_commit(
    shared: &Shared,
    height: u64,
    block_hash: &str,
    certificate: &verinoo_consensus::CommitCertificate,
) -> Result<(), String> {
    if shared.consensus_mode != "bft-v2" {
        return Ok(());
    }
    if certificate.height != height || certificate.block_hash != block_hash {
        return Err("finalized certificate does not bind its height and block hash".into());
    }
    // Same canonicalisation the FinalityManager applies, so quorum checks
    // agree with the set that produced the certificate.
    let mut validators = shared.params.validators.clone();
    validators.sort();
    validators.dedup();
    verinoo_consensus::verify_certificate(certificate, &validators)
        .map_err(|e| format!("finalized certificate rejected: {e}"))?;
    let value = shared
        .store
        .get_block_by_hash(block_hash)
        .ok_or_else(|| format!("certificate for unknown block {block_hash}"))?;
    let block: Block = serde_json::from_value(value)
        .map_err(|e| format!("stored proposal {block_hash} is malformed: {e}"))?;
    if block.header.hash != block_hash || block.header.index != height {
        return Err("stored block does not match its certificate".into());
    }
    let head = shared.head().ok_or("no head")?;
    if head.header.index + 1 != height || head.header.hash != block.header.previous_hash {
        return Err(format!(
            "finalized block #{height} does not extend head #{} ({})",
            head.header.index,
            &head.header.hash[..16.min(head.header.hash.len())]
        ));
    }
    // Re-run the hostile-network gate: the block may have been stored for a
    // while and this is the last check before canonical state changes.
    validate_block(
        &block,
        &head.header,
        &shared.allowed_validators(),
        shared.now_ms(),
    )
    .map_err(|e| format!("finalized block #{height} re-validation failed: {e}"))?;
    let producer = block.header.validator.clone();
    commit_finalized(&shared.store, &block, &producer)
        .map_err(|e| format!("commit_finalized at #{height} failed: {e}"))
}

/// Persist finalized height/hash and the certificate that proves them.
/// Monotonic: a late or duplicate certificate can never move finality back.
fn persist_finalized(
    shared: &Shared,
    height: u64,
    block_hash: &str,
    certificate: &verinoo_consensus::CommitCertificate,
) {
    if height <= shared.store.meta_u64_or("finalizedHeight", 0) {
        return;
    }
    shared.store.set_meta_u64("finalizedHeight", height);
    shared
        .store
        .set_meta("finalizedHash", &serde_json::json!(block_hash));
    shared.store.set_meta(
        &format!("commit:{height}"),
        &serde_json::to_value(certificate).expect("certificate serialize"),
    );
}

/// Local finalized tip. The BFT manager is authoritative when present; legacy
/// mode falls back to the persisted metadata written by finality outputs.
async fn current_finalized(shared: &Shared) -> (u64, String) {
    if let Some(manager) = &shared.finality {
        let manager = manager.lock().await;
        return (
            manager.finalized_height(),
            manager.finalized_hash().to_string(),
        );
    }
    (
        shared.store.meta_u64_or("finalizedHeight", 0),
        shared
            .store
            .get_meta("finalizedHash")
            .and_then(|value| value.as_str().map(str::to_owned))
            .unwrap_or_default(),
    )
}

/// The hostile-network gate. Nothing from the wire touches state without
/// passing this.
fn admit_block(shared: &Shared, block: &Block) -> Result<(), String> {
    let head = shared.head().ok_or("no head")?;
    validate_block(
        block,
        &head.header,
        &shared.allowed_validators(),
        shared.now_ms(),
    )
    .map_err(|e| e.to_string())
}

/// Persist + execute an admitted block.
fn commit_block(shared: &Shared, block: &Block, producer_address: &str) -> bool {
    shared.store.put_block_by_hash(
        &block.header.hash,
        &serde_json::to_value(block).expect("block serialize"),
    );
    let base_fee = shared
        .store
        .meta_u64_or("baseFee", shared.params.initial_base_fee);
    if let Err(e) =
        verinoo_exec::engine::apply_block(&shared.store, block, base_fee, producer_address)
    {
        tracing::error!("apply failed after admission: {e}");
        return false;
    }
    // Execute-by-consensus check: the block's state root must commit to the
    // outputs this node derived. A missing model file or a divergent runtime
    // shows up here instead of silently forking.
    let root = verinoo_exec::engine::compute_state_root(&shared.store);
    if root != block.header.state_root {
        let ours = &root[..16.min(root.len())];
        let theirs = &block.header.state_root[..16.min(block.header.state_root.len())];
        tracing::error!(
            "state root mismatch at #{} (ours {ours}, block {theirs}) — model set differs? refusing to advance",
            block.header.index
        );
        return false;
    }
    set_head(&shared.store, block);
    true
}

/// Put transactions the producer skipped only for gas back at the front of the
/// mempool, preserving their batch order. Invalid or reverted transactions are
/// intentionally dropped, and IDs already re-admitted by gossip are not
/// duplicated.
fn requeue_gas_skipped(
    mempool: &mut Vec<verinoo_types::tx::TxEnvelope>,
    batch: &[verinoo_types::tx::TxEnvelope],
    skipped: &[(String, String)],
) {
    let requeued: Vec<verinoo_types::tx::TxEnvelope> = batch
        .iter()
        .filter(|tx| {
            skipped.iter().any(|(tx_id, reason)| {
                tx_id == &tx.tx_id && reason == verinoo_exec::engine::GAS_LIMIT_SKIP_REASON
            })
        })
        .filter(|tx| !mempool.iter().any(|existing| existing.tx_id == tx.tx_id))
        .cloned()
        .collect();
    if requeued.is_empty() {
        return;
    }
    let mut merged = Vec::with_capacity(requeued.len() + mempool.len());
    merged.extend(requeued);
    for tx in mempool.drain(..) {
        if !merged.iter().any(|existing| existing.tx_id == tx.tx_id) {
            merged.push(tx);
        }
    }
    *mempool = merged;
}

/// Validator loop: produce on interval from the mempool and broadcast.
async fn validator_loop(
    shared: Arc<Shared>,
    net: NetworkNodeAlias,
    sk: ed25519_dalek::SigningKey,
    block_time_ms: u64,
) {
    use sha2::Digest as _;
    let vk = sk.verifying_key();
    let pk_hex = hex::encode(vk.to_bytes());
    let mut h = sha2::Sha256::new();
    h.update(hex::decode(&pk_hex).unwrap());
    let addr = format!("ai_{}", hex::encode(&h.finalize()[..20]));

    let producer = verinoo_exec::engine::ProducerKey::from_parts(addr.clone(), sk);
    let mut ticker =
        tokio::time::interval(std::time::Duration::from_millis(block_time_ms.max(500)));
    use verinoo_exec::leader::LeaderPolicy as _;
    // `interval` fires immediately; consume that tick so peers have one block
    // time to connect and exchange gossip subscriptions first. Producing at
    // t=0 publishes into a mesh that does not exist yet and the block is lost.
    ticker.tick().await;

    // bft-v2 round 0 proposes each height exactly once; the pacemaker that
    // would retry or advance rounds is Phase B. Track the last height this
    // leader proposed so the head not moving (no certificate yet) does not
    // stack a second, conflicting proposal on the same height.
    let bft_v2 = shared.consensus_mode == "bft-v2";
    let mut last_proposed: u64 = 0;

    loop {
        ticker.tick().await;
        let Some(head) = shared.head() else {
            continue;
        };
        // Produce only in this validator's scheduled slot, otherwise the net
        // gets competing blocks at the same height and forks.
        let next_height = head.header.index + 1;
        let is_leader = match verinoo_exec::leader::RoundRobin
            .leader_at(next_height, &shared.allowed_validators())
        {
            Some(leader) => leader == addr,
            None => true,
        };
        if !is_leader {
            continue;
        }
        // An isolated validator would produce a block nobody else receives;
        // with a single-validator set that is exactly the expected behavior.
        if shared.allowed_validators().len() > 1 && net.peer_count() == 0 {
            tracing::debug!(
                "no peers connected — waiting before producing at height {next_height}"
            );
            continue;
        }
        if bft_v2 && next_height <= last_proposed {
            // Still waiting for the round-0 proposal to gather a certificate.
            continue;
        }
        let batch: Vec<_> = shared.mempool.lock().await.drain(..).collect();

        let _guard = shared.chain_lock.lock().await;
        if bft_v2 {
            // Proposal path: build on a checkpoint clone, store by hash, and
            // publish. The canonical head is NOT advanced here; it moves only
            // when the finality manager emits Finalized with a certificate
            // (see `shared_finality_commit`).
            match propose_block_with_report(
                &shared.store,
                &batch,
                &producer,
                shared.now_ms(),
                &shared.proposal_work_dir,
            ) {
                Ok(report) => {
                    let block = report.block;
                    let n = block.transactions.len();
                    last_proposed = block.header.index;
                    shared.store.put_block_by_hash(
                        &block.header.hash,
                        &serde_json::to_value(&block).expect("block serialize"),
                    );
                    {
                        let mut mempool = shared.mempool.lock().await;
                        requeue_gas_skipped(&mut mempool, &batch, &report.skipped);
                    }
                    tracing::info!(
                        "proposed block #{} | {} txs | {}",
                        block.header.index,
                        n,
                        &block.header.hash[..16.min(block.header.hash.len())]
                    );
                    net.publish_block(&serde_json::to_string(&block).expect("block serialize"));
                    shared.finality_observe(&net, &block).await;
                }
                Err(e) => {
                    shared.mempool.lock().await.extend(batch);
                    tracing::error!("propose failed: {e}");
                }
            }
            continue;
        }
        match produce_block_with_report(&shared.store, &batch, &producer, shared.now_ms()) {
            Ok(report) => {
                let block = report.block;
                let n = block.transactions.len();
                set_head(&shared.store, &block);
                {
                    let mut mempool = shared.mempool.lock().await;
                    requeue_gas_skipped(&mut mempool, &batch, &report.skipped);
                }
                tracing::info!(
                    "produced block #{} | {} txs | {}",
                    block.header.index,
                    n,
                    &block.header.hash[..16.min(block.header.hash.len())]
                );
                net.publish_block(&serde_json::to_string(&block).expect("block serialize"));
                shared.finality_observe(&net, &block).await;
            }
            Err(e) => {
                shared.mempool.lock().await.extend(batch);
                tracing::error!("produce failed: {e}");
            }
        }
    }
}

type NetworkNodeAlias = verinoo_network::NetworkNode;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    // Wallet subcommands
    if args.len() > 1 && args[1] == "wallet" {
        return wallet_cli(&args[2..]).await;
    }

    // Inference client: submit a task, print its receipt handle.
    if args.len() > 1 && args[1] == "infer" {
        return infer_cli(&args[2..]).await;
    }

    // Validator address for a seed. Lets operators build VNN_VALIDATORS before
    // any node starts, instead of booting one to discover its own address.
    if args.len() > 1 && args[1] == "address" {
        return address_cli(&args[2..]);
    }

    // API-key administration (issue / list / revoke tenant keys).
    if args.len() > 1 && args[1] == "apikey" {
        return auth::cli(&args[2..]).map_err(|e| e.into());
    }

    // Signed state snapshots: create, verify, apply.
    if args.len() > 1 && args[1] == "snapshot" {
        return snapshot::cli(&args[2..]);
    }

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    // API keys are optional: no file (and no VNN_API_KEYS_FILE) means the
    // public demo mode where any funded key may submit.
    let api_auth = auth::Auth::from_env()?;
    if api_auth.is_enabled() {
        tracing::info!(
            "api keys enabled ({} entries) | {}",
            api_auth.len(),
            api_auth
                .path()
                .map(|p| p.display().to_string())
                .unwrap_or_default()
        );
        api_auth.spawn_reload_on_sighup();
    } else {
        tracing::warn!("api keys disabled — any funded key may submit inference");
    }

    // Input encryption uses one network X25519 key shared by every validator.
    // Without it, inputs are plaintext in blocks and API state.
    if let Ok(raw) = std::env::var("VNN_INPUT_KEY") {
        if !raw.trim().is_empty() {
            let key = verinoo_crypto::NetworkKey::from_hex(raw.trim())?;
            let public = key.public_key_hex();
            verinoo_crypto::install_input_key(key)
                .map_err(|_| "VNN_INPUT_KEY was already installed".to_string())?;
            tracing::info!("input encryption enabled | network key {}", &public[..16]);
        }
    }
    if verinoo_crypto::input_key().is_none() {
        tracing::warn!(
            "input encryption disabled — plaintext inputs are public; set VNN_INPUT_KEY"
        );
    }

    // Config via env (deploy-friendly): VNN_* variables.
    let validator = std::env::var("VNN_VALIDATOR")
        .map(|v| v == "1" || v == "true")
        .unwrap_or(false);
    let key_seed_raw = std::env::var("VNN_KEY_SEED").unwrap_or_default();
    // A validator with a default/empty seed has a publicly derivable signing
    // key: anyone could forge blocks and its PeerId would collide with every
    // other node started without a seed. Refuse rather than run unsafely.
    if validator && key_seed_raw.trim().is_empty() {
        return Err("VNN_KEY_SEED must be set to a unique value when VNN_VALIDATOR=true".into());
    }

    let key_seed: [u8; 32] = {
        use sha2::Digest as _;
        let mut h = sha2::Sha256::new();
        h.update(key_seed_raw.as_bytes());
        h.finalize().into()
    };
    let signing = ed25519_dalek::SigningKey::from_bytes(&key_seed);
    let own_address = verinoo_exec::engine::public_key_to_address(&hex::encode(
        signing.verifying_key().to_bytes(),
    ))?;

    // Active validator set. Blocks from producers outside this set are
    // rejected, so an empty set means a node that can never accept a peer's
    // block. A lone validator therefore authorizes itself by default; a
    // multi-node testnet must pass the full set via VNN_VALIDATORS.
    let mut genesis = GenesisParams::default();
    if let Ok(raw) = std::env::var("VNN_VALIDATORS") {
        genesis.validators = raw
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect();
    }
    if genesis.validators.is_empty() && validator {
        genesis.validators.push(own_address.clone());
        tracing::warn!(
            "VNN_VALIDATORS unset — running single-validator with {}; peers will reject its blocks",
            own_address
        );
    }

    let cfg = NodeConfig {
        db_path: PathBuf::from(std::env::var("VNN_DB").unwrap_or_else(|_| "./vnn-data".into())),
        api_port: std::env::var("VNN_API_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(7100),
        p2p_port: std::env::var("VNN_P2P_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(7101),
        key_seed,
        seed_peers: std::env::var("VNN_SEEDS")
            .map(|v| {
                v.split(',')
                    .filter(|s| !s.is_empty())
                    .map(String::from)
                    .collect()
            })
            .unwrap_or_default(),
        validator,
        genesis,
    };

    let faucet_seed: [u8; 32] = {
        use sha2::Digest as _;
        let mut h = sha2::Sha256::new();
        h.update(FAUCET_SEED_LABEL.as_bytes());
        h.finalize().into()
    };
    let faucet_signing = ed25519_dalek::SigningKey::from_bytes(&faucet_seed);
    let faucet_address = verinoo_exec::engine::public_key_to_address(&hex::encode(
        faucet_signing.verifying_key().to_bytes(),
    ))?;
    let faucet = verinoo_exec::engine::ProducerKey::from_parts(faucet_address, faucet_signing);

    let store = StateStore::open(&cfg.db_path)?;

    // Bind the database to a chain id so snapshot apply can reject a snapshot
    // from another chain without trusting the operator to pass the right one.
    match store
        .get_meta("chainId")
        .and_then(|value| value.as_str().map(str::to_owned))
    {
        Some(existing) if existing != cfg.genesis.chain_id => {
            return Err(format!(
                "database chain id mismatch: database has `{existing}`, configured chain is `{}`",
                cfg.genesis.chain_id
            )
            .into());
        }
        Some(_) => {}
        None => store.set_meta("chainId", &serde_json::json!(cfg.genesis.chain_id)),
    }

    // Model set: graph models executed deterministically inside block
    // application. Every node in a network must run the same set, otherwise
    // the state root diverges and peers reject each other's blocks.
    let model_dir = std::env::var("VNN_MODEL_DIR").unwrap_or_else(|_| "./models".into());
    let (registry, problems) = verinoo_runtime::Registry::load_dir(&model_dir);
    for problem in &problems {
        tracing::warn!("model set: {problem}");
    }
    if registry.is_empty() {
        tracing::debug!("no graph models loaded from {model_dir}");
    } else {
        for id in registry.ids() {
            tracing::info!("model loaded | {id}");
        }
        tracing::info!(
            "model set: {} model(s) | hash {}",
            registry.len(),
            &registry.hash()[..16.min(registry.hash().len())]
        );
    }
    verinoo_runtime::install_once(registry);

    // Transformer encoder models are a separate runtime but part of the same
    // consensus model set. Every validator must load the same files.
    let (transformers, transformer_problems) =
        verinoo_runtime::transformer::TransformerRegistry::load_dir(&model_dir);
    for problem in &transformer_problems {
        tracing::warn!("transformer set: {problem}");
    }
    if !transformers.is_empty() {
        for id in transformers.ids() {
            tracing::info!("transformer loaded | {id}");
        }
        tracing::info!(
            "transformer set: {} model(s) | hash {}",
            transformers.len(),
            &transformers.hash()[..16.min(transformers.hash().len())]
        );
    }
    verinoo_runtime::transformer::install_transformers_once(transformers);

    // Generative decoder models: BPE tokenizer, causal attention, KV cache and
    // greedy decoding. Same consensus rule: every validator loads the same set.
    let (decoders, decoder_problems) = verinoo_runtime::gpt::GptRegistry::load_dir(&model_dir);
    for problem in &decoder_problems {
        tracing::warn!("decoder set: {problem}");
    }
    if !decoders.is_empty() {
        for id in decoders.ids() {
            tracing::info!("decoder loaded | {id}");
        }
        tracing::info!(
            "decoder set: {} model(s) | hash {}",
            decoders.len(),
            &decoders.hash()[..16.min(decoders.hash().len())]
        );
    }
    verinoo_runtime::gpt::install_gpt_once(decoders);

    // Readiness uses the same effective interval as block production.
    let block_time_ms = std::env::var("VNN_BLOCK_TIME_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(cfg.genesis.block_time_ms)
        .max(500);
    if block_time_ms != cfg.genesis.block_time_ms {
        tracing::info!("block time override: {block_time_ms} ms");
    }
    let peer_lag_blocks = std::env::var("VNN_PEER_LAG_BLOCKS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(2);

    // A configured policy must load and validate; falling back to permissive
    // here would silently disable an operator's blocklist.
    let request_policy = policy_gate::load_from_env()?;
    let snapshot_dir = std::env::var("VNN_SNAPSHOT_DIR")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .map(PathBuf::from);
    if let Some(dir) = &snapshot_dir {
        tracing::info!("serving snapshots from {}", dir.display());
    }
    let policy_config = request_policy.config();
    tracing::info!(
        "request policy | maxInputBytes={} | allowedModels={} | blockedTerms={} | blockedPatterns={} | requirePlaintext={}",
        policy_config.max_input_bytes,
        policy_config.allowed_models.len(),
        policy_config.blocked_terms.len(),
        policy_config.blocked_patterns.len(),
        policy_config.require_plaintext,
    );

    let consensus_mode = std::env::var("VNN_CONSENSUS").unwrap_or_else(|_| "legacy".into());
    if !matches!(consensus_mode.as_str(), "legacy" | "bft" | "bft-v2") {
        return Err(format!(
            "VNN_CONSENSUS must be `legacy`, `bft`, or `bft-v2`, got `{consensus_mode}`"
        )
        .into());
    }
    if consensus_mode != "legacy" && !validator {
        return Err(format!("VNN_CONSENSUS={consensus_mode} requires VNN_VALIDATOR=true").into());
    }
    let finalized_height = store.meta_u64_or("finalizedHeight", 0);
    let finalized_hash = store
        .get_meta("finalizedHash")
        .and_then(|value| value.as_str().map(str::to_owned))
        .unwrap_or_default();
    let finality = if consensus_mode != "legacy" {
        Some(tokio::sync::Mutex::new(finality::FinalityManager::new(
            &cfg.genesis.chain_id,
            cfg.genesis.validators.clone(),
            &own_address,
            ed25519_dalek::SigningKey::from_bytes(&key_seed),
            finalized_height,
            &finalized_hash,
            block_time_ms.saturating_mul(2),
        )?))
    } else {
        None
    };
    // Checkpoint clones for bft-v2 proposals live under this directory.
    let proposal_work_dir = std::env::var("VNN_PROPOSE_WORK_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| cfg.db_path.join("propose-work"));
    if consensus_mode == "bft-v2" {
        tracing::info!("bft-v2: head advances only on a 2f+1 commit certificate");
    }

    let shared = Arc::new(Shared {
        store,
        params: cfg.genesis.clone(),
        mempool: tokio::sync::Mutex::new(Vec::new()),
        faucet,
        chain_lock: tokio::sync::Mutex::new(()),
        syncing: AtomicBool::new(false),
        best_peer_height: AtomicU64::new(0),
        block_time_ms,
        peer_lag_blocks,
        request_policy,
        task_streams: Arc::new(tokio::sync::Semaphore::new(MAX_ACTIVE_TASK_STREAMS)),
        snapshot_dir,
        finality,
        consensus_mode,
        proposal_work_dir,
        policy: std::sync::Mutex::new(Policy::default()),
    });
    let (net, mut event_rx) = verinoo_network::NetworkNode::new(
        cfg.p2p_port,
        cfg.key_seed,
        &cfg.seed_peers,
        &shared.params.chain_id,
    )
    .await?;

    tracing::info!(
        "verinoo-node up | chain={} | api=:{} | p2p=:{} | validator={}",
        shared.params.chain_id,
        cfg.api_port,
        cfg.p2p_port,
        cfg.validator
    );

    // Genesis allocations (VNN_GENESIS_ALLOCATIONS = {"addr": amount})
    // applied once when the chain is empty.
    if shared.head().is_none() {
        if let Ok(raw) = std::env::var("VNN_GENESIS_ALLOCATIONS") {
            if let Ok(alloc) = serde_json::from_str::<serde_json::Value>(&raw) {
                if let Some(map) = alloc.as_object() {
                    for (addr, bal) in map {
                        let mut acc = shared.store.account_or_default(addr.as_str());
                        acc.balance = bal.as_i64().unwrap_or(0);
                        shared.store.set_account(&acc);
                    }
                    tracing::info!("genesis allocations applied ({} entries)", map.len());
                }
            }
        }
    }

    // Fund the faucet once, at genesis, identically on every node.
    if shared.head().is_none() {
        let mut acc = shared.store.account_or_default(&shared.faucet.address);
        acc.balance = FAUCET_BALANCE;
        shared.store.set_account(&acc);
        tracing::info!("faucet account funded | {}", shared.faucet.address);
    }

    // Genesis block when the chain is empty (mirrors reference engine).
    if shared.head().is_none() {
        let mut h = verinoo_exec::engine::BlockHeader {
            hash: String::new(),
            version: 1,
            index: 0,
            timestamp: 0,
            previous_hash: "0".repeat(64),
            validator: "genesis".into(),
            validator_pub_key: String::new(),
            validator_signature: String::new(),
            state_root: sha256_hex_str("genesis"),
            tx_root: sha256_hex_str("empty"),
            inference_tasks_root: sha256_hex_str("empty-tasks"),
            compute_root: sha256_hex_str("empty-compute"),
        };
        h.hash = verinoo_exec::engine::hash_header(&h);
        let genesis = Block {
            header: h.clone(),
            transactions: vec![],
        };
        shared
            .store
            .put_block_by_hash(&h.hash, &serde_json::to_value(&genesis).unwrap());
        verinoo_exec::engine::set_head(&shared.store, &genesis);
        tracing::info!("genesis initialized | {}", &h.hash[..16.min(h.hash.len())]);
    }

    // Seed the active validator set (idempotent across restarts).
    shared
        .store
        .seed_genesis_validators(&shared.params.validators, 0);
    if !shared.params.validators.is_empty() {
        tracing::info!(
            "active validators ({}): {}",
            shared.params.validators.len(),
            shared.params.validators.join(",")
        );
    }

    // Existing databases predate the height index; walk back once on startup
    // so /blocks/since can serve ranges immediately.
    backfill_height_index(&shared.store);

    // RPC server.
    tokio::spawn(rpc::serve(
        shared.clone(),
        net.clone(),
        cfg.api_port,
        api_auth.clone(),
    ));

    // Event pump consumes the wire stream.
    let pump_shared = shared.clone();
    let pump_net = net.clone();
    tokio::spawn(async move {
        let shared = pump_shared;
        let net = pump_net;
        while let Some(ev) = event_rx.recv().await {
            match ev {
                NetworkEvent::Tx(payload) => {
                    if !shared.mark_seen(&payload) {
                        continue;
                    }
                    match serde_json::from_str::<verinoo_types::tx::TxEnvelope>(&payload) {
                        Ok(tx) => {
                            let mut mp = shared.mempool.lock().await;
                            if !mp.iter().any(|t| t.tx_id == tx.tx_id)
                                && mp.len() < MAX_MEMPOOL_TRANSACTIONS
                            {
                                mp.push(tx);
                            }
                        }
                        Err(e) => tracing::debug!("bad tx payload: {e}"),
                    }
                }
                NetworkEvent::Block(payload) => {
                    if !shared.mark_seen(&payload) {
                        continue;
                    }
                    match serde_json::from_str::<Block>(&payload) {
                        Ok(block) => {
                            let _guard = shared.chain_lock.lock().await;
                            match admit_block(&shared, &block) {
                                Ok(()) => {
                                    let producer = block.header.validator.clone();
                                    if shared.consensus_mode == "bft-v2" {
                                        // Receiver path: store the proposal by
                                        // hash and vote, but do not apply or
                                        // advance the head. The head moves only
                                        // when the manager emits Finalized
                                        // with a certificate.
                                        shared.store.put_block_by_hash(
                                            &block.header.hash,
                                            &serde_json::to_value(&block).expect("block serialize"),
                                        );
                                        tracing::info!(
                                            "admitted proposal #{} from {}",
                                            block.header.index,
                                            &producer[..12.min(producer.len())]
                                        );
                                        shared.finality_observe(&net, &block).await;
                                    } else {
                                        tracing::info!(
                                            "admitted block #{} from {}",
                                            block.header.index,
                                            &producer[..12.min(producer.len())]
                                        );
                                        if commit_block(&shared, &block, &producer) {
                                            shared.finality_observe(&net, &block).await;
                                        }
                                    }
                                }
                                Err(reason) => {
                                    tracing::warn!("rejected block: {reason}");
                                }
                            }
                        }
                        Err(e) => tracing::debug!("bad block payload: {e}"),
                    }
                }
                NetworkEvent::Proposal(_payload) => {
                    // Round > 0 proposals are not used by the single-producer
                    // finality integration; the block topic is the round-0
                    // proposal.
                }
                NetworkEvent::Vote(payload) => {
                    match serde_json::from_str::<verinoo_consensus::Vote>(&payload) {
                        Ok(vote) => {
                            if shared.consensus_mode == "bft-v2" {
                                // A vote quorum can commit to the canonical
                                // store here; hold the chain lock so commits
                                // never interleave with proposal production.
                                let _guard = shared.chain_lock.lock().await;
                                shared.finality_vote(&net, vote).await;
                            } else {
                                shared.finality_vote(&net, vote).await;
                            }
                        }
                        Err(e) => tracing::debug!("bad vote payload: {e}"),
                    }
                }
                NetworkEvent::Evidence(payload) => {
                    match serde_json::from_str::<verinoo_consensus::Evidence>(&payload) {
                        Ok(evidence) => {
                            if evidence.verify(&shared.params.validators).is_ok() {
                                let id = evidence.id();
                                shared.store.set_meta(
                                    &format!("evid:{id}"),
                                    &serde_json::to_value(&evidence).expect("evidence serialize"),
                                );
                            }
                        }
                        Err(e) => tracing::debug!("bad evidence payload: {e}"),
                    }
                }
                NetworkEvent::PeerBanned(peer, reason) => {
                    tracing::warn!("peer banned: {peer} — {reason}");
                }
            }
        }
    });

    // Validator loop (optional). Block time is a network-wide setting: every
    // validator must use the same value or slots drift.
    if cfg.validator {
        let v_shared = shared.clone();
        let v_net = net.clone();
        tokio::spawn(validator_loop(v_shared, v_net, signing, block_time_ms));
    }

    // Catch-up sync: poll peer HTTP APIs and replay missing blocks through the
    // same validation as gossip. A node that starts late or falls behind can
    // recover instead of stalling the network.
    let sync_urls: Vec<String> = std::env::var("VNN_SYNC_URLS")
        .map(|raw| {
            raw.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect()
        })
        .unwrap_or_default();
    if !sync_urls.is_empty() {
        tracing::info!("catch-up sync from {}", sync_urls.join(","));
        tokio::spawn(sync_loop(shared.clone(), sync_urls));
    }

    // Idle forever; tasks do the work.
    std::future::pending::<()>().await;
    Ok(())
}

/// Walk back from the head once and record height -> hash for each block that
/// predates the index. Stops at the first indexed height so restarts are cheap.
fn backfill_height_index(store: &verinoo_state::StateStore) {
    let Some(head) = current_head(store) else {
        return;
    };
    let mut cur = Some(head);
    let mut indexed = 0usize;
    while let Some(block) = cur {
        let key = format!("blockheight:{}", block.header.index);
        if store.get_meta(&key).is_some() {
            break;
        }
        store.set_meta(&key, &serde_json::json!(block.header.hash));
        indexed += 1;
        if block.header.index == 0 {
            break;
        }
        cur = store
            .get_block_by_hash(&block.header.previous_hash)
            .and_then(|v| serde_json::from_value::<Block>(v).ok());
    }
    if indexed > 0 {
        tracing::info!("indexed {indexed} block height(s) for catch-up sync");
    }
}

/// A silent peer must not hold a request forever: bound connection setup,
/// per-read waits and the whole request.
const SYNC_CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);
const SYNC_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);
const SYNC_REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

struct SyncReport {
    applied: usize,
    peer_height: Option<u64>,
}

async fn sync_loop(shared: Arc<Shared>, urls: Vec<String>) {
    let client = match reqwest::Client::builder()
        .connect_timeout(SYNC_CONNECT_TIMEOUT)
        .read_timeout(SYNC_READ_TIMEOUT)
        .timeout(SYNC_REQUEST_TIMEOUT)
        .build()
    {
        Ok(client) => client,
        Err(e) => {
            tracing::error!("catch-up sync client failed: {e}");
            return;
        }
    };
    let mut ticker = tokio::time::interval(std::time::Duration::from_secs(3));
    loop {
        ticker.tick().await;
        let mut best_peer_height = None;
        for url in &urls {
            match sync_once(&shared, &client, url).await {
                Ok(report) => {
                    if let Some(height) = report.peer_height {
                        best_peer_height =
                            Some(best_peer_height.map_or(height, |best: u64| best.max(height)));
                    }
                    if report.applied > 0 {
                        tracing::info!("synced {} block(s) from {url}", report.applied);
                    }
                }
                Err(e) => tracing::debug!("sync from {url} failed: {e}"),
            }
        }
        // Refresh (not fetch_max) so a stale or wrong-chain height cannot make
        // /ready report behind_peer forever.
        shared.set_best_peer_height(best_peer_height);
    }
}

/// Clears `Shared::syncing` on every exit path from `sync_once`.
struct SyncingGuard<'a>(&'a AtomicBool);

impl<'a> SyncingGuard<'a> {
    fn enter(flag: &'a AtomicBool) -> Self {
        flag.store(true, Ordering::Relaxed);
        Self(flag)
    }
}

impl Drop for SyncingGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
struct FinalizedPair {
    block: Block,
    certificate: verinoo_consensus::CommitCertificate,
}

/// Fetch finalized pairs from a peer. `Ok(None)` means the peer does not serve
/// `/finalized/since` (older build), so the caller can fall back to plain block
/// catch-up without changing local finality.
async fn fetch_finalized_pairs(
    client: &reqwest::Client,
    base: &str,
    from: u64,
) -> Result<Option<Vec<FinalizedPair>>, String> {
    let response = client
        .get(format!("{base}/finalized/since/{from}"))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    let status = response.status();
    if matches!(
        status,
        reqwest::StatusCode::NOT_FOUND
            | reqwest::StatusCode::METHOD_NOT_ALLOWED
            | reqwest::StatusCode::NOT_IMPLEMENTED
    ) {
        return Ok(None);
    }
    if !status.is_success() {
        return Err(format!("peer /finalized/since returned HTTP {status}"));
    }
    response
        .json::<Vec<FinalizedPair>>()
        .await
        .map(Some)
        .map_err(|e| e.to_string())
}

/// Verify a peer-served finalized run, apply the blocks through the same
/// admission/state-root gate as gossip, then persist finality. Certificates
/// are checked against the configured set and bound to their block hash before
/// any block or finality marker is touched, so an uncertified block can never
/// advance finalized height.
async fn apply_finalized_pairs(
    shared: &Shared,
    pairs: Vec<FinalizedPair>,
) -> Result<usize, String> {
    let mut adopted = 0usize;
    for pair in pairs {
        let FinalizedPair { block, certificate } = pair;
        let height = block.header.index;
        if certificate.height != height {
            return Err(format!(
                "finalized certificate height {} does not match block #{height}",
                certificate.height
            ));
        }
        if certificate.block_hash != block.header.hash {
            return Err(format!(
                "finalized certificate at #{height} does not bind the block hash"
            ));
        }
        if certificate.chain_id != shared.params.chain_id {
            return Err(format!(
                "finalized certificate at #{height} chain id mismatch"
            ));
        }
        verinoo_consensus::verify_certificate(&certificate, &shared.params.validators)
            .map_err(|e| format!("finalized certificate at #{height} rejected: {e}"))?;

        let (local_finalized, _) = current_finalized(shared).await;
        if height <= local_finalized {
            continue;
        }
        if height != local_finalized + 1 {
            tracing::warn!(
                height = height,
                finalized = local_finalized,
                "finalized catch-up has a gap; stopping"
            );
            break;
        }

        {
            let _guard = shared.chain_lock.lock().await;
            let head_height = shared.head().map(|head| head.header.index).unwrap_or(0);
            if height <= head_height {
                // Already replayed: the certificate must still bind the block
                // this node has at that height, otherwise refuse to finalize.
                let local_hash = shared
                    .store
                    .get_meta(&format!("blockheight:{height}"))
                    .and_then(|value| value.as_str().map(String::from))
                    .unwrap_or_default();
                if local_hash != block.header.hash {
                    return Err(format!(
                        "finalized certificate at #{height} does not match the local block"
                    ));
                }
            } else if height == head_height + 1 {
                admit_block(shared, &block)?;
                let producer = block.header.validator.clone();
                if shared.consensus_mode == "bft-v2" {
                    // Same commit gate as the gossip path: apply, verify the
                    // derived state root, only then advance the head. The
                    // certificate binding this block was verified above.
                    verinoo_exec::engine::commit_finalized(&shared.store, &block, &producer)
                        .map_err(|e| {
                            format!(
                                "finalized block #{height} did not advance head (state root or model set mismatch?): {e}"
                            )
                        })?;
                } else if !commit_block(shared, &block, &producer) {
                    return Err(format!(
                        "finalized block #{height} did not advance head (state root or model set mismatch?)"
                    ));
                }
            } else {
                return Err(format!(
                    "finalized block #{height} skips local head #{head_height}"
                ));
            }
        }

        persist_finalized(shared, height, &block.header.hash, &certificate);
        if let Some(manager) = &shared.finality {
            manager
                .lock()
                .await
                .adopt_finalized(height, &block.header.hash);
        }
        adopted += 1;
    }
    Ok(adopted)
}

async fn sync_once(
    shared: &Shared,
    client: &reqwest::Client,
    base: &str,
) -> Result<SyncReport, String> {
    let base = base.trim_end_matches('/');
    // The routine peer-height poll is not "syncing": /ready must stay 200 while
    // this request is in flight, and the client timeouts bound it.
    let status: serde_json::Value = client
        .get(format!("{base}/status"))
        .send()
        .await
        .map_err(|e| e.to_string())?
        .json()
        .await
        .map_err(|e| e.to_string())?;
    let peer_chain_id = status["chainId"]
        .as_str()
        .ok_or_else(|| "peer /status is missing chainId".to_string())?;
    if peer_chain_id != shared.params.chain_id {
        return Err(format!(
            "peer chainId mismatch: expected {}, got {peer_chain_id}",
            shared.params.chain_id
        ));
    }
    let peer_height = match status["height"].as_u64() {
        Some(height) => height,
        None => {
            return Ok(SyncReport {
                applied: 0,
                peer_height: None,
            })
        }
    };
    // BFT nodes exchange certificates, not just blocks: a late node adopts
    // finalized history first so it never votes for heights that are already
    // final. A peer without the endpoint still gets the legacy block path and
    // leaves local finality untouched. In bft-v2 this is the only path that
    // may move the head during catch-up: certificates are verified before any
    // block is applied.
    let mut finalized_applied = 0usize;
    if shared.consensus_mode != "legacy" {
        let peer_finalized = status["finalizedHeight"].as_u64().unwrap_or(0);
        let (local_finalized, _) = current_finalized(shared).await;
        if peer_finalized > local_finalized {
            match fetch_finalized_pairs(client, base, local_finalized).await? {
                Some(pairs) => {
                    let _syncing = SyncingGuard::enter(&shared.syncing);
                    let adopted = apply_finalized_pairs(shared, pairs).await?;
                    finalized_applied = adopted;
                    if adopted > 0 {
                        tracing::info!("adopted {adopted} finalized block(s) from {base}");
                    }
                }
                None => {
                    if shared.consensus_mode == "bft-v2" {
                        tracing::warn!(
                            "peer {base} does not serve /finalized/since; bft-v2 catch-up is certificate-only"
                        );
                    } else {
                        tracing::debug!(
                            "peer {base} does not serve /finalized/since; falling back to block catch-up"
                        );
                    }
                }
            }
        }
    }
    let our_height = shared.head().map(|b| b.header.index).unwrap_or(0);
    if peer_height <= our_height {
        return Ok(SyncReport {
            applied: finalized_applied,
            peer_height: Some(peer_height),
        });
    }
    if shared.consensus_mode == "bft-v2" {
        // Plain block catch-up would apply uncertified blocks; the bft-v2 head
        // advances only on a verified commit certificate.
        let frontier = our_height + 1;
        return Err(format!(
            "peer {base} has uncertified blocks #{frontier}..#{peer_height}; \
             bft-v2 refuses to adopt blocks without certificates"
        ));
    }
    let blocks: Vec<Block> = client
        .get(format!("{base}/blocks/since/{our_height}"))
        .send()
        .await
        .map_err(|e| e.to_string())?
        .json()
        .await
        .map_err(|e| e.to_string())?;
    let mut applied = finalized_applied;
    {
        // Only block application is catch-up work; the polls above are routine.
        let _syncing = SyncingGuard::enter(&shared.syncing);
        for block in blocks {
            let _guard = shared.chain_lock.lock().await;
            let current = shared.head().map(|b| b.header.index).unwrap_or(0);
            if block.header.index != current + 1 {
                break;
            }
            admit_block(shared, &block)?;
            let producer = block.header.validator.clone();
            if !commit_block(shared, &block, &producer) {
                return Err(format!(
                    "block #{} did not advance head (state root or model set mismatch?)",
                    block.header.index
                ));
            }
            applied += 1;
        }
    }
    Ok(SyncReport {
        applied,
        peer_height: Some(peer_height),
    })
}

/// `verinoo-node address --seed <string>` — the validator address a node with
/// that `VNN_KEY_SEED` will sign with. Reads `VNN_KEY_SEED` when `--seed` is
/// omitted, so a running node's shell environment prints the same value.
fn address_cli(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let mut raw = std::env::var("VNN_KEY_SEED").unwrap_or_default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" if i + 1 < args.len() => {
                raw = args[i + 1].clone();
                i += 1;
            }
            "--help" | "-h" => {
                println!("Usage: verinoo-node address --seed <string>");
                return Ok(());
            }
            other => return Err(format!("unknown argument `{other}`").into()),
        }
        i += 1;
    }
    if raw.trim().is_empty() {
        return Err("--seed (or VNN_KEY_SEED) is required".into());
    }
    use sha2::Digest as _;
    let mut h = sha2::Sha256::new();
    h.update(raw.as_bytes());
    let seed: [u8; 32] = h.finalize().into();
    let signing = ed25519_dalek::SigningKey::from_bytes(&seed);
    let address = verinoo_exec::engine::public_key_to_address(&hex::encode(
        signing.verifying_key().to_bytes(),
    ))?;
    println!("{address}");
    Ok(())
}

/// `verinoo-node infer --key-file key.json --input "where is my package"`
async fn infer_cli(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    use base64::Engine as _;

    let mut key_file = String::new();
    let mut input = String::new();
    let mut model = verinoo_model::MODEL_ID.to_string();
    let mut max_fee: u64 = 100;
    let mut rpc_url = "http://localhost:7100".to_string();
    let mut api_key: Option<String> = None;
    let mut plaintext = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--plaintext" => plaintext = true,
            "--key-file" if i + 1 < args.len() => {
                key_file = args[i + 1].clone();
                i += 1;
            }
            "--input" if i + 1 < args.len() => {
                input = args[i + 1].clone();
                i += 1;
            }
            "--model" if i + 1 < args.len() => {
                model = args[i + 1].clone();
                i += 1;
            }
            "--fee" if i + 1 < args.len() => {
                max_fee = args[i + 1].parse().unwrap_or(100);
                i += 1;
            }
            "--node" if i + 1 < args.len() => {
                rpc_url = args[i + 1].clone();
                i += 1;
            }
            "--api-key" if i + 1 < args.len() => {
                api_key = Some(args[i + 1].clone());
                i += 1;
            }
            _ => {}
        }
        i += 1;
    }

    if key_file.is_empty() || input.is_empty() {
        eprintln!("Usage: verinoo-node infer --key-file <file> --input \"<text>\" [--model id] [--fee n] [--node url] [--api-key key] [--plaintext]");
        std::process::exit(1);
    }

    let (_, secret_b64, address) = wallet::load_key(&key_file)?;
    let acct: serde_json::Value = reqwest::get(format!("{rpc_url}/account/{address}"))
        .await?
        .json()
        .await?;
    let nonce = acct["nonce"].as_u64().unwrap_or(0) + 1;

    let sk_bytes = base64::engine::general_purpose::STANDARD.decode(&secret_b64)?;
    let sk_arr: [u8; 32] = sk_bytes[..32].try_into().map_err(|_| "bad key length")?;
    let signing = ed25519_dalek::SigningKey::from_bytes(&sk_arr);

    let deadline = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs()
        + 3_600;

    // Encrypt by default when the node advertises a network input key. The
    // plaintext never enters the transaction, block or task state.
    let network_key: serde_json::Value = match reqwest::get(format!("{rpc_url}/v1/input-key")).await
    {
        Ok(resp) => resp.json().await.unwrap_or(serde_json::Value::Null),
        Err(_) => serde_json::Value::Null,
    };
    let public_key = network_key["publicKey"].as_str().unwrap_or("");
    let body = if !plaintext && !public_key.is_empty() {
        let commitment = verinoo_crypto::sha256_hex(&input);
        let encrypted = verinoo_crypto::encrypt(public_key, &input, &commitment)?;
        wallet::build_signed_inference_encrypted(
            &signing,
            &address,
            &model,
            &encrypted,
            &commitment,
            max_fee,
            deadline,
            nonce,
        )?
    } else {
        wallet::build_signed_inference(
            &signing, &address, &model, &input, max_fee, deadline, nonce,
        )?
    };

    let mut req = reqwest::Client::new()
        .post(format!("{rpc_url}/v1/infer"))
        .json(&body);
    if let Some(key) = &api_key {
        req = req.header("authorization", format!("Bearer {key}"));
    }
    let resp = req.send().await?;
    let status = resp.status();
    let text = resp.text().await?;
    if !status.is_success() {
        return Err(format!("HTTP {status}: {text}").into());
    }
    let v: serde_json::Value = serde_json::from_str(&text)?;
    let task_id = v["taskId"].as_str().unwrap_or("-").to_string();
    println!("task: {task_id}");
    println!("tx:   {}", v["txId"].as_str().unwrap_or("-"));
    println!("poll: curl -s {rpc_url}/v1/tasks/{task_id}");
    Ok(())
}

fn sha256_hex_str(input: &str) -> String {
    use sha2::Digest as _;
    let mut h = sha2::Sha256::new();
    h.update(input.as_bytes());
    hex::encode(h.finalize())
}

// ============================ wallet CLI ============================

async fn wallet_cli(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.is_empty() {
        println!("Usage: verinoo-node wallet <command> [options]");
        println!("  generate --output <file>");
        println!("  balance  --key-file <file> [--node <url>]");
        println!(
            "  send     --key-file <file> --to <addr> --amount <n> [--node <url>] [--api-key key]"
        );
        println!("\nInference:\n  infer    --key-file <file> --input \"<text>\" [--model id] [--fee n] [--node <url>] [--api-key key]");
        return Ok(());
    }

    let cmd = args[0].as_str();
    let mut output = "wallet.json".to_string();
    let mut key_file = String::new();
    let mut to_addr = String::new();
    let mut amount: u64 = 0;
    let mut rpc_url = "http://localhost:7100".to_string();
    let mut api_key: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--output" | "--key-file" => {
                if i + 1 < args.len() {
                    key_file = args[i + 1].clone();
                    if cmd == "generate" {
                        output = key_file.clone();
                    }
                    i += 1;
                }
            }
            "--to" => {
                if i + 1 < args.len() {
                    to_addr = args[i + 1].clone();
                    i += 1;
                }
            }
            "--amount" => {
                if i + 1 < args.len() {
                    amount = args[i + 1].parse().unwrap_or(0);
                    i += 1;
                }
            }
            "--node" if i + 1 < args.len() => {
                rpc_url = args[i + 1].clone();
                i += 1;
            }
            "--api-key" if i + 1 < args.len() => {
                api_key = Some(args[i + 1].clone());
                i += 1;
            }
            _ => {}
        }
        i += 1;
    }

    match cmd {
        "generate" => {
            wallet::generate_key(&output)?;
        }
        "balance" => {
            let (_, _, addr) = wallet::load_key(&key_file)?;
            let bal = wallet::query_balance(&rpc_url, &addr).await?;
            println!("Address: {addr}");
            println!("Balance: {bal}");
        }
        "send" => {
            if to_addr.is_empty() || amount == 0 {
                eprintln!("Error: --to and --amount are required for send");
                std::process::exit(1);
            }
            let (_, _, from_addr) = wallet::load_key(&key_file)?;
            let (_, secret_b64, _) = wallet::load_key(&key_file)?;

            // Query current nonce
            let acct_url = format!("{rpc_url}/account/{from_addr}");
            let body: serde_json::Value = reqwest::get(&acct_url)
                .await
                .map_err(|e| e.to_string())?
                .json()
                .await
                .map_err(|e| e.to_string())?;
            let nonce = body["nonce"].as_u64().unwrap_or(0) + 1;

            let result = wallet::send_transfer(
                &rpc_url,
                &from_addr,
                &secret_b64,
                &to_addr,
                amount,
                nonce,
                api_key.as_deref(),
            )
            .await?;
            println!("✓ Transfer submitted: {result}");
        }
        _ => eprintln!("Unknown wallet command: {cmd}"),
    }
    Ok(())
}

/// Unique scratch directory for bft-v2 proposal checkpoints in tests.
#[cfg(test)]
pub(crate) fn test_proposal_work_dir() -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    std::env::temp_dir().join(format!("verinoo-propose-test-{}-{n}", std::process::id()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tx(id: &str) -> verinoo_types::tx::TxEnvelope {
        verinoo_types::tx::TxEnvelope {
            tx_id: id.into(),
            from: "ai_0123456789abcdef0123456789abcdef01234567".into(),
            to: String::new(),
            value: 0,
            nonce: 1,
            data: verinoo_types::tx::TxData {
                r#type: verinoo_types::tx::TxKind::Transfer,
                data: serde_json::json!({}),
            },
            gas_limit: None,
            gas_price: None,
            public_key: String::new(),
            signature: String::new(),
        }
    }

    fn gas_limit_skip(id: &str) -> (String, String) {
        (
            id.into(),
            verinoo_exec::engine::GAS_LIMIT_SKIP_REASON.into(),
        )
    }

    #[test]
    fn requeue_gas_skipped_puts_only_gas_skips_at_the_front() {
        let a = tx("a");
        let b = tx("b");
        let c = tx("c");
        let new_arrival = tx("new");
        let batch = vec![a, b.clone(), c.clone()];
        let skipped = vec![gas_limit_skip("b"), ("c".into(), "bad nonce".into())];

        let mut mempool = vec![new_arrival];
        requeue_gas_skipped(&mut mempool, &batch, &skipped);

        let ids: Vec<_> = mempool.iter().map(|t| t.tx_id.as_str()).collect();
        assert_eq!(ids, vec!["b", "new"]);
    }

    #[test]
    fn requeue_gas_skipped_preserves_order_and_dedupes_gossip() {
        let a = tx("a");
        let b = tx("b");
        let c = tx("c");
        let batch = vec![a, b, c.clone()];
        let skipped = vec![gas_limit_skip("b"), gas_limit_skip("c")];

        let mut mempool = vec![c, tx("z")];
        requeue_gas_skipped(&mut mempool, &batch, &skipped);

        let ids: Vec<_> = mempool.iter().map(|t| t.tx_id.as_str()).collect();
        assert_eq!(ids, vec!["b", "c", "z"]);
    }

    fn test_validators(count: u32) -> Vec<verinoo_exec::engine::ProducerKey> {
        let mut keys: Vec<_> = (0..count)
            .map(|index| verinoo_exec::engine::ProducerKey::deterministic("bft-catchup", index))
            .collect();
        keys.sort_by(|a, b| a.address.cmp(&b.address));
        keys
    }

    fn test_genesis() -> Block {
        let mut header = verinoo_exec::engine::BlockHeader {
            hash: String::new(),
            version: 1,
            index: 0,
            timestamp: 0,
            previous_hash: "0".repeat(64),
            validator: "genesis".into(),
            validator_pub_key: String::new(),
            validator_signature: String::new(),
            state_root: sha256_hex_str("genesis"),
            tx_root: sha256_hex_str("empty"),
            inference_tasks_root: sha256_hex_str("empty-tasks"),
            compute_root: sha256_hex_str("empty-compute"),
        };
        header.hash = verinoo_exec::engine::hash_header(&header);
        Block {
            header,
            transactions: vec![],
        }
    }

    fn test_store_with_genesis() -> verinoo_state::StateStore {
        let store = verinoo_state::StateStore::open_temp().unwrap();
        let genesis = test_genesis();
        store.put_block_by_hash(
            &genesis.header.hash,
            &serde_json::to_value(&genesis).unwrap(),
        );
        verinoo_exec::engine::set_head(&store, &genesis);
        store
    }

    fn test_bft_shared(
        store: verinoo_state::StateStore,
        keys: &[verinoo_exec::engine::ProducerKey],
        finalized_height: u64,
        finalized_hash: &str,
    ) -> Arc<Shared> {
        test_bft_shared_mode(store, keys, 0, finalized_height, finalized_hash, "bft")
    }

    #[allow(clippy::too_many_arguments)]
    fn test_bft_shared_mode(
        store: verinoo_state::StateStore,
        keys: &[verinoo_exec::engine::ProducerKey],
        local_index: usize,
        finalized_height: u64,
        finalized_hash: &str,
        consensus_mode: &str,
    ) -> Arc<Shared> {
        let params = GenesisParams {
            validators: keys.iter().map(|key| key.address.clone()).collect(),
            ..GenesisParams::default()
        };
        let manager = finality::FinalityManager::new(
            &params.chain_id,
            params.validators.clone(),
            &keys[local_index].address,
            keys[local_index].signing.clone(),
            finalized_height,
            finalized_hash,
            1_000,
        )
        .unwrap();
        let faucet = verinoo_exec::engine::ProducerKey::deterministic("faucet-test", 0);
        Arc::new(Shared {
            store,
            params,
            mempool: tokio::sync::Mutex::new(Vec::new()),
            faucet,
            chain_lock: tokio::sync::Mutex::new(()),
            syncing: AtomicBool::new(false),
            best_peer_height: AtomicU64::new(0),
            block_time_ms: 1_000,
            peer_lag_blocks: 2,
            request_policy: verinoo_policy::Policy::default(),
            task_streams: Arc::new(tokio::sync::Semaphore::new(MAX_ACTIVE_TASK_STREAMS)),
            snapshot_dir: None,
            finality: Some(tokio::sync::Mutex::new(manager)),
            consensus_mode: consensus_mode.into(),
            proposal_work_dir: test_proposal_work_dir(),
            policy: std::sync::Mutex::new(Policy::default()),
        })
    }

    fn test_bft_v2_shared(
        store: verinoo_state::StateStore,
        keys: &[verinoo_exec::engine::ProducerKey],
        local_index: usize,
    ) -> Arc<Shared> {
        test_bft_shared_mode(store, keys, local_index, 0, "", "bft-v2")
    }

    fn signed_transfer(
        signer: &ed25519_dalek::SigningKey,
        from: &str,
        to: &str,
        value: u64,
        nonce: u64,
    ) -> verinoo_types::tx::TxEnvelope {
        use ed25519_dalek::Signer as _;
        use verinoo_types::canonical::UnsignedPayload;
        let data = verinoo_types::tx::TxData {
            r#type: verinoo_types::tx::TxKind::Transfer,
            data: serde_json::json!({}),
        };
        let payload = UnsignedPayload {
            data: serde_json::to_value(&data).unwrap(),
            from: from.into(),
            nonce,
            to: to.into(),
            value,
            gas_limit: None,
            gas_price: None,
        };
        let tx_id = payload.tx_id().unwrap();
        let signature = hex::encode(signer.sign(tx_id.as_bytes()).to_bytes());
        verinoo_types::tx::TxEnvelope {
            tx_id,
            from: from.into(),
            to: to.into(),
            value,
            nonce,
            data,
            gas_limit: None,
            gas_price: None,
            public_key: hex::encode(signer.verifying_key().to_bytes()),
            signature,
        }
    }

    fn test_certificate(
        chain_id: &str,
        block: &Block,
        keys: &[verinoo_exec::engine::ProducerKey],
    ) -> verinoo_consensus::CommitCertificate {
        let votes = keys
            .iter()
            .map(|key| {
                verinoo_consensus::Vote::sign(
                    chain_id,
                    verinoo_consensus::VoteKind::Precommit,
                    block.header.index,
                    0,
                    &block.header.hash,
                    &key.address,
                    &key.signing,
                )
            })
            .collect();
        verinoo_consensus::CommitCertificate {
            chain_id: chain_id.into(),
            height: block.header.index,
            round: 0,
            block_hash: block.header.hash.clone(),
            votes,
        }
    }

    fn produce_finalized_pairs(
        store: &verinoo_state::StateStore,
        chain_id: &str,
        keys: &[verinoo_exec::engine::ProducerKey],
        count: u64,
    ) -> Vec<FinalizedPair> {
        let mut pairs = Vec::new();
        for height in 1..=count {
            let producer = &keys[((height - 1) as usize) % keys.len()];
            let block =
                verinoo_exec::engine::produce_block(store, &[], producer, 1_000 + height).unwrap();
            verinoo_exec::engine::set_head(store, &block);
            let certificate = test_certificate(chain_id, &block, keys);
            pairs.push(FinalizedPair { block, certificate });
        }
        pairs
    }

    async fn spawn_peer(
        chain_id: String,
        pairs: Vec<FinalizedPair>,
        with_finalized_endpoint: bool,
    ) -> String {
        use axum::routing::get;
        use axum::{Json, Router};

        let height = pairs
            .last()
            .map(|pair| pair.block.header.index)
            .unwrap_or(0);
        let status = serde_json::json!({
            "chainId": chain_id,
            "height": height,
            "finalizedHeight": height,
        });
        let blocks: Vec<Block> = pairs.iter().map(|pair| pair.block.clone()).collect();
        let pairs = Arc::new(pairs);
        let blocks = Arc::new(blocks);

        let mut app =
            Router::new().route("/status", get(move || async move { Json(status.clone()) }));
        if with_finalized_endpoint {
            let pairs = pairs.clone();
            app = app.route(
                "/finalized/since/:index",
                get(
                    move |axum::extract::Path(index): axum::extract::Path<u64>| async move {
                        let out: Vec<serde_json::Value> = pairs
                            .iter()
                            .filter(|pair| pair.block.header.index > index)
                            .take(200)
                            .map(|pair| {
                                serde_json::json!({
                                    "block": pair.block,
                                    "certificate": pair.certificate,
                                })
                            })
                            .collect();
                        Json(serde_json::json!(out))
                    },
                ),
            );
        }
        let blocks_for_route = blocks.clone();
        app = app.route(
            "/blocks/since/:index",
            get(
                move |axum::extract::Path(index): axum::extract::Path<u64>| async move {
                    let out: Vec<Block> = blocks_for_route
                        .iter()
                        .filter(|block| block.header.index > index)
                        .take(200)
                        .cloned()
                        .collect();
                    Json(serde_json::json!(out))
                },
            ),
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn sync_once_adopts_certified_history_from_an_empty_store() {
        let keys = test_validators(4);
        let chain_id = GenesisParams::default().chain_id;
        let peer_store = test_store_with_genesis();
        let pairs = produce_finalized_pairs(&peer_store, &chain_id, &keys, 3);
        let peer_head = verinoo_exec::engine::current_head(&peer_store).unwrap();
        let shared = test_bft_shared(test_store_with_genesis(), &keys, 0, "");

        let url = spawn_peer(chain_id, pairs, true).await;
        let client = reqwest::Client::new();
        let report = sync_once(&shared, &client, &url).await.unwrap();

        assert_eq!(report.applied, 3);
        assert_eq!(report.peer_height, Some(3));
        let head = shared.head().unwrap();
        assert_eq!(head.header.index, peer_head.header.index);
        assert_eq!(head.header.hash, peer_head.header.hash);
        let (finalized_height, finalized_hash) = current_finalized(&shared).await;
        assert_eq!(finalized_height, 3);
        assert_eq!(finalized_hash, peer_head.header.hash);
        for height in 1..=3 {
            assert!(shared.store.get_meta(&format!("commit:{height}")).is_some());
        }
        assert!(shared
            .store
            .get_meta("consensusOwnVote:1:0:prevote")
            .is_none());
    }

    #[tokio::test]
    async fn finalized_catch_up_rejects_tampered_certificates() {
        let keys = test_validators(4);
        let chain_id = GenesisParams::default().chain_id;
        let peer_store = test_store_with_genesis();
        let pairs = produce_finalized_pairs(&peer_store, &chain_id, &keys, 2);

        let mut bad_signature = pairs.clone();
        bad_signature[0].certificate.votes[0].signature = "00".repeat(64);
        let shared = test_bft_shared(test_store_with_genesis(), &keys, 0, "");
        assert!(apply_finalized_pairs(&shared, bad_signature).await.is_err());
        assert_eq!(shared.head().unwrap().header.index, 0);
        assert_eq!(current_finalized(&shared).await.0, 0);

        let mut bad_binding = pairs.clone();
        bad_binding[0].certificate.block_hash = "ff".repeat(32);
        let shared = test_bft_shared(test_store_with_genesis(), &keys, 0, "");
        assert!(apply_finalized_pairs(&shared, bad_binding).await.is_err());
        assert_eq!(current_finalized(&shared).await.0, 0);

        let mut no_quorum = pairs;
        no_quorum[0].certificate.votes.truncate(2);
        let shared = test_bft_shared(test_store_with_genesis(), &keys, 0, "");
        assert!(apply_finalized_pairs(&shared, no_quorum).await.is_err());
        assert_eq!(current_finalized(&shared).await.0, 0);
    }

    #[tokio::test]
    async fn sync_once_falls_back_when_finalized_endpoint_is_missing() {
        let keys = test_validators(4);
        let chain_id = GenesisParams::default().chain_id;
        let peer_store = test_store_with_genesis();
        let pairs = produce_finalized_pairs(&peer_store, &chain_id, &keys, 3);
        let peer_head = verinoo_exec::engine::current_head(&peer_store).unwrap();
        let shared = test_bft_shared(test_store_with_genesis(), &keys, 0, "");

        let url = spawn_peer(chain_id, pairs, false).await;
        let client = reqwest::Client::new();
        let report = sync_once(&shared, &client, &url).await.unwrap();

        assert_eq!(report.applied, 3);
        let head = shared.head().unwrap();
        assert_eq!(head.header.index, peer_head.header.index);
        assert_eq!(head.header.hash, peer_head.header.hash);
        let (finalized_height, finalized_hash) = current_finalized(&shared).await;
        assert_eq!(finalized_height, 0);
        assert!(finalized_hash.is_empty());
    }

    #[tokio::test]
    async fn bft_v2_single_validator_finalizes_and_commits_only_with_a_certificate() {
        let keys = test_validators(1);
        let shared = test_bft_v2_shared(test_store_with_genesis(), &keys, 0);

        // Fund a sender so the proposal carries a real transaction.
        let sender = verinoo_exec::engine::ProducerKey::deterministic("bft-v2-sender", 0);
        let mut acc = shared.store.account_or_default(&sender.address);
        acc.balance = 1_000;
        shared.store.set_account(&acc);
        let tx = signed_transfer(&sender.signing, &sender.address, "ai_recipient", 25, 1);

        // Leader path: build the block on a checkpoint clone.
        let report = propose_block_with_report(
            &shared.store,
            std::slice::from_ref(&tx),
            &keys[0],
            1_000,
            &shared.proposal_work_dir,
        )
        .unwrap();
        let block = report.block;
        assert_eq!(block.header.index, 1);
        assert_eq!(block.transactions.len(), 1);
        // The clone was cleaned up.
        assert_eq!(
            std::fs::read_dir(&shared.proposal_work_dir)
                .map(|entries| entries.count())
                .unwrap_or(1),
            0
        );

        // Proposing did not touch canonical state.
        assert_eq!(shared.head().unwrap().header.index, 0);
        assert_eq!(
            shared.store.get_account(&sender.address).unwrap().balance,
            1_000
        );

        // The leader stores the proposal by hash and feeds the finality
        // manager; a single validator reaches quorum on its own.
        shared
            .store
            .put_block_by_hash(&block.header.hash, &serde_json::to_value(&block).unwrap());
        let output = {
            let mut manager = shared.finality.as_ref().unwrap().lock().await;
            manager.observe_block(block.header.index, &block.header.hash)
        };
        let (height, hash, certificate) = output
            .finalized
            .expect("single validator reaches quorum alone");
        assert_eq!(height, 1);
        assert_eq!(hash, block.header.hash);

        // The certificate exists, but the head has not moved yet.
        assert_eq!(shared.head().unwrap().header.index, 0);

        shared_finality_commit(&shared, height, &hash, &certificate)
            .expect("certificate commits the stored proposal");
        let head = shared.head().unwrap();
        assert_eq!(head.header.index, 1);
        assert_eq!(head.header.hash, block.header.hash);
        assert_eq!(
            shared.store.get_account("ai_recipient").unwrap().balance,
            25
        );
        assert!(shared.store.get_account(&sender.address).unwrap().balance < 1_000);
    }

    #[tokio::test]
    async fn bft_v2_receiver_does_not_advance_the_head_before_a_certificate() {
        let keys = test_validators(2); // keys[0] = leader, keys[1] = receiver
        let chain_id = GenesisParams::default().chain_id;

        // The leader runs its own store and publishes a proposal at height 1.
        let leader_store = test_store_with_genesis();
        let block =
            verinoo_exec::engine::produce_block(&leader_store, &[], &keys[0], 1_000).unwrap();
        verinoo_exec::engine::set_head(&leader_store, &block);

        // The receiver starts at genesis; it is a validator but not the leader.
        let shared = test_bft_v2_shared(test_store_with_genesis(), &keys, 1);

        admit_block(&shared, &block).expect("proposal extends the receiver head");
        shared
            .store
            .put_block_by_hash(&block.header.hash, &serde_json::to_value(&block).unwrap());
        assert_eq!(shared.head().unwrap().header.index, 0);

        // Feeding the manager emits the receiver's prevote but cannot finalize
        // a two-validator set alone, and it must not move the head either.
        let output = {
            let mut manager = shared.finality.as_ref().unwrap().lock().await;
            manager.observe_block(block.header.index, &block.header.hash)
        };
        assert!(output.finalized.is_none());
        assert!(output
            .votes
            .iter()
            .any(|vote| vote.validator == keys[1].address));
        assert_eq!(shared.head().unwrap().header.index, 0);

        // Once a 2f+1 certificate over the proposal exists, the receiver
        // commits and only then advances the head.
        let certificate = test_certificate(&chain_id, &block, &keys);
        shared_finality_commit(&shared, 1, &block.header.hash, &certificate)
            .expect("certificate commits the stored proposal");
        let head = shared.head().unwrap();
        assert_eq!(head.header.index, 1);
        assert_eq!(head.header.hash, block.header.hash);
    }

    #[tokio::test]
    async fn bft_v2_certificates_for_unknown_or_invalid_blocks_do_not_finalize() {
        let keys = test_validators(4);
        let chain_id = GenesisParams::default().chain_id;
        let shared = test_bft_v2_shared(test_store_with_genesis(), &keys, 0);

        // Certificate for a block this node never admitted.
        let unknown = "ff".repeat(32);
        let certificate = {
            let votes = keys
                .iter()
                .map(|key| {
                    verinoo_consensus::Vote::sign(
                        &chain_id,
                        verinoo_consensus::VoteKind::Precommit,
                        1,
                        0,
                        &unknown,
                        &key.address,
                        &key.signing,
                    )
                })
                .collect();
            verinoo_consensus::CommitCertificate {
                chain_id: chain_id.clone(),
                height: 1,
                round: 0,
                block_hash: unknown.clone(),
                votes,
            }
        };
        assert!(shared_finality_commit(&shared, 1, &unknown, &certificate).is_err());
        assert_eq!(shared.head().unwrap().header.index, 0);

        // Certificate for a stored block whose state root does not match what
        // applying derives. The header is re-signed so every structural check
        // passes and only the execution check can reject it.
        let leader_store = test_store_with_genesis();
        let mut block =
            verinoo_exec::engine::produce_block(&leader_store, &[], &keys[0], 1_000).unwrap();
        block.header.state_root = "00".repeat(32);
        block.header.hash = verinoo_exec::engine::hash_header(&block.header);
        let signature = {
            use ed25519_dalek::Signer as _;
            hex::encode(
                keys[0]
                    .signing
                    .sign(block.header.hash.as_bytes())
                    .to_bytes(),
            )
        };
        block.header.validator_signature = signature;
        admit_block(&shared, &block).expect("re-signed header passes structural admission");
        shared
            .store
            .put_block_by_hash(&block.header.hash, &serde_json::to_value(&block).unwrap());
        let certificate = test_certificate(&chain_id, &block, &keys);
        assert!(shared_finality_commit(&shared, 1, &block.header.hash, &certificate).is_err());
        assert_eq!(shared.head().unwrap().header.index, 0);
    }
}
