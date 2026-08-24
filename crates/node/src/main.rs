//! Verinoo testnet node — wires state + execution + networking + RPC.
//!
//! Hostile-network rule: NOTHING from the wire touches state without
//! passing `validate_block` (header hash, producer signature +
//! authorization, chain linkage, timestamp sanity, txRoot, duplicate
//! rejection, per-tx canonical + signature checks).

mod rpc;

use std::path::PathBuf;
use std::sync::Arc;

use verinoo_exec::engine::{current_head, produce_block, set_head, validate_block, Block};
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
    policy: std::sync::Mutex<Policy>,
}

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
fn commit_block(shared: &Shared, block: &Block, producer_address: &str) {
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
        return;
    }
    set_head(&shared.store, block);
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

    loop {
        ticker.tick().await;
        if shared.head().is_none() {
            continue;
        }
        let batch: Vec<_> = shared.mempool.lock().await.drain(..).collect();

        match produce_block(&shared.store, &batch, &producer, shared.now_ms()) {
            Ok(block) => {
                let n = block.transactions.len();
                set_head(&shared.store, &block);
                tracing::info!(
                    "produced block #{} | {} txs | {}",
                    block.header.index,
                    n,
                    &block.header.hash[..16.min(block.header.hash.len())]
                );
                net.publish_block(&serde_json::to_string(&block).expect("block serialize"));
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
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    // Config via env (deploy-friendly): VNN_* variables.
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
        key_seed: {
            use sha2::Digest as _;
            let s = std::env::var("VNN_KEY_SEED").unwrap_or_else(|_| "verinoo".into());
            let mut h = sha2::Sha256::new();
            h.update(s.as_bytes());
            h.finalize().into()
        },
        seed_peers: std::env::var("VNN_SEEDS")
            .map(|v| {
                v.split(',')
                    .filter(|s| !s.is_empty())
                    .map(String::from)
                    .collect()
            })
            .unwrap_or_default(),
        validator: std::env::var("VNN_VALIDATOR")
            .map(|v| v == "1" || v == "true")
            .unwrap_or(false),
        genesis: GenesisParams::default(),
    };

    let store = StateStore::open(&cfg.db_path)?;
    let shared = Arc::new(Shared {
        store,
        params: cfg.genesis.clone(),
        mempool: tokio::sync::Mutex::new(Vec::new()),
        policy: std::sync::Mutex::new(Policy::default()),
    });

    let signing = ed25519_dalek::SigningKey::from_bytes(&cfg.key_seed);
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

    // RPC server.
    tokio::spawn(rpc::serve(shared.clone(), cfg.api_port));

    // Event pump consumes the wire stream.
    let pump_shared = shared.clone();
    tokio::spawn(async move {
        let shared = pump_shared;
        while let Some(ev) = event_rx.recv().await {
            match ev {
                NetworkEvent::Tx(payload) => {
                    if !shared.mark_seen(&payload) {
                        continue;
                    }
                    match serde_json::from_str::<verinoo_types::tx::TxEnvelope>(&payload) {
                        Ok(tx) => {
                            let mut mp = shared.mempool.lock().await;
                            if !mp.iter().any(|t| t.tx_id == tx.tx_id) && mp.len() < 5_000 {
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
                        Ok(block) => match admit_block(&shared, &block) {
                            Ok(()) => {
                                let producer = block.header.validator.clone();
                                tracing::info!(
                                    "admitted block #{} from {}",
                                    block.header.index,
                                    &producer[..12.min(producer.len())]
                                );
                                commit_block(&shared, &block, &producer);
                            }
                            Err(reason) => {
                                tracing::warn!("rejected block: {reason}");
                            }
                        },
                        Err(e) => tracing::debug!("bad block payload: {e}"),
                    }
                }
                NetworkEvent::PeerBanned(peer, reason) => {
                    tracing::warn!("peer banned: {peer} — {reason}");
                }
            }
        }
    });

    // Validator loop (optional).
    if cfg.validator {
        let v_shared = shared.clone();
        let v_net = net.clone();
        tokio::spawn(validator_loop(
            v_shared,
            v_net,
            signing,
            shared_params_block_time(&shared),
        ));
    }

    // Idle forever; tasks do the work.
    std::future::pending::<()>().await;
    Ok(())
}

fn shared_params_block_time(shared: &Shared) -> u64 {
    shared.params.block_time_ms
}

fn sha256_hex_str(input: &str) -> String {
    use sha2::Digest as _;
    let mut h = sha2::Sha256::new();
    h.update(input.as_bytes());
    hex::encode(h.finalize())
}
