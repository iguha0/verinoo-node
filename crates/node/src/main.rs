//! Verinoo testnet node — wires state + execution + networking + RPC.
//!
//! Hostile-network rule: NOTHING from the wire touches state without
//! passing `validate_block` (header hash, producer signature +
//! authorization, chain linkage, timestamp sanity, txRoot, duplicate
//! rejection, per-tx canonical + signature checks).

mod rpc;
mod wallet;

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
    /// Testnet faucet identity. Fixed and public by design: every node must
    /// derive the same funded genesis account, otherwise faucet drips would
    /// only exist on the node that served them.
    pub faucet: verinoo_exec::engine::ProducerKey,
    policy: std::sync::Mutex<Policy>,
}

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
    use verinoo_exec::leader::LeaderPolicy as _;
    // `interval` fires immediately; consume that tick so peers have one block
    // time to connect and exchange gossip subscriptions first. Producing at
    // t=0 publishes into a mesh that does not exist yet and the block is lost.
    ticker.tick().await;

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

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

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

    let shared = Arc::new(Shared {
        store,
        params: cfg.genesis.clone(),
        mempool: tokio::sync::Mutex::new(Vec::new()),
        faucet,
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

    // RPC server.
    tokio::spawn(rpc::serve(shared.clone(), net.clone(), cfg.api_port));

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

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
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
            _ => {}
        }
        i += 1;
    }

    if key_file.is_empty() || input.is_empty() {
        eprintln!("Usage: verinoo-node infer --key-file <file> --input \"<text>\" [--model id] [--fee n] [--node url]");
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
    let body = wallet::build_signed_inference(
        &signing, &address, &model, &input, max_fee, deadline, nonce,
    )?;

    let resp = reqwest::Client::new()
        .post(format!("{rpc_url}/v1/infer"))
        .json(&body)
        .send()
        .await?;
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

fn shared_params_block_time(shared: &Shared) -> u64 {
    shared.params.block_time_ms
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
        println!("  send     --key-file <file> --to <addr> --amount <n> [--node <url>]");
        println!("\nInference:\n  infer    --key-file <file> --input \"<text>\" [--model id] [--fee n] [--node <url>]");
        return Ok(());
    }

    let cmd = args[0].as_str();
    let mut output = "wallet.json".to_string();
    let mut key_file = String::new();
    let mut to_addr = String::new();
    let mut amount: u64 = 0;
    let mut rpc_url = "http://localhost:7100".to_string();

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

            let result =
                wallet::send_transfer(&rpc_url, &from_addr, &secret_b64, &to_addr, amount, nonce)
                    .await?;
            println!("✓ Transfer submitted: {result}");
        }
        _ => eprintln!("Unknown wallet command: {cmd}"),
    }
    Ok(())
}
