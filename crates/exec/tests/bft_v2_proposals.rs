//! bft-v2 Phase A: proposal execution on a checkpoint clone and
//! certificate-gated commit.
//!
//! A `bft-v2` leader must be able to build the next block without touching
//! canonical state, and the head may only advance when a (node-verified)
//! commit certificate has called `commit_finalized`. These tests pin both
//! halves against real RocksDB stores.

use ed25519_dalek::Signer;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use verinoo_exec::engine::{
    commit_finalized, compute_state_root, current_head, produce_block, produce_block_with_report,
    propose_block, propose_block_with_report, set_head, Block, ProducerKey, GAS_LIMIT_SKIP_REASON,
};
use verinoo_state::genesis::INITIAL_BASE_FEE;
use verinoo_state::StateStore;
use verinoo_types::canonical::UnsignedPayload;
use verinoo_types::tx::{TxData, TxEnvelope, TxKind};

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!(
            "verinoo-exec-bft-v2-{tag}-{}-{n}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn keypair(name: &str) -> ed25519_dalek::SigningKey {
    use sha2::Digest;
    let mut h = sha2::Sha256::new();
    h.update(format!("key:{name}").as_bytes());
    let digest: [u8; 32] = h.finalize().into();
    ed25519_dalek::SigningKey::from_bytes(&digest)
}

fn make_tx(
    signer: &ed25519_dalek::SigningKey,
    from_addr: &str,
    to: &str,
    value: u64,
    nonce: u64,
) -> TxEnvelope {
    let data = TxData {
        r#type: TxKind::Transfer,
        data: serde_json::json!({}),
    };
    let payload = UnsignedPayload {
        data: serde_json::to_value(&data).unwrap(),
        from: from_addr.into(),
        nonce,
        to: to.into(),
        value,
        gas_limit: None,
        gas_price: None,
    };
    let tx_id = payload.tx_id().unwrap();
    let sig = signer.sign(tx_id.as_bytes());
    TxEnvelope {
        tx_id,
        from: from_addr.into(),
        to: to.into(),
        value,
        nonce,
        data,
        gas_limit: None,
        gas_price: None,
        public_key: hex::encode(signer.verifying_key().to_bytes()),
        signature: hex::encode(sig.to_bytes()),
    }
}

struct Chain {
    store: StateStore,
    producer: ProducerKey,
    now_ms: u64,
}

fn address_of(signer: &ed25519_dalek::SigningKey) -> String {
    verinoo_exec::engine::public_key_to_address(&hex::encode(signer.verifying_key().to_bytes()))
        .unwrap()
}

impl Chain {
    fn fresh() -> Self {
        let store = StateStore::open_temp().unwrap();
        let producer = ProducerKey::deterministic("validator-a", 1);
        let genesis = Block {
            header: verinoo_exec::engine::BlockHeader {
                hash: "0".repeat(64),
                version: 1,
                index: 0,
                timestamp: 0,
                previous_hash: "f".repeat(64),
                validator: "genesis".into(),
                validator_pub_key: String::new(),
                validator_signature: String::new(),
                state_root: String::new(),
                tx_root: String::new(),
                inference_tasks_root: String::new(),
                compute_root: String::new(),
            },
            transactions: vec![],
        };
        set_head(&store, &genesis);
        Self {
            store,
            producer,
            now_ms: 1_000_000,
        }
    }

    fn fund(&self, address: &str, amount: i64) {
        let mut acc = self.store.account_or_default(address);
        acc.balance = amount;
        self.store.set_account(&acc);
    }
}

#[test]
fn propose_block_matches_production_and_leaves_the_store_untouched() {
    let chain = Chain::fresh();
    let sender = keypair("propose-sender");
    let sender_addr = address_of(&sender);
    chain.fund(&sender_addr, 10_000);
    let txs = vec![make_tx(&sender, &sender_addr, "ai_recipient", 25, 1)];

    let before_root = compute_state_root(&chain.store);
    let before_head = current_head(&chain.store).unwrap();
    let root = TempDir::new("propose-clean");
    // The work directory is created on demand.
    let work_dir = root.path().join("nested").join("propose");
    assert!(!work_dir.exists());

    let proposed =
        propose_block(&chain.store, &txs, &chain.producer, chain.now_ms, &work_dir).unwrap();
    assert_eq!(proposed.header.index, 1);
    assert_eq!(proposed.transactions.len(), 1);

    // Canonical state is untouched: head, accounts, state root, base fee and
    // the block-by-hash index all still reflect the pre-proposal chain.
    assert_eq!(current_head(&chain.store).unwrap(), before_head);
    assert_eq!(compute_state_root(&chain.store), before_root);
    assert_eq!(
        chain.store.get_account(&sender_addr).unwrap().balance,
        10_000
    );
    assert_eq!(chain.store.get_meta("baseFee"), None);
    assert_eq!(chain.store.get_block_by_hash(&proposed.header.hash), None);

    // The checkpoint clone was cleaned up.
    assert!(work_dir.exists());
    assert_eq!(std::fs::read_dir(&work_dir).unwrap().count(), 0);

    // Production over the same mempool builds the identical block — and the
    // untouched store means the proposal did not consume any nonces.
    let produced = produce_block(&chain.store, &txs, &chain.producer, chain.now_ms).unwrap();
    assert_eq!(proposed, produced);
}

#[test]
fn propose_block_reports_gas_skips_like_production() {
    let chain = Chain::fresh();
    let sender = keypair("gas-skip-sender");
    let sender_addr = address_of(&sender);
    // 22 transfers at 50 gas each: 20 fit the 1_000 block gas limit, 2 are
    // skipped for gas and must be requeueable by the leader.
    chain.fund(&sender_addr, 10_000);
    let txs: Vec<_> = (1..=22)
        .map(|nonce| make_tx(&sender, &sender_addr, "ai_recipient", 1, nonce))
        .collect();

    let root = TempDir::new("gas-skip");
    let report = propose_block_with_report(
        &chain.store,
        &txs,
        &chain.producer,
        chain.now_ms,
        root.path(),
    )
    .unwrap();

    assert_eq!(report.block.transactions.len(), 20);
    assert_eq!(report.skipped.len(), 2);
    assert!(report
        .skipped
        .iter()
        .all(|(_, reason)| reason == GAS_LIMIT_SKIP_REASON));

    let produced =
        produce_block_with_report(&chain.store, &txs, &chain.producer, chain.now_ms).unwrap();
    assert_eq!(report.block, produced.block);
    assert_eq!(report.skipped, produced.skipped);
}

#[test]
fn commit_finalized_applies_a_valid_block_and_advances_the_head() {
    let chain = Chain::fresh();
    let sender = keypair("commit-sender");
    let sender_addr = address_of(&sender);
    chain.fund(&sender_addr, 1_000);
    let txs = vec![make_tx(&sender, &sender_addr, "ai_recipient", 100, 1)];

    let root = TempDir::new("commit-valid");
    let block = propose_block(
        &chain.store,
        &txs,
        &chain.producer,
        chain.now_ms,
        root.path(),
    )
    .unwrap();

    commit_finalized(&chain.store, &block, &chain.producer.address).unwrap();
    let head = current_head(&chain.store).unwrap();
    assert_eq!(head.header.hash, block.header.hash);
    assert_eq!(head.header.index, 1);
    assert_eq!(compute_state_root(&chain.store), block.header.state_root);
    assert_eq!(
        chain.store.get_account("ai_recipient").unwrap().balance,
        100
    );
    // transfer: 50 gas * base fee 1 = 50 charged to the sender.
    assert_eq!(
        chain.store.get_account(&sender_addr).unwrap().balance,
        1_000 - 100 - 50
    );
    // The fee market advances exactly like the production path would.
    assert_eq!(
        chain.store.meta_u64_or("baseFee", 0),
        verinoo_exec::gas::compute_base_fee(INITIAL_BASE_FEE, 50)
    );
}

#[test]
fn commit_finalized_rejects_a_wrong_state_root_and_keeps_the_head() {
    let chain = Chain::fresh();
    let sender = keypair("commit-bad-root");
    let sender_addr = address_of(&sender);
    chain.fund(&sender_addr, 1_000);
    let txs = vec![make_tx(&sender, &sender_addr, "ai_recipient", 100, 1)];

    let root = TempDir::new("commit-bad-root");
    let mut block = propose_block(
        &chain.store,
        &txs,
        &chain.producer,
        chain.now_ms,
        root.path(),
    )
    .unwrap();
    // A certificate for an invalid block carries a state root this node does
    // not derive.
    block.header.state_root = "00".repeat(32);

    let err = commit_finalized(&chain.store, &block, &chain.producer.address);
    assert!(err.is_err());
    assert!(err.unwrap_err().to_string().contains("state root mismatch"));
    let head = current_head(&chain.store).unwrap();
    assert_eq!(head.header.index, 0);
    assert_eq!(head.header.hash, "0".repeat(64));
}
