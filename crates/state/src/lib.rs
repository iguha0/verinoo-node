//! RocksDB-backed canonical state.
//!
//! Key scheme (prefix-addressed KV):
//!   acc:<address>      Account
//!   model:<model_id>   ModelRegistryEntry
//!   task:<task_id>     InferenceTask
//!   cnode:<node_id>    ComputeNode
//!   agent:<address>    AgentAccount
//!   game:<game_id>     VerificationGame (opaque JSON until R4)
//!   meta:<key>         raw JSON values (latest head, baseFee, ...)

pub mod genesis;
pub mod models;
pub mod validators;

use crate::models::{Account, AgentAccount, ComputeNode, InferenceTask, ModelRegistryEntry};
use rocksdb::{DBWithThreadMode, SingleThreaded};
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug)]
pub struct StateStore {
    db: DBWithThreadMode<SingleThreaded>,
}

fn prefixed(prefix: &str, key: &str) -> String {
    format!("{prefix}:{key}")
}

impl StateStore {
    pub fn open(path: &Path) -> Result<Self, rocksdb::Error> {
        let db = DBWithThreadMode::open_default(path)?;
        Ok(Self { db })
    }

    pub fn open_temp() -> Result<Self, rocksdb::Error> {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir =
            std::env::temp_dir().join(format!("verinoo-state-test-{}-{n}", std::process::id()));
        Self::open(&dir)
    }

    fn del_raw(&self, key: &str) {
        self.db.delete(key.as_bytes()).expect("rocksdb delete");
    }
    fn get_raw(&self, key: &str) -> Option<Vec<u8>> {
        self.db.get(key.as_bytes()).expect("rocksdb read")
    }
    fn put_raw(&self, key: &str, value: &[u8]) {
        self.db.put(key.as_bytes(), value).expect("rocksdb write");
    }
    fn get_typed<T: DeserializeOwned>(&self, key: &str) -> Option<T> {
        let raw = self.get_raw(key)?;
        Some(serde_json::from_slice(&raw).expect("state JSON corruption"))
    }
    fn put_typed<T: Serialize>(&self, key: &str, value: &T) {
        let json = serde_json::to_vec(value).expect("serialize state entity");
        self.put_raw(key, &json);
    }

    // ---- accounts ----
    pub fn get_account(&self, address: &str) -> Option<Account> {
        self.get_typed(&prefixed("acc", address))
    }
    pub fn set_account(&self, account: &Account) {
        self.put_typed(&prefixed("acc", &account.address), account);
    }
    /// Absent accounts read as zero-balance zero-nonce (reference behavior).
    pub fn account_or_default(&self, address: &str) -> Account {
        self.get_account(address).unwrap_or(Account {
            address: address.to_string(),
            public_key: String::new(),
            nonce: 0,
            balance: 0,
            updated_at: 0,
        })
    }

    // ---- models ----
    pub fn set_model(&self, model: &ModelRegistryEntry) {
        self.put_typed(&prefixed("model", &model.model_id), model);
    }
    pub fn get_model(&self, model_id: &str) -> Option<ModelRegistryEntry> {
        self.get_typed(&prefixed("model", model_id))
    }
    pub fn iter_models(&self) -> Vec<ModelRegistryEntry> {
        self.iter_prefix("model:")
    }

    // ---- tasks ----
    pub fn set_task(&self, task: &InferenceTask) {
        self.put_typed(&prefixed("task", &task.task_id), task);
    }
    pub fn get_task(&self, task_id: &str) -> Option<InferenceTask> {
        self.get_typed(&prefixed("task", task_id))
    }
    pub fn iter_tasks_by_status(&self, status: &str) -> Vec<InferenceTask> {
        let tasks: Vec<InferenceTask> = self.iter_prefix("task:");
        tasks.into_iter().filter(|t| t.status == status).collect()
    }

    // ---- compute nodes ----
    pub fn set_node(&self, node: &ComputeNode) {
        self.put_typed(&prefixed("cnode", &node.node_id), node);
    }
    pub fn get_node(&self, node_id: &str) -> Option<ComputeNode> {
        self.get_typed(&prefixed("cnode", node_id))
    }
    pub fn iter_nodes_by_reputation(&self) -> Vec<ComputeNode> {
        let mut nodes = self.iter_prefix::<ComputeNode>("cnode:");
        nodes.sort_by_key(|n| std::cmp::Reverse(n.reputation));
        nodes
    }

    // ---- agents ----
    pub fn set_agent(&self, agent: &AgentAccount) {
        self.put_typed(&prefixed("agent", &agent.address), agent);
    }
    pub fn get_agent(&self, address: &str) -> Option<AgentAccount> {
        self.get_typed(&prefixed("agent", address))
    }

    // ---- games (opaque until R4 typed) ----
    pub fn set_game(&self, game_id: &str, game: &Value) {
        self.put_typed(&prefixed("game", game_id), game);
    }
    pub fn get_game(&self, game_id: &str) -> Option<Value> {
        self.get_typed(&prefixed("game", game_id))
    }

    // ---- blocks (hash-addressed JSON; schema owned by exec crate) ----
    pub fn put_block_by_hash(&self, hash: &str, value: &Value) {
        self.put_typed(&format!("hash:{hash}"), value);
    }
    pub fn get_block_by_hash(&self, hash: &str) -> Option<Value> {
        self.get_typed(&format!("hash:{hash}"))
    }

    // ---- meta ----
    pub fn set_meta(&self, key: &str, value: &Value) {
        self.put_typed(&prefixed("meta", key), value);
    }
    pub fn get_meta(&self, key: &str) -> Option<Value> {
        self.get_typed(&prefixed("meta", key))
    }
    pub fn meta_u64_or(&self, key: &str, default: u64) -> u64 {
        self.get_meta(key)
            .and_then(|v| v.as_u64())
            .unwrap_or(default)
    }
    pub fn set_meta_u64(&self, key: &str, v: u64) {
        self.set_meta(key, &serde_json::json!(v));
    }

    // ---- iteration helper ----
    pub fn iter_prefix<T: DeserializeOwned>(&self, prefix: &str) -> Vec<T> {
        let mut it = self.db.raw_iterator();
        it.seek(prefix.as_bytes());
        let mut out = Vec::new();
        while it.valid() {
            let k = match it.key() {
                Some(k) => k,
                None => break,
            };
            if !k.starts_with(prefix.as_bytes()) {
                break;
            }
            let v = it.value().unwrap();
            out.push(serde_json::from_slice(v).expect("state JSON corruption"));
            it.next();
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn account_roundtrip_and_default() {
        let store = StateStore::open_temp().unwrap();
        assert_eq!(store.get_account("ai_x"), None);
        let acc = store.account_or_default("ai_x");
        assert_eq!(acc.balance, 0);
        store.set_account(&Account {
            address: "ai_x".into(),
            public_key: "pk".into(),
            nonce: 4,
            balance: 1000,
            updated_at: 7,
        });
        let got = store.get_account("ai_x").unwrap();
        assert_eq!(got.nonce, 4);
        assert_eq!(got.balance, 1000);
    }

    #[test]
    fn meta_and_tasks() {
        let store = StateStore::open_temp().unwrap();
        store.set_meta_u64("baseFee", 3);
        assert_eq!(store.meta_u64_or("baseFee", 1), 3);
        assert_eq!(store.meta_u64_or("missing", 9), 9);

        let task: InferenceTask =
            serde_json::from_value(json!({"taskId":"t1","requester":"r","targetModel":"m",
                "maxFee":10,"deadline":5,"status":"pending"}))
            .unwrap();
        store.set_task(&task);
        assert_eq!(store.iter_tasks_by_status("pending").len(), 1);
        assert_eq!(store.iter_tasks_by_status("assigned").len(), 0);
    }
}

// ---- validator set (Phase R3) ----
use crate::validators::ValidatorEntry;

impl StateStore {
    pub fn set_validator(&self, v: &ValidatorEntry) {
        self.put_typed(&prefixed("val", &v.address), v);
    }
    pub fn get_validator(&self, address: &str) -> Option<ValidatorEntry> {
        self.get_typed(&prefixed("val", address))
    }
    pub fn remove_validator(&self, address: &str) {
        self.del_raw(&prefixed("val", address));
    }
    /// ACTIVE validators sorted by address — the leader-election order.
    pub fn active_validators(&self) -> Vec<ValidatorEntry> {
        let mut vals = self.iter_prefix::<ValidatorEntry>("val:");
        vals.sort_by(|a, b| a.address.cmp(&b.address));
        vals
    }
    /// Active validator ADDRESSES in election order.
    pub fn validator_addresses(&self) -> Vec<String> {
        self.active_validators()
            .into_iter()
            .map(|v| v.address)
            .collect()
    }
}

// ---- genesis seeding ----
impl StateStore {
    /// Seed the initial validator set (genesis-approved addresses).
    /// Entries carry the `genesis` flag: exempt from stake minimums.
    pub fn seed_genesis_validators(&self, addresses: &[String], height: u64) {
        for addr in addresses {
            if self.get_validator(addr).is_some() {
                continue;
            }
            self.set_validator(&crate::validators::ValidatorEntry {
                address: addr.clone(),
                public_key: String::new(),
                staked_amount: 0,
                registered_at_height: height,
                genesis: true,
            });
        }
    }
}
