//! Read-only RPC surface for the Verinoo node (axum).
//!
//! Mutations (`POST /tx`) validate + enqueue to mempool; they never touch
//! state directly. Block/tx payloads are canonical JSON.

use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};
use std::sync::Arc;

use crate::Shared;

pub async fn serve(shared: Arc<Shared>, port: u16) -> Result<(), String> {
    let app = Router::new()
        .route("/status", get(status))
        .route("/blocks", get(blocks))
        .route("/mempool", get(mempool))
        .route("/account/:address", get(account))
        .route("/validators", get(validators))
        .route("/faucet", post(faucet))
        .route("/metrics", get(metrics))
        .route("/tx", post(submit_tx))
        .with_state(shared);

    let addr = format!("0.0.0.0:{port}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| e.to_string())?;
    tracing::info!("rpc listening on {addr}");
    axum::serve(listener, app).await.map_err(|e| e.to_string())
}

async fn status(State(shared): State<Arc<Shared>>) -> Json<serde_json::Value> {
    let head = shared.head();
    let (height, latest_hash, validator) = match head {
        Some(b) => (b.header.index, b.header.hash, b.header.validator),
        None => (0, String::new(), String::new()),
    };
    Json(serde_json::json!({
        "chainId": shared.params.chain_id,
        "height": height,
        "latestHash": latest_hash,
        "validator": validator,
        "mempool": shared.mempool.lock().await.len(),
        "validators": shared.params.validators,
    }))
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
async fn faucet(
    State(shared): State<Arc<Shared>>,
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

    let mut acc = shared.store.account_or_default(&addr);
    acc.balance += DRIP_AMOUNT;
    acc.updated_at = shared.now_ms();
    shared.store.set_account(&acc);

    Ok(Json(serde_json::json!({
        "dripped": DRIP_AMOUNT,
        "address": addr,
        "balance": acc.balance
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
        "chainId": shared.params.chain_id,
        "blockTimeMs": shared.params.block_time_ms,
    }))
}

async fn submit_tx(
    State(shared): State<Arc<Shared>>,
    Json(tx): Json<verinoo_types::tx::TxEnvelope>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, Json<serde_json::Value>)> {
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
    if let Err(e) = validator.validate_transaction(&tx) {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": e })),
        ));
    }

    let mut mp = shared.mempool.lock().await;
    if mp.iter().any(|t| t.tx_id == tx.tx_id) {
        return Err((
            axum::http::StatusCode::CONFLICT,
            Json(serde_json::json!({ "error": "already in mempool" })),
        ));
    }
    mp.push(tx.clone());

    Ok(Json(
        serde_json::json!({ "accepted": true, "txId": tx.tx_id }),
    ))
}
