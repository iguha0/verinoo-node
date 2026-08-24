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

async fn mempool(State(shared): State<Arc<Shared>>) -> Json<serde_json::Value> {
    let mp = shared.mempool.lock().await;
    Json(serde_json::json!({ "count": mp.len() }))
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
