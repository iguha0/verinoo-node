//! Read-only RPC surface for the Verinoo node (axum).
//!
//! Mutations (`POST /tx`) validate + enqueue to mempool; they never touch
//! state directly. Block/tx payloads are canonical JSON.

use axum::{
    extract::State,
    routing::{get, post},
    Extension, Json, Router,
};
use std::sync::Arc;

use crate::Shared;

pub async fn serve(
    shared: Arc<Shared>,
    net: verinoo_network::NetworkNode,
    port: u16,
) -> Result<(), String> {
    let app = Router::new()
        .route("/status", get(status))
        .route("/health", get(health))
        .route("/blocks", get(blocks))
        .route("/mempool", get(mempool))
        .route("/account/:address", get(account))
        .route("/validators", get(validators))
        .route("/faucet", post(faucet))
        .route("/metrics", get(metrics))
        .route("/tx", post(submit_tx))
        // ---- v1 product surface ----
        .route("/v1/models", get(v1_models))
        .route("/v1/infer", post(submit_tx))
        .route("/v1/tasks/:task_id", get(v1_task))
        .route("/v1/verify/:task_id", get(v1_verify))
        .route("/v1/usage/:address", get(v1_usage))
        .route("/v1/usage/:address/csv", get(v1_usage_csv))
        .layer(axum::middleware::from_fn(cors))
        .with_state(shared)
        .layer(Extension(net));

    let addr = format!("0.0.0.0:{port}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| e.to_string())?;
    tracing::info!("rpc listening on {addr}");
    axum::serve(listener, app).await.map_err(|e| e.to_string())
}

/// Liveness/readiness probe for load balancers and systemd.
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
        HeaderValue::from_static("content-type"),
    );
    res
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
        "chainId": shared.params.chain_id,
        "blockTimeMs": shared.params.block_time_ms,
    }))
}

async fn submit_tx(
    State(shared): State<Arc<Shared>>,
    Extension(net): Extension<verinoo_network::NetworkNode>,
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
    let payload = serde_json::to_string(&tx).map_err(|e| {
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
    })?;
    mp.push(tx.clone());
    drop(mp);

    // Without this, a transaction submitted to a non-validator node never
    // leaves the process and is silently lost.
    net.publish_tx(&payload);

    Ok(Json(serde_json::json!({
        "accepted": true,
        "txId": tx.tx_id,
        "taskId": verinoo_exec::engine::task_id_for(&tx),
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
        "runtime": "deterministic-q16.16",
        "verification": "recompute",
    }))
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
    Ok(Json(serde_json::json!({
        "task": task,
        "receipt": receipt,
        "anchored": receipt.is_some(),
    })))
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

    let recomputed = model.result_json(&task.input);
    let recomputed_hash = verinoo_model::sha256_hex(&recomputed);
    let on_chain = task.result_hash.clone().unwrap_or_default();
    let receipt = verinoo_exec::engine::receipt_for(&shared.store, &task_id);
    let matches = recomputed_hash == on_chain;

    Ok(Json(serde_json::json!({
        "taskId": task_id,
        "verified": matches && receipt.is_some(),
        "outputMatches": matches,
        "anchored": receipt.is_some(),
        "recomputedHash": recomputed_hash,
        "onChainHash": on_chain,
        "output": recomputed,
        "modelId": model.model_id(),
        "weightsHash": model.weights_hash(),
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
    axum::extract::Path(address): axum::extract::Path<String>,
) -> Json<serde_json::Value> {
    Json(usage_snapshot(&shared, &address))
}

/// Metering export for invoicing.
async fn v1_usage_csv(
    State(shared): State<Arc<Shared>>,
    axum::extract::Path(address): axum::extract::Path<String>,
) -> impl axum::response::IntoResponse {
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
    ([(axum::http::header::CONTENT_TYPE, "text/csv")], body)
}
