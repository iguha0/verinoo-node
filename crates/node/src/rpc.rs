//! Read-only RPC surface for the Verinoo node (axum).
//!
//! Mutations (`POST /tx`) validate + enqueue to mempool; they never touch
//! state directly. Block/tx payloads are canonical JSON.

use axum::{
    extract::{ConnectInfo, State},
    http::{HeaderMap, StatusCode},
    routing::{get, post},
    Extension, Json, Router,
};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::auth::{Auth, AuthError, SelfServeError};
use crate::readiness::{self, ReadinessInput};
use crate::Shared;

pub async fn serve(
    shared: Arc<Shared>,
    net: verinoo_network::NetworkNode,
    port: u16,
    auth: Auth,
) -> Result<(), String> {
    let app = Router::new()
        .route("/status", get(status))
        .route("/health", get(health))
        .route("/ready", get(ready))
        .route("/blocks", get(blocks))
        .route("/blocks/since/:index", get(blocks_since))
        .route("/mempool", get(mempool))
        .route("/account/:address", get(account))
        .route("/validators", get(validators))
        .route("/faucet", post(faucet))
        .route("/metrics", get(metrics))
        .route("/tx", post(submit_tx))
        // ---- v1 product surface ----
        .route("/v1/models", get(v1_models))
        .route("/v1/input-key", get(v1_input_key))
        .route("/v1/keys", post(v1_issue_key))
        .route("/v1/infer", post(submit_tx))
        .route("/v1/tasks/:task_id", get(v1_task))
        .route("/v1/verify/:task_id", get(v1_verify))
        .route("/v1/usage/:address", get(v1_usage))
        .route("/v1/usage/:address/csv", get(v1_usage_csv))
        .layer(axum::middleware::from_fn(cors))
        .with_state(shared)
        .layer(Extension(net))
        .layer(Extension(auth))
        .layer(Extension(KeyRateLimit::default()));

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
    let (serving, height, ts) = match &head {
        Some(block) => (true, block.header.index, block.header.timestamp),
        None => (false, 0, 0),
    };
    let head_age_ms = now_ms.saturating_sub(ts);
    let syncing = shared.syncing.load(Ordering::Relaxed);
    let decision = readiness::evaluate(ReadinessInput {
        serving,
        syncing,
        height,
        head_age_ms,
        block_time_ms: shared.block_time_ms,
        best_peer_height: shared.best_peer_height(),
        peer_lag_blocks: shared.peer_lag_blocks,
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
    Json(serde_json::json!({
        "chainId": shared.params.chain_id,
        "height": height,
        "latestHash": latest_hash,
        "validator": validator,
        "headAgeMs": head_age_ms,
        "peerCount": net.peer_count(),
        "syncing": shared.syncing.load(Ordering::Relaxed),
        "validatorSetHash": shared.validator_set_hash(),
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
        if caller.address != tx.from {
            return Err(auth_error(AuthError::AddressMismatch));
        }
        tracing::debug!(
            tenant = %caller.tenant,
            address = %caller.address,
            "api key authorized"
        );
        Some(caller)
    } else {
        None
    };

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

    if tx.data.r#type == verinoo_types::tx::TxKind::SubmitInference {
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
        let bad = |msg: &str| {
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": msg })),
            )
        };
        if has_encrypted && has_plain {
            return Err(bad("submit either input or encryptedInput, not both"));
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
    }

    let mut mp = shared.mempool.lock().await;
    if mp.iter().any(|t| t.tx_id == tx.tx_id) {
        return Err((
            axum::http::StatusCode::CONFLICT,
            Json(serde_json::json!({ "error": "already in mempool" })),
        ));
    }
    if let Some(caller) = &caller {
        if tx.data.r#type == verinoo_types::tx::TxKind::SubmitInference {
            auth.consume_quota(&shared.store, caller, shared.now_ms())
                .map_err(auth_error)?;
        }
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

async fn v1_task(
    State(shared): State<Arc<Shared>>,
    axum::extract::Path(task_id): axum::extract::Path<String>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, Json<serde_json::Value>)> {
    let task = shared
        .store
        .get_task(&task_id)
        .ok_or_else(|| not_found("task not found"))?;
    let receipt = verinoo_exec::engine::receipt_for(&shared.store, &task_id);
    let input_commitment = task.input_commitment.clone();
    let encrypted = task.encrypted_input.is_some();
    let mut task_json = serde_json::to_value(&task).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
    })?;
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
    Ok(Json(serde_json::json!({
        "task": task_json,
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
    let recomputed = match model.result_json(&plaintext) {
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
            policy: std::sync::Mutex::new(crate::Policy::default()),
        })
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
}
