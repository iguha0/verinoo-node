//! CLI wallet — generate keys, query balances, send transfers.
//!
//! Usage:
//!   verinoo-node wallet --key-file key.json balance [--node http://localhost:7100]
//!   verinoo-node wallet --key-file key.json send --to <addr> --amount <n> [--node ...]
//!   verinoo-node wallet generate --output new-key.json
//!
//! Key file format (JSON): { publicKey, privateKey(base64), address }

use base64::Engine as _;
use ed25519_dalek::{Signer as _, SigningKey};
use serde_json::json;
use sha2::Digest as _;
use verinoo_types::canonical::canonical_json;

pub fn public_key_to_address(pk_hex: &str) -> String {
    let bytes = hex::decode(pk_hex).unwrap_or_default();
    let mut h = sha2::Sha256::new();
    h.update(&bytes);
    format!("ai_{}", hex::encode(&h.finalize()[..20]))
}

/// Load or create a keypair from a JSON file.
pub fn load_key(path: &str) -> Result<(String, String, String), String> {
    if std::path::Path::new(path).exists() {
        let raw = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let v: serde_json::Value = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
        let pub_hex = v["publicKey"]
            .as_str()
            .ok_or("missing publicKey")?
            .to_string();
        let priv_b64 = v["privateKey"]
            .as_str()
            .ok_or("missing privateKey")?
            .to_string();
        let addr = v["address"].as_str().ok_or("missing address")?.to_string();
        Ok((pub_hex, priv_b64, addr))
    } else {
        Err(format!("key file not found: {path}"))
    }
}

/// Generate a new keypair and save to file.
pub fn generate_key(path: &str) -> Result<(), String> {
    let sk = SigningKey::generate(&mut rand::rngs::OsRng);
    let vk = sk.verifying_key();
    let pk_hex = hex::encode(vk.to_bytes());
    let priv_b64 = base64::engine::general_purpose::STANDARD.encode(sk.to_bytes());
    let addr = public_key_to_address(&pk_hex);

    let data = json!({
        "publicKey": pk_hex,
        "privateKey": priv_b64,
        "address": addr,
    });
    std::fs::write(
        path,
        serde_json::to_string_pretty(&data).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;

    println!("✓ New keypair written to {path}");
    println!("  Address:   {addr}");
    println!("  PublicKey: {}...", &pk_hex[..16]);
    Ok(())
}

/// Query an address's balance via RPC.
pub async fn query_balance(rpc_url: &str, address: &str) -> Result<u64, String> {
    let url = format!("{rpc_url}/account/{address}");
    let resp = reqwest::get(&url).await.map_err(|e| e.to_string())?;
    let text = resp.text().await.map_err(|e| e.to_string())?;
    let body: serde_json::Value = serde_json::from_str(&text).unwrap_or(serde_json::json!({}));
    if body.get("error").is_some() || body.get("balance").is_none() {
        return Ok(0);
    }
    Ok(body["balance"].as_u64().unwrap_or(0))
}

/// Build a signed transfer envelope. Used by the CLI wallet and by the
/// faucet, so both produce byte-identical transactions.
pub fn build_signed_transfer(
    signing: &SigningKey,
    from_addr: &str,
    to: &str,
    amount: u64,
    nonce: u64,
) -> Result<serde_json::Value, String> {
    let data = json!({ "type": "transfer", "data": {} });
    let payload = json!({
        "data": data,
        "from": from_addr,
        "nonce": nonce,
        "to": to,
        "value": amount,
    });

    // Canonical encoding
    let payload_str = canonical_json(&payload).map_err(|e| e.to_string())?;
    let mut h = sha2::Sha256::new();
    h.update(payload_str.as_bytes());
    let tx_id = hex::encode(h.finalize());

    // Ed25519 signature over txId UTF-8 bytes; transported as HEX
    let sig = signing.sign(tx_id.as_bytes());
    let sig_hex = hex::encode(sig.to_bytes());
    let pk_hex = hex::encode(signing.verifying_key().to_bytes());

    Ok(json!({
        "txId": tx_id,
        "from": from_addr,
        "to": to,
        "value": amount,
        "nonce": nonce,
        "data": data,
        "publicKey": pk_hex,
        "signature": sig_hex,
    }))
}

/// Build a signed `submitInference` envelope for the built-in model.
pub fn build_signed_inference(
    signing: &SigningKey,
    from_addr: &str,
    model: &str,
    input: &str,
    max_fee: u64,
    deadline: u64,
    nonce: u64,
) -> Result<serde_json::Value, String> {
    let mut h = sha2::Sha256::new();
    h.update(input.as_bytes());
    let input_commitment = hex::encode(h.finalize());

    let data = json!({
        "type": "submitInference",
        "data": {
            "requester": from_addr,
            "targetModel": model,
            "input": input,
            "inputCommitment": input_commitment,
            "maxFee": max_fee,
            "deadline": deadline,
            "verificationType": "optimistic",
        }
    });
    let payload = json!({
        "data": data,
        "from": from_addr,
        "nonce": nonce,
        "to": "",
        "value": 0,
    });

    let payload_str = canonical_json(&payload).map_err(|e| e.to_string())?;
    let mut h = sha2::Sha256::new();
    h.update(payload_str.as_bytes());
    let tx_id = hex::encode(h.finalize());

    let sig = signing.sign(tx_id.as_bytes());
    Ok(json!({
        "txId": tx_id,
        "from": from_addr,
        "to": "",
        "value": 0,
        "nonce": nonce,
        "data": data,
        "publicKey": hex::encode(signing.verifying_key().to_bytes()),
        "signature": hex::encode(sig.to_bytes()),
    }))
}

/// Build + submit a signed transfer transaction.
pub async fn send_transfer(
    rpc_url: &str,
    from_addr: &str,
    secret_b64: &str,
    to: &str,
    amount: u64,
    nonce: u64,
    api_key: Option<&str>,
) -> Result<String, String> {
    let sk_bytes = base64::engine::general_purpose::STANDARD
        .decode(secret_b64)
        .map_err(|e| e.to_string())?;
    let sk_arr: [u8; 32] = sk_bytes[..32]
        .try_into()
        .map_err(|_| "bad key length".to_string())?;
    let signing = SigningKey::from_bytes(&sk_arr);

    let body = build_signed_transfer(&signing, from_addr, to, amount, nonce)?;

    let client = reqwest::Client::new();
    let mut req = client
        .post(format!("{rpc_url}/tx"))
        .header("content-type", "application/json");
    if let Some(key) = api_key {
        req = req.header("authorization", format!("Bearer {key}"));
    }
    let resp = req
        .body(serde_json::to_string(&body).map_err(|e| e.to_string())?)
        .send()
        .await
        .map_err(|e| e.to_string())?;

    let status = resp.status().as_u16();
    let text = resp.text().await.map_err(|e| e.to_string())?;
    if status == 200 {
        Ok(text)
    } else {
        Err(format!("HTTP {status}: {text}"))
    }
}
