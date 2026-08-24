//! Conformance tests — byte-parity with the TypeScript reference client.

#[cfg(test)]
mod conformance {
    use ed25519_dalek::{Signature, Verifier, VerifyingKey};
    use serde_json::Value;
    use sha2::{Digest, Sha256};
    use verinoo_types::canonical::canonical_json;
    use verinoo_types::tx::TxEnvelope;

    pub(crate) fn load_fixture(name: &str) -> Value {
        let path = format!("{}/fixtures/{}", env!("CARGO_MANIFEST_DIR"), name);
        let raw = std::fs::read_to_string(path).expect("fixture file present");
        serde_json::from_str(&raw).expect("fixture parses")
    }

    #[test]
    fn canonical_encoding_matches_reference_byte_for_byte() {
        let cases = load_fixture("canonical.json");
        let arr = cases.as_array().expect("array");
        for case in arr {
            let name = case["name"].as_str().unwrap();
            let input: Value = case["input"].clone();
            let expected = case["expected"].as_str().unwrap();

            // Key order in the input JSON is intentionally shuffled by the
            // exporter; serde maps preserve insertion, canonical must not care.
            let got = canonical_json(&input)
                .unwrap_or_else(|e| panic!("case {name}: canonical failed: {e}"));
            assert_eq!(got, expected, "canonical mismatch for case {name}");
        }
    }

    #[test]
    fn tx_vectors_match_canonical_string_and_txid() {
        let fx = load_fixture("txs.json");
        assert_eq!(
            fx["scheme"].as_str().unwrap(),
            "ed25519-detached-over-txid-utf8"
        );

        for vector in fx["vectors"].as_array().unwrap() {
            let name = vector["name"].as_str().unwrap();

            // Parse the full envelope through our strict types.
            let envelope: TxEnvelope = serde_json::from_value(vector["tx"].clone())
                .unwrap_or_else(|e| panic!("case {name}: envelope parse failed: {e}"));

            // Canonical string must match the reference byte-for-byte.
            let payload = verinoo_types::canonical::UnsignedPayload {
                data: serde_json::to_value(&envelope.data).unwrap(),
                from: envelope.from.clone(),
                nonce: envelope.nonce,
                to: envelope.to.clone(),
                value: envelope.value,
                gas_limit: envelope.gas_limit,
                gas_price: envelope.gas_price,
            };
            let got = payload
                .to_canonical_string()
                .unwrap_or_else(|e| panic!("case {name}: {e}"));
            let want = vector["payloadCanonical"].as_str().unwrap();
            assert_eq!(got, want, "canonical string mismatch for {name}");

            // txId = sha256(canonical string).
            let mut h = Sha256::new();
            h.update(got.as_bytes());
            let tx_id = hex::encode(h.finalize());
            assert_eq!(
                tx_id,
                vector["txId"].as_str().unwrap(),
                "txId mismatch for {name}"
            );
            assert_eq!(
                tx_id, envelope.tx_id,
                "envelope txId field mismatch for {name}"
            );
        }
    }

    #[test]
    fn ed25519_signatures_verify_over_txid_utf8() {
        let fx = load_fixture("txs.json");

        assert_eq!(
            fx["scheme"].as_str().unwrap(),
            "ed25519-detached-over-txid-utf8"
        );

        for vector in fx["vectors"].as_array().unwrap() {
            let name = vector["name"].as_str().unwrap();
            let tx_id = vector["txId"].as_str().unwrap();
            let sig_hex = vector["tx"]["signature"].as_str().unwrap();
            let pk_hex = vector["tx"]["publicKey"].as_str().unwrap();

            let vk = VerifyingKey::from_bytes(&hex::decode(pk_hex).unwrap().try_into().unwrap())
                .expect("valid verifying key");
            let sig = Signature::from_slice(&hex::decode(sig_hex).unwrap())
                .expect("signature is 64 bytes");

            // Ed25519 detached over the UTF-8 bytes of the txId string.
            vk.verify(tx_id.as_bytes(), &sig)
                .unwrap_or_else(|e| panic!("{name}: signature must verify ({e})"));

            // Tampered txId must fail.
            let mut bad = tx_id.as_bytes().to_vec();
            bad[0] ^= 0xff;
            assert!(
                vk.verify(&bad, &sig).is_err(),
                "{name}: tampered txId must fail"
            );
        }
    }

    #[test]
    fn tampered_content_breaks_txid_binding() {
        let fx = load_fixture("txs.json");
        let mut vector = fx["vectors"][0].clone(); // transfer
                                                   // redirect the payment
        vector["tx"]["to"] = Value::String("ai_attacker".into());

        let envelope: TxEnvelope = serde_json::from_value(vector["tx"].clone()).unwrap();
        let payload = verinoo_types::canonical::UnsignedPayload {
            data: serde_json::to_value(&envelope.data).unwrap(),
            from: envelope.from.clone(),
            nonce: envelope.nonce,
            to: envelope.to.clone(),
            value: envelope.value,
            gas_limit: envelope.gas_limit,
            gas_price: envelope.gas_price,
        };
        let recomputed = payload.tx_id().unwrap();
        assert_ne!(
            recomputed, envelope.tx_id,
            "redirected payment must change txId"
        );
    }
}

#[cfg(test)]
mod apply_blocks_conformance {
    use crate::conformance::load_fixture;

    #[test]
    fn apply_blocks_replay_matches_reference_state() {
        let fx = load_fixture("apply_blocks.json");
        let validator_addr = fx["validatorAddress"].as_str().unwrap().to_string();
        let base_fee = fx["baseFee"].as_u64().unwrap();

        // Seed genesis allocations exactly as exported.
        let dir = std::env::temp_dir().join(format!(
            "verinoo-conform-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .subsec_nanos()
        ));
        let store = verinoo_state::StateStore::open(&dir).expect("open state");

        for (addr, spec) in fx["initialAccounts"].as_object().unwrap() {
            let acc = verinoo_state::models::Account {
                address: addr.clone(),
                public_key: String::new(),
                nonce: spec["nonce"].as_u64().unwrap(),
                balance: spec["balance"].as_i64().unwrap(),
                updated_at: 0,
            };
            store.set_account(&acc);
        }

        for block_fx in fx["blocks"].as_array().unwrap() {
            let name = block_fx["name"].as_str().unwrap();
            let height = block_fx["height"].as_u64().unwrap();

            // Reconstruct the block body from the fixture's canonical txs.
            let mut transactions = Vec::new();
            for t in block_fx["transactions"].as_array().unwrap() {
                transactions.push(
                    serde_json::from_value::<verinoo_types::tx::TxEnvelope>(t.clone())
                        .unwrap_or_else(|e| panic!("{name}: envelope parse failed: {e}")),
                );
            }
            if std::env::var("CONF_DEBUG").is_ok() {
                let v = verinoo_exec::Validator {
                    store: &store,
                    base_fee,
                };
                for t in &transactions {
                    if let Err(e) = v.validate_transaction(t) {
                        eprintln!("[debug] {name} tx {} rejected: {e}", t.tx_id);
                        eprintln!("[debug]   tx dump: {}", serde_json::to_string(&t).unwrap());
                    }
                }
            }
            let block = verinoo_exec::engine::Block {
                header: verinoo_exec::engine::BlockHeader {
                    hash: String::new(),
                    version: 1,
                    index: height,
                    timestamp: 0,
                    previous_hash: String::new(),
                    validator: String::new(),
                    validator_pub_key: String::new(),
                    validator_signature: String::new(),
                    state_root: String::new(),
                    tx_root: String::new(),
                    inference_tasks_root: String::new(),
                    compute_root: String::new(),
                },
                transactions,
            };

            let applied =
                verinoo_exec::engine::apply_block(&store, &block, base_fee, &validator_addr)
                    .unwrap_or_else(|e| panic!("{name}: apply failed: {e}"));

            // Executed set must match reference exactly.
            let mut got_ids = applied.executed_ids.clone();
            got_ids.sort();
            let mut want_ids: Vec<String> = block_fx["executedTxIds"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect();
            want_ids.sort();
            assert_eq!(got_ids, want_ids, "{name}: executed tx mismatch");

            // Observable accounts must match.
            for (addr, spec) in block_fx["expect"]["accounts"].as_object().unwrap() {
                let acc = store.account_or_default(addr);
                assert_eq!(
                    acc.balance,
                    spec["balance"].as_i64().unwrap(),
                    "{name}: balance mismatch for {addr}"
                );
                assert_eq!(
                    acc.nonce,
                    spec["nonce"].as_u64().expect("nonce present"),
                    "{name}: nonce mismatch for {addr}"
                );
            }

            // Task lifecycle.
            let task_expect = block_fx["expect"]["task"].clone();
            if !task_expect.is_null() {
                let task_id = fx["taskId"].as_str().unwrap();
                let all_tasks = store.iter_prefix::<verinoo_state::models::InferenceTask>("task:");
                eprintln!(
                    "[dbg] {name}: looking for task {task_id}; present: {:?}",
                    all_tasks
                        .iter()
                        .map(|t| (t.task_id.clone(), t.status.clone()))
                        .collect::<Vec<_>>()
                );
                let task = store.get_task(task_id).expect("task exists");
                assert_eq!(task.status, task_expect["status"], "{name}: task status");
                assert_eq!(
                    task.assigned_to.as_deref(),
                    task_expect["assignedTo"].as_str(),
                    "{name}: assignee"
                );
            }

            // Game progression.
            let game_expect = block_fx["expect"]["game"].clone();
            if !game_expect.is_null() {
                let game_id = fx["gameId"].as_str().unwrap();
                let game = store.get_game(game_id).expect("game exists");
                assert_eq!(game["status"], game_expect["status"], "{name}: game status");
                assert_eq!(game["low"], game_expect["low"], "{name}: low");
                assert_eq!(game["high"], game_expect["high"], "{name}: high");
                assert_eq!(
                    game["currentStep"], game_expect["currentStep"],
                    "{name}: step"
                );
                assert_eq!(
                    game["disputedLayer"], game_expect["disputedLayer"],
                    "{name}: disputed layer"
                );
            }
        }
    }
}
