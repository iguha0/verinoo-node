//! Conformance tests — byte-parity with the TypeScript reference client.

#[cfg(test)]
mod conformance {
    use ed25519_dalek::{Signature, Verifier, VerifyingKey};
    use serde_json::Value;
    use sha2::{Digest, Sha256};
    use verinoo_types::canonical::canonical_json;
    use verinoo_types::tx::TxEnvelope;

    fn load_fixture(name: &str) -> Value {
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
