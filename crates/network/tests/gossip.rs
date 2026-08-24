//! Two-node gossip proof: node B publishes a tx payload, node A receives it.

use verinoo_network::NetworkNode;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_nodes_exchange_tx_over_gossip() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_test_writer()
        .try_init();
    let (_node_a, mut rx_a) = NetworkNode::new(26_001, [1u8; 32], &[], "verinoo-test-1")
        .await
        .expect("node A");
    // B dials A explicitly (WAN-style seed); mDNS would also find it locally.
    let (node_b, _rx_b) = NetworkNode::new(
        26_002,
        [2u8; 32],
        &["/ip4/127.0.0.1/tcp/26001".to_string()],
        "verinoo-test-1",
    )
    .await
    .expect("node B");

    // Mesh settle: connection + gossipsub graft take a moment.
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    let payload = r#"{"kind":"tx","txId":"deadbeef","from":"ai_x"}"#;

    // Publish with retries until received (mesh may graft lazily).
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(15);
    let mut received = false;
    let mut iter = 0;
    while tokio::time::Instant::now() < deadline {
        iter += 1;
        eprintln!("[loop] iter={iter} events_so_far checked");
        while let Ok(ev) = rx_a.try_recv() {
            if let verinoo_network::NetworkEvent::Tx(p) = ev {
                if p == payload {
                    received = true;
                }
            }
        }
        if received {
            break;
        }
        node_b.publish_tx(payload); // idempotent retry
        tokio::time::sleep(std::time::Duration::from_millis(400)).await;
    }
    assert!(received, "node A must receive the gossiped tx within 15s");

    // (Dedup policy lives in the node layer — tested there.)
}
