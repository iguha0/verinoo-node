//! Two-node gossip proof: node B publishes tx/consensus payloads, node A receives.

use verinoo_network::{NetworkEvent, NetworkNode};

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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_nodes_exchange_consensus_payloads_over_gossip() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_test_writer()
        .try_init();
    let (_node_a, mut rx_a) = NetworkNode::new(26_101, [3u8; 32], &[], "verinoo-test-consensus")
        .await
        .expect("node A");
    let (node_b, _rx_b) = NetworkNode::new(
        26_102,
        [4u8; 32],
        &["/ip4/127.0.0.1/tcp/26101".to_string()],
        "verinoo-test-consensus",
    )
    .await
    .expect("node B");

    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    let proposal = r#"{"kind":"proposal","height":7}"#;
    let vote = r#"{"kind":"vote","height":7,"validator":"v1"}"#;
    let evidence = r#"{"kind":"evidence","height":7,"reason":"double-sign"}"#;

    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(15);
    let (mut got_proposal, mut got_vote, mut got_evidence) = (false, false, false);
    while tokio::time::Instant::now() < deadline {
        while let Ok(ev) = rx_a.try_recv() {
            match ev {
                NetworkEvent::Proposal(p) if p == proposal => got_proposal = true,
                NetworkEvent::Vote(p) if p == vote => got_vote = true,
                NetworkEvent::Evidence(p) if p == evidence => got_evidence = true,
                _ => {}
            }
        }
        if got_proposal && got_vote && got_evidence {
            break;
        }
        node_b.publish_proposal(proposal);
        node_b.publish_vote(vote);
        node_b.publish_evidence(evidence);
        tokio::time::sleep(std::time::Duration::from_millis(400)).await;
    }
    assert!(got_proposal, "node A must receive the proposal within 15s");
    assert!(got_vote, "node A must receive the vote within 15s");
    assert!(got_evidence, "node A must receive the evidence within 15s");
}
