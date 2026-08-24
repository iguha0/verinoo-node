//! Verinoo P2P networking — libp2p gossipsub for block/tx propagation.
//!
//! Design (Phase R2):
//!   - Two topics: `verinoo/blocks/1` and `verinoo/txs/1`
//!   - Messages carry hex-encoded canonical JSON payloads
//!   - Duplicate suppression via seen-set with LRU-ish cap
//!   - Peer scoring skeleton: ban list with expiry (R3 extends to
//!     automatic bans on invalid blocks)
//!
//! The engine boundary stays hostile-network safe: every received block
//! MUST pass validate_block before persistence — enforced by the caller.

use futures::StreamExt;
use libp2p::{gossipsub, identity, mdns, noise, swarm::NetworkBehaviour, tcp, yamux, Multiaddr};
use std::time::Duration;
use tokio::sync::mpsc;

pub const BLOCKS_TOPIC: &str = "verinoo/blocks/1";
pub const TXS_TOPIC: &str = "verinoo/txs/1";

#[derive(Debug)]
pub enum NetworkEvent {
    /// A block payload arrived on the blocks topic (hex JSON).
    Block(String),
    /// A transaction payload arrived on the txs topic.
    Tx(String),
    /// A peer was banned (id, reason).
    PeerBanned(String, String),
}

#[derive(NetworkBehaviour)]
pub struct Behaviour {
    gossipsub: gossipsub::Behaviour,
    mdns: mdns::tokio::Behaviour,
}

/// Publish-only handle (cloneable). Event stream comes back separately;
/// dedup + ban policy live with the node's shared state.
enum Command {
    PublishBlock(String),
    PublishTx(String),
}

/// Publish-only handle (cloneable). Event stream comes back separately;
/// dedup + ban policy live with the node's shared state.
#[derive(Clone)]
pub struct NetworkNode {
    _peer_id: String,
    command_tx: mpsc::UnboundedSender<Command>,
}

impl NetworkNode {
    /// Spin up a node listening on `listen_port`, optionally dialing seeds.
    /// `key_seed` must be UNIQUE per node — identical seeds mean identical
    /// PeerIds and libp2p aborts self-connections.
    #[allow(clippy::type_complexity)]
    pub async fn new(
        listen_port: u16,
        key_seed: [u8; 32],
        seed_peers: &[String],
        chain_id: &str,
    ) -> Result<(Self, mpsc::UnboundedReceiver<NetworkEvent>), Box<dyn std::error::Error>> {
        let key = identity::Keypair::ed25519_from_bytes(key_seed)?;
        let peer_id = identity::PeerId::from(key.public());

        // Gossipsub: sign messages, 1MB max, per-topic mesh.
        let gs_config = gossipsub::ConfigBuilder::default()
            .max_transmit_size(1_000_000)
            .validation_mode(gossipsub::ValidationMode::Strict)
            .heartbeat_interval(Duration::from_millis(500))
            .mesh_n_low(1) // 2-node meshes are legal on the testnet
            .mesh_outbound_min(1) // allow publishing to a single peer
            .mesh_n(6)
            .build()
            .map_err(|e| format!("gossipsub config: {e:?}"))?;

        let mut gossipsub = gossipsub::Behaviour::new(
            gossipsub::MessageAuthenticity::Signed(key.clone()),
            gs_config,
        )
        .map_err(|e| format!("gossipsub: {e:?}"))?;

        let blocks_topic = gossipsub::IdentTopic::new(format!("{chain_id}/{BLOCKS_TOPIC}"));
        let txs_topic = gossipsub::IdentTopic::new(format!("{chain_id}/{TXS_TOPIC}"));
        gossipsub.subscribe(&blocks_topic)?;
        gossipsub.subscribe(&txs_topic)?;

        let mdns = mdns::tokio::Behaviour::new(mdns::Config::default(), peer_id)?;

        let mut swarm = libp2p::SwarmBuilder::with_existing_identity(key.clone())
            .with_tokio()
            .with_tcp(
                tcp::Config::default(),
                noise::Config::new,
                yamux::Config::default,
            )?
            .with_behaviour(|_key| Behaviour { gossipsub, mdns })?
            .build();

        let listen = Multiaddr::empty()
            .with(libp2p::multiaddr::Protocol::Ip4(
                "0.0.0.0".parse().expect("valid ip"),
            ))
            .with(libp2p::multiaddr::Protocol::Tcp(listen_port));
        swarm.listen_on(listen)?;

        for peer in seed_peers {
            let addr: Multiaddr = peer
                .parse()
                .map_err(|e| format!("bad seed addr {peer}: {e:?}"))?;
            let _ = swarm.dial(addr);
        }

        let (_cmd_tx, mut cmd_rx) = mpsc::unbounded_channel::<Command>();
        let (event_tx, event_rx) = mpsc::unbounded_channel::<NetworkEvent>();

        // Bridge: outbound publishes + inbound events pumped in one task.
        let blocks_topic_str = blocks_topic.to_string();
        let txs_topic_str = txs_topic.to_string();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    cmd = cmd_rx.recv() => {
                        match cmd {
                            Some(Command::PublishBlock(payload)) => {
                                if let Err(e) = swarm.behaviour_mut().gossipsub
                                    .publish(gossipsub::IdentTopic::new(blocks_topic_str.clone()), payload.as_bytes())
                                {
                                    tracing::warn!("publish block failed: {e:?}");
                                }
                            }
                            Some(Command::PublishTx(payload)) => {
                                if let Err(e) = swarm.behaviour_mut().gossipsub
                                    .publish(gossipsub::IdentTopic::new(txs_topic_str.clone()), payload.as_bytes())
                                {
                                    tracing::warn!("publish tx failed: {e:?}");
                                }
                            }
                            None => break,
                        }
                    }
                    event = swarm.select_next_some() => {
                        match event {
                            libp2p::swarm::SwarmEvent::Behaviour(BehaviourEvent::Gossipsub(
                                gossipsub::Event::Message { message, .. },
                            )) => {
                                let topic = message.topic.as_str();
                                if let Ok(payload) = String::from_utf8(message.data) {
                                    let ev = if topic.ends_with(BLOCKS_TOPIC) {
                                        NetworkEvent::Block(payload)
                                    } else if topic.ends_with(TXS_TOPIC) {
                                        NetworkEvent::Tx(payload)
                                    } else {
                                        continue;
                                    };
                                    let _ = event_tx.send(ev);
                                }
                            }
                            libp2p::swarm::SwarmEvent::NewListenAddr { address, .. } => {
                                tracing::info!("listening on {address}");
                            }
                            _ => {}
                        }
                    }
                }
            }
        });

        let node = Self {
            _peer_id: peer_id.to_string(),
            command_tx: _cmd_tx,
        };
        Ok((node, event_rx))
    }
}

impl NetworkNode {
    /// Broadcast a block payload (JSON string).
    pub fn publish_block(&self, payload: &str) {
        let _ = self.command_tx.send(Command::PublishBlock(payload.into()));
    }

    /// Broadcast a transaction payload.
    pub fn publish_tx(&self, payload: &str) {
        let _ = self.command_tx.send(Command::PublishTx(payload.into()));
    }
}
