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
use libp2p::{
    gossipsub, identity, mdns, noise, swarm::NetworkBehaviour, tcp, yamux, Multiaddr, PeerId,
};
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
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

/// Holds established connections open.
///
/// In this libp2p version gossipsub only answers `connection_keep_alive() ==
/// true` once the peer is already in a mesh, and mesh membership cannot be
/// established before the connection survives long enough to exchange
/// subscriptions. Every link is therefore closed with `KeepAliveTimeout` a
/// couple of milliseconds after it opens and no block or transaction ever
/// propagates. This behaviour's only job is to keep the connection alive.
mod keepalive {
    use libp2p::core::upgrade::DeniedUpgrade;
    use libp2p::core::Endpoint;
    use libp2p::swarm::behaviour::FromSwarm;
    use libp2p::swarm::handler::{ConnectionEvent, ConnectionHandler};
    use libp2p::swarm::{
        ConnectionDenied, ConnectionHandlerEvent, ConnectionId, NetworkBehaviour,
        SubstreamProtocol, ToSwarm,
    };
    use libp2p::{Multiaddr, PeerId};
    use std::convert::Infallible;
    use std::task::{Context, Poll};

    pub struct KeepAlive;

    impl NetworkBehaviour for KeepAlive {
        type ConnectionHandler = Handler;
        type ToSwarm = Infallible;

        fn handle_established_inbound_connection(
            &mut self,
            _: ConnectionId,
            _: PeerId,
            _: &Multiaddr,
            _: &Multiaddr,
        ) -> Result<Self::ConnectionHandler, ConnectionDenied> {
            Ok(Handler)
        }

        fn handle_established_outbound_connection(
            &mut self,
            _: ConnectionId,
            _: PeerId,
            _: &Multiaddr,
            _: Endpoint,
        ) -> Result<Self::ConnectionHandler, ConnectionDenied> {
            Ok(Handler)
        }

        fn on_swarm_event(&mut self, _: FromSwarm) {}

        fn on_connection_handler_event(
            &mut self,
            _: PeerId,
            _: ConnectionId,
            event: <Self::ConnectionHandler as ConnectionHandler>::ToBehaviour,
        ) {
            match event {}
        }

        fn poll(
            &mut self,
            _: &mut Context<'_>,
        ) -> Poll<
            ToSwarm<Self::ToSwarm, <Self::ConnectionHandler as ConnectionHandler>::FromBehaviour>,
        > {
            Poll::Pending
        }
    }

    pub struct Handler;

    impl ConnectionHandler for Handler {
        type FromBehaviour = Infallible;
        type ToBehaviour = Infallible;
        type InboundProtocol = DeniedUpgrade;
        type OutboundProtocol = DeniedUpgrade;
        type InboundOpenInfo = ();
        type OutboundOpenInfo = ();

        fn listen_protocol(
            &self,
        ) -> SubstreamProtocol<Self::InboundProtocol, Self::InboundOpenInfo> {
            SubstreamProtocol::new(DeniedUpgrade, ())
        }

        fn on_behaviour_event(&mut self, event: Self::FromBehaviour) {
            match event {}
        }

        fn on_connection_event(
            &mut self,
            _: ConnectionEvent<
                Self::InboundProtocol,
                Self::OutboundProtocol,
                Self::InboundOpenInfo,
                Self::OutboundOpenInfo,
            >,
        ) {
        }

        fn poll(
            &mut self,
            _: &mut Context<'_>,
        ) -> Poll<
            ConnectionHandlerEvent<
                Self::OutboundProtocol,
                Self::OutboundOpenInfo,
                Self::ToBehaviour,
            >,
        > {
            Poll::Pending
        }

        fn connection_keep_alive(&self) -> bool {
            true
        }
    }
}

#[derive(NetworkBehaviour)]
pub struct Behaviour {
    gossipsub: gossipsub::Behaviour,
    keepalive: keepalive::KeepAlive,
    /// Neither gossipsub nor identify keeps an idle connection alive in this
    /// version: their handlers inherit `KeepAlive::No`, so a freshly opened
    /// link is closed with KeepAliveTimeout before the gossip mesh forms and
    /// no block ever propagates. Ping holds the connection open.
    ping: libp2p::ping::Behaviour,
    identify: libp2p::identify::Behaviour,
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
    peers: Arc<AtomicUsize>,
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
            .mesh_outbound_min(0) // inbound-only peers must be allowed in the mesh
            .mesh_n(6)
            // A handful of testnet nodes cannot fill a gossip mesh. Relay every
            // message to every subscribed peer so blocks/txs always propagate.
            .flood_publish(true)
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

        let ping = libp2p::ping::Behaviour::new(libp2p::ping::Config::new());
        let identify = libp2p::identify::Behaviour::new(libp2p::identify::Config::new(
            "/verinoo/1.0.0".into(),
            key.public(),
        ));
        let mdns = mdns::tokio::Behaviour::new(mdns::Config::default(), peer_id)?;

        let mut swarm = libp2p::SwarmBuilder::with_existing_identity(key.clone())
            .with_tokio()
            .with_tcp(
                tcp::Config::default(),
                noise::Config::new,
                yamux::Config::default,
            )?
            .with_behaviour(|_key| Behaviour {
                gossipsub,
                keepalive: keepalive::KeepAlive,
                ping,
                identify,
                mdns,
            })?
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
        let peer_count = Arc::new(AtomicUsize::new(0));
        let peer_count_task = peer_count.clone();

        // Bridge: outbound publishes + inbound events pumped in one task.
        let blocks_topic_str = blocks_topic.to_string();
        let txs_topic_str = txs_topic.to_string();
        let seed_addrs: Vec<Multiaddr> = seed_peers.iter().filter_map(|p| p.parse().ok()).collect();
        tokio::spawn(async move {
            let mut peers: HashSet<PeerId> = HashSet::new();
            // A node that starts a moment before its seed is listening would
            // otherwise never connect: the initial dial fails once and is not
            // retried. Keep dialing until at least one peer is up.
            let mut redial = tokio::time::interval(Duration::from_secs(5));
            redial.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                tokio::select! {
                    _ = redial.tick() => {
                        if peers.is_empty() {
                            for addr in &seed_addrs {
                                tracing::debug!("dialing seed {addr}");
                                let _ = swarm.dial(addr.clone());
                            }
                        }
                    }
                    cmd = cmd_rx.recv() => {
                        match cmd {
                            Some(Command::PublishBlock(payload)) => {
                                if let Err(e) = swarm.behaviour_mut().gossipsub
                                    .publish(gossipsub::IdentTopic::new(blocks_topic_str.clone()), payload.as_bytes())
                                {
                                    // A single-node testnet has no mesh peers;
                                    // that is expected, not a fault.
                                    if matches!(e, gossipsub::PublishError::InsufficientPeers) {
                                        tracing::debug!("block not gossiped: no peers yet");
                                    } else {
                                        tracing::warn!("publish block failed: {e:?}");
                                    }
                                }
                            }
                            Some(Command::PublishTx(payload)) => {
                                if let Err(e) = swarm.behaviour_mut().gossipsub
                                    .publish(gossipsub::IdentTopic::new(txs_topic_str.clone()), payload.as_bytes())
                                {
                                    if matches!(e, gossipsub::PublishError::InsufficientPeers) {
                                        tracing::debug!("tx not gossiped: no peers yet");
                                    } else {
                                        tracing::warn!("publish tx failed: {e:?}");
                                    }
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
                            // mDNS only *reports* peers; without dialing on the
                            // discovery event two nodes on the same host never
                            // connect and gossipsub stays silent.
                            libp2p::swarm::SwarmEvent::Behaviour(
                                BehaviourEvent::Mdns(mdns::Event::Discovered(list)),
                            ) => {
                                for (peer_id, addr) in list {
                                    tracing::debug!("mdns discovered {peer_id} at {addr}");
                                    if !peers.contains(&peer_id) {
                                        if let Err(e) = swarm.dial(addr) {
                                            tracing::debug!("mdns dial failed: {e:?}");
                                        }
                                    }
                                }
                            }
                            libp2p::swarm::SwarmEvent::ConnectionEstablished {
                                peer_id, ..
                            } => {
                                if peers.insert(peer_id) {
                                    peer_count_task.store(peers.len(), Ordering::Relaxed);
                                    tracing::info!("peer connected: {peer_id}");
                                }
                            }
                            libp2p::swarm::SwarmEvent::ConnectionClosed {
                                peer_id,
                                cause,
                                ..
                            } => {
                                tracing::info!("peer disconnected: {peer_id} ({cause:?})");
                                peers.remove(&peer_id);
                                peer_count_task.store(peers.len(), Ordering::Relaxed);
                            }
                            libp2p::swarm::SwarmEvent::OutgoingConnectionError {
                                error, ..
                            } => {
                                tracing::debug!("outgoing connection failed: {error}");
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
            peers: peer_count,
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

    /// Currently connected peers. A validator uses this to avoid producing
    /// blocks that no other node will ever see.
    pub fn peer_count(&self) -> usize {
        self.peers.load(Ordering::Relaxed)
    }
}
