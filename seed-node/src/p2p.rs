//! Minimal Kaspa P2P gossip client for the seed-node.
//!
//! This lets the seed-node join the Xenomorph P2P network as a peer that only
//! speaks checkpoint gossip: it announces the checkpoints it holds and answers
//! `RequestCheckpoint` messages.  It does not participate in consensus or block
//! relay.

use anyhow::{Context, Result};
use kaspa_consensus_core::network::{NetworkId, NetworkType};
use kaspa_core::time::unix_now;
use kaspa_p2p_lib::{
    common::ProtocolError,
    make_message, make_response,
    pb::{kaspad_message::Payload, AddressesMessage, CheckpointAnnouncementMessage, NetAddress, PongMessage, VersionMessage},
    Adaptor, ConnectionInitializer, Hub, KaspadHandshake, KaspadMessagePayloadType, Router,
};
use kaspa_utils_tower::counters::TowerConnectionCounters;
use model_crypto::gossip::{verify_announcement, Announcement, GossipIdentity, GossipRegistry};
use std::{net::SocketAddr, path::Path, sync::Arc, time::Duration};
use tokio::sync::Mutex;
use tracing::{debug, info, trace, warn};
use uuid::Uuid;

const MAX_CONNECT_RETRIES: u8 = 60;
const CONNECT_RETRY_INTERVAL: Duration = Duration::from_secs(2);

/// Handle to the seed-node P2P gossip client.
pub struct P2pGossipHandle {
    inner: Arc<P2pGossipInner>,
    hub: Hub,
    adaptor: Arc<Adaptor>,
    peer_address: String,
}

struct P2pGossipInner {
    identity: GossipIdentity,
    registry: Mutex<GossipRegistry>,
    network_type: NetworkType,
}

/// Per-connection initializer: handshakes, subscribes and spawns the receive loop.
struct GossipInitializer {
    inner: Arc<P2pGossipInner>,
}

impl P2pGossipHandle {
    pub async fn connect<P: AsRef<Path>>(peer_address: String, network_type: NetworkType, key_path: P) -> Result<Arc<Self>> {
        let identity =
            GossipIdentity::load_or_generate(key_path, network_type).context("Failed to load or generate gossip identity")?;
        info!("P2P gossip identity: {}", identity.address());

        let inner = Arc::new(P2pGossipInner { identity, registry: Mutex::new(GossipRegistry::new()), network_type });
        let hub = Hub::new();
        let initializer: Arc<dyn ConnectionInitializer> = Arc::new(GossipInitializer { inner: inner.clone() });
        let counters = Arc::new(TowerConnectionCounters::default());
        let adaptor = Adaptor::client_only(hub.clone(), initializer, counters);

        let handle = Arc::new(Self { inner: inner.clone(), hub: hub.clone(), adaptor: adaptor.clone(), peer_address });

        // Block startup while trying to connect.  This gives xeno-node time to finish
        // its own P2P startup without failing immediately.
        handle.connect_with_backoff().await?;

        Ok(handle)
    }

    async fn connect_with_backoff(&self) -> Result<()> {
        for attempt in 1..=MAX_CONNECT_RETRIES {
            info!("P2P gossip: connecting to {} (attempt {}/{})", self.peer_address, attempt, MAX_CONNECT_RETRIES);
            match self.adaptor.connect_peer(self.peer_address.clone()).await {
                Ok(_) => {
                    info!("P2P gossip: connected to {}", self.peer_address);
                    return Ok(());
                }
                Err(e) => {
                    warn!("P2P gossip: connection attempt {} to {} failed: {}", attempt, self.peer_address, e);
                    if attempt < MAX_CONNECT_RETRIES {
                        tokio::time::sleep(CONNECT_RETRY_INTERVAL).await;
                    }
                }
            }
        }
        Err(anyhow::anyhow!("failed to connect to {} after {} attempts", self.peer_address, MAX_CONNECT_RETRIES))
    }

    /// Sign and broadcast a checkpoint announcement.
    pub async fn announce(&self, model_id: String, weights_hash: [u8; 32], cid: [u8; 32], listen_addr: Option<SocketAddr>) {
        let announcement = Announcement {
            model_id,
            weights_hash,
            cid,
            timestamp: unix_now(),
            is_genome: false,
            node_address: String::new(),
            public_key: [0u8; 33],
            listen_addr,
            signature: [0u8; 64],
        };

        match self.inner.identity.sign(announcement) {
            Ok(signed) => {
                let proto = announcement_to_proto(&signed);
                let msg = make_message!(Payload::CheckpointAnnouncement, proto);
                self.hub.broadcast(msg).await;
                trace!("P2P gossip: announced checkpoint for {}", signed.model_id);
            }
            Err(e) => {
                warn!("P2P gossip: failed to sign announcement: {e}");
            }
        }
    }

    /// Get known announcements for a checkpoint from the local registry.
    pub async fn known_peers(&self, weights_hash: &[u8; 32]) -> Vec<Announcement> {
        self.inner.registry.lock().await.get(weights_hash)
    }
}

#[tonic::async_trait]
impl ConnectionInitializer for GossipInitializer {
    async fn initialize_connection(&self, router: Arc<Router>) -> Result<(), ProtocolError> {
        let inner = self.inner.clone();

        // Build the local version message
        let network_name = NetworkId::new(inner.network_type).to_prefixed();
        let version_message = VersionMessage {
            protocol_version: 6,
            services: 0,
            timestamp: unix_now() as i64,
            address: None,
            id: Uuid::new_v4().as_bytes().to_vec(),
            user_agent: "xenom-seed-gossip/0.1".to_string(),
            disable_relay_tx: true,
            subnetwork_id: None,
            network: network_name,
            dns_seeder: String::new(),
            hashing_algo_version: "PyrinHashv2".to_string(),
        };

        router.start();

        let mut handshake = KaspadHandshake::new(&router);
        handshake.handshake(version_message).await?;
        handshake.exchange_ready_messages().await?;

        // Subscribe to all message types so the router never logs "no flow registered".
        // Gossip and peer-maintenance messages are handled explicitly; everything else is dropped.
        let mut incoming_route = router.subscribe(vec![
            KaspadMessagePayloadType::Addresses,
            KaspadMessagePayloadType::Block,
            KaspadMessagePayloadType::Transaction,
            KaspadMessagePayloadType::BlockLocator,
            KaspadMessagePayloadType::RequestAddresses,
            KaspadMessagePayloadType::RequestRelayBlocks,
            KaspadMessagePayloadType::RequestTransactions,
            KaspadMessagePayloadType::IbdBlock,
            KaspadMessagePayloadType::InvRelayBlock,
            KaspadMessagePayloadType::InvTransactions,
            KaspadMessagePayloadType::Ping,
            KaspadMessagePayloadType::Pong,
            KaspadMessagePayloadType::TransactionNotFound,
            KaspadMessagePayloadType::Reject,
            KaspadMessagePayloadType::PruningPointUtxoSetChunk,
            KaspadMessagePayloadType::RequestIbdBlocks,
            KaspadMessagePayloadType::UnexpectedPruningPoint,
            KaspadMessagePayloadType::IbdBlockLocator,
            KaspadMessagePayloadType::IbdBlockLocatorHighestHash,
            KaspadMessagePayloadType::RequestNextPruningPointUtxoSetChunk,
            KaspadMessagePayloadType::DonePruningPointUtxoSetChunks,
            KaspadMessagePayloadType::IbdBlockLocatorHighestHashNotFound,
            KaspadMessagePayloadType::BlockWithTrustedData,
            KaspadMessagePayloadType::DoneBlocksWithTrustedData,
            KaspadMessagePayloadType::RequestPruningPointAndItsAnticone,
            KaspadMessagePayloadType::BlockHeaders,
            KaspadMessagePayloadType::RequestNextHeaders,
            KaspadMessagePayloadType::DoneHeaders,
            KaspadMessagePayloadType::RequestPruningPointUtxoSet,
            KaspadMessagePayloadType::RequestHeaders,
            KaspadMessagePayloadType::RequestBlockLocator,
            KaspadMessagePayloadType::PruningPoints,
            KaspadMessagePayloadType::RequestPruningPointProof,
            KaspadMessagePayloadType::PruningPointProof,
            KaspadMessagePayloadType::BlockWithTrustedDataV4,
            KaspadMessagePayloadType::TrustedData,
            KaspadMessagePayloadType::RequestIbdChainBlockLocator,
            KaspadMessagePayloadType::IbdChainBlockLocator,
            KaspadMessagePayloadType::RequestAntipast,
            KaspadMessagePayloadType::RequestNextPruningPointAndItsAnticoneBlocks,
            KaspadMessagePayloadType::CheckpointAnnouncement,
            KaspadMessagePayloadType::RequestCheckpoint,
        ]);

        tokio::spawn(async move {
            debug!("P2P gossip: receive loop started");
            while let Some(msg) = incoming_route.recv().await {
                let request_id = msg.request_id;
                match msg.payload {
                    Some(Payload::CheckpointAnnouncement(announcement_msg)) => {
                        if let Some(announcement) = proto_to_announcement(announcement_msg) {
                            if verify_announcement(&announcement, inner.network_type).is_ok() {
                                inner.registry.lock().await.insert(announcement);
                            } else {
                                warn!("P2P gossip: invalid announcement signature");
                            }
                        }
                    }
                    Some(Payload::RequestCheckpoint(request)) => {
                        let weights_hash = request.weights_hash.try_into().ok();
                        let response = if let Some(weights_hash) = weights_hash {
                            inner.registry.lock().await.get(&weights_hash)
                        } else {
                            Vec::new()
                        };

                        for ann in response {
                            let proto = announcement_to_proto(&ann);
                            let response_msg = make_response!(Payload::CheckpointAnnouncement, proto, request_id);
                            if let Err(e) = router.enqueue(response_msg).await {
                                debug!("P2P gossip: failed to send response: {e}");
                                break;
                            }
                        }
                    }
                    Some(Payload::RequestAddresses(_)) => {
                        let response = make_response!(Payload::Addresses, AddressesMessage { address_list: vec![] }, request_id);
                        if let Err(e) = router.enqueue(response).await {
                            debug!("P2P gossip: failed to send addresses response: {e}");
                        }
                    }
                    Some(Payload::Ping(ping)) => {
                        let pong = make_message!(Payload::Pong, PongMessage { nonce: ping.nonce });
                        if let Err(e) = router.enqueue(pong).await {
                            debug!("P2P gossip: failed to send pong: {e}");
                        }
                    }
                    Some(Payload::Pong(_)) => {
                        // The seed-node does not send pings, so ignore pongs.
                    }
                    _ => {
                        // Other consensus/relay traffic is intentionally ignored by this client-only
                        // gossip peer.
                    }
                }
            }
            debug!("P2P gossip: receive loop exited");
        });

        Ok(())
    }
}

fn proto_to_announcement(msg: CheckpointAnnouncementMessage) -> Option<Announcement> {
    let weights_hash = msg.weights_hash.try_into().ok()?;
    let cid = msg.cid.try_into().ok()?;
    let public_key = msg.public_key.try_into().ok()?;
    let signature = msg.signature.try_into().ok()?;
    let listen_addr = msg.listen_addr.as_ref().and_then(net_address_to_socket);

    Some(Announcement {
        model_id: msg.model_id,
        weights_hash,
        cid,
        timestamp: msg.timestamp,
        is_genome: msg.is_genome,
        node_address: msg.node_address,
        public_key,
        listen_addr,
        signature,
    })
}

fn announcement_to_proto(ann: &Announcement) -> CheckpointAnnouncementMessage {
    CheckpointAnnouncementMessage {
        model_id: ann.model_id.clone(),
        weights_hash: ann.weights_hash.to_vec(),
        cid: ann.cid.to_vec(),
        timestamp: ann.timestamp,
        is_genome: ann.is_genome,
        node_address: ann.node_address.clone(),
        public_key: ann.public_key.to_vec(),
        listen_addr: ann.listen_addr.map(socket_to_net_address),
        signature: ann.signature.to_vec(),
    }
}

fn net_address_to_socket(addr: &NetAddress) -> Option<SocketAddr> {
    let ip = match addr.ip.len() {
        4 => {
            let mut octets = [0u8; 4];
            octets.copy_from_slice(&addr.ip);
            std::net::IpAddr::V4(std::net::Ipv4Addr::from(octets))
        }
        16 => {
            let mut octets = [0u8; 16];
            octets.copy_from_slice(&addr.ip);
            std::net::IpAddr::V6(std::net::Ipv6Addr::from(octets))
        }
        _ => return None,
    };
    Some(SocketAddr::new(ip, addr.port as u16))
}

fn socket_to_net_address(addr: SocketAddr) -> NetAddress {
    NetAddress {
        timestamp: 0,
        ip: match addr.ip() {
            std::net::IpAddr::V4(ip) => ip.octets().to_vec(),
            std::net::IpAddr::V6(ip) => ip.octets().to_vec(),
        },
        port: addr.port() as u32,
    }
}
