use crate::{flow_context::FlowContext, flow_trait::Flow};
use kaspa_core::{debug, trace, warn};
use kaspa_p2p_lib::{
    common::ProtocolError,
    make_message, make_response,
    pb::{kaspad_message::Payload, CheckpointAnnouncementMessage, NetAddress},
    IncomingRoute, Router,
};
use model_crypto::gossip::{verify_announcement, Announcement};
use std::{net::SocketAddr, sync::Arc};

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

/// Receives checkpoint gossip announcements and answers checkpoint requests.
pub struct ModelGossipFlow {
    ctx: FlowContext,
    router: Arc<Router>,
    incoming_route: IncomingRoute,
}

#[async_trait::async_trait]
impl Flow for ModelGossipFlow {
    fn router(&self) -> Option<Arc<Router>> {
        Some(self.router.clone())
    }

    async fn start(&mut self) -> Result<(), ProtocolError> {
        self.start_impl().await
    }
}

impl ModelGossipFlow {
    pub fn new(ctx: FlowContext, router: Arc<Router>, incoming_route: IncomingRoute) -> Self {
        Self { ctx, router, incoming_route }
    }

    async fn start_impl(&mut self) -> Result<(), ProtocolError> {
        let network_type = self.ctx.config.net.network_type;

        while let Some(msg) = self.incoming_route.recv().await {
            match msg.payload {
                Some(Payload::CheckpointAnnouncement(announcement_msg)) => {
                    if let Some(announcement) = proto_to_announcement(announcement_msg) {
                        trace!("ModelGossipFlow: received checkpoint announcement for {}", announcement.model_id);
                        if verify_announcement(&announcement, network_type).is_ok() {
                            self.ctx.gossip_registry.lock().insert(announcement);
                        } else {
                            warn!("ModelGossipFlow: invalid checkpoint announcement signature");
                        }
                    }
                }
                Some(Payload::RequestCheckpoint(request)) => {
                    trace!("ModelGossipFlow: received checkpoint request for {}", request.model_id);
                    let weights_hash = request.weights_hash.try_into().ok();
                    let response = if let Some(weights_hash) = weights_hash {
                        self.ctx.gossip_registry.lock().get(&weights_hash)
                    } else {
                        Vec::new()
                    };

                    let request_id = msg.request_id;
                    for ann in response {
                        let proto = announcement_to_proto(&ann);
                        let response_msg = make_response!(Payload::CheckpointAnnouncement, proto, request_id);
                        if let Err(e) = self.router.enqueue(response_msg).await {
                            debug!("ModelGossipFlow: failed to send checkpoint response: {e}");
                            break;
                        }
                    }
                }
                _ => {
                    warn!("ModelGossipFlow: unexpected message payload");
                    break;
                }
            }
        }

        Ok(())
    }
}

impl FlowContext {
    /// Broadcast a signed checkpoint announcement to all connected peers.
    pub async fn broadcast_checkpoint_announcement(&self, announcement: &Announcement) -> Result<(), ProtocolError> {
        let proto = announcement_to_proto(announcement);
        let msg = make_message!(Payload::CheckpointAnnouncement, proto);
        self.hub().broadcast(msg).await;
        Ok(())
    }

    /// Sign and broadcast an announcement using the node's gossip identity.
    pub async fn announce_checkpoint(&self, model_id: String, weights_hash: [u8; 32], cid: [u8; 32], listen_addr: Option<SocketAddr>) {
        let identity = match self.gossip_identity.as_ref() {
            Some(id) => id,
            None => {
                trace!("announce_checkpoint: no gossip identity configured");
                return;
            }
        };

        let announcement = Announcement {
            model_id,
            weights_hash,
            cid,
            timestamp: kaspa_core::time::unix_now(),
            is_genome: false,
            node_address: String::new(),
            public_key: [0u8; 33],
            listen_addr,
            signature: [0u8; 64],
        };

        match identity.sign(announcement) {
            Ok(signed) => {
                if let Err(e) = self.broadcast_checkpoint_announcement(&signed).await {
                    warn!("announce_checkpoint: broadcast failed: {e}");
                }
            }
            Err(e) => {
                warn!("announce_checkpoint: signing failed: {e}");
            }
        }
    }
}
