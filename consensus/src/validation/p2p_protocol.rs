//! P2P message propagation for validator coordination
//! 
//! This module implements a gossip protocol for propagating validation messages
//! across the network, ensuring efficient and reliable validator coordination.

use kaspa_hashes::Hash;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

use super::network::{ValidationMessage, NetworkEvent, NetworkError};
use super::bls_signatures::AggregatedSignature;

/// P2P message with metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct P2PMessage {
    pub message: ValidationMessage,
    pub sender: String,
    pub timestamp: u64,
    pub signature: Vec<u8>, // Sender's signature for authentication
}

/// P2P peer information
#[derive(Clone, Debug)]
pub struct PeerInfo {
    pub address: String,
    pub public_key: Vec<u8>,
    pub last_seen: Instant,
    pub is_validator: bool,
}

/// Gossip protocol configuration
#[derive(Clone, Debug)]
pub struct GossipConfig {
    pub fanout: usize, // Number of peers to gossip to
    pub interval_ms: u64, // Gossip interval
    pub max_message_age_ms: u64, // Maximum message age
    pub max_peers: usize, // Maximum number of peers
}

impl Default for GossipConfig {
    fn default() -> Self {
        Self {
            fanout: 8,
            interval_ms: 1000,
            max_message_age_ms: 60000, // 1 minute
            max_peers: 100,
        }
    }
}

/// P2P gossip protocol for validator coordination
pub struct P2PProtocol {
    local_address: String,
    peers: Arc<Mutex<HashMap<String, PeerInfo>>>,
    seen_messages: Arc<Mutex<HashSet<Vec<u8>>>>, // Message IDs to prevent duplicates
    message_queue: Arc<Mutex<Vec<P2PMessage>>>,
    config: GossipConfig,
    sender: mpsc::UnboundedSender<P2PMessage>,
}

impl P2PProtocol {
    /// Create a new P2P protocol instance
    pub fn new(local_address: String, config: GossipConfig) -> (Self, mpsc::UnboundedReceiver<P2PMessage>) {
        let (sender, receiver) = mpsc::unbounded_channel();
        
        let protocol = Self {
            local_address,
            peers: Arc::new(Mutex::new(HashMap::new())),
            seen_messages: Arc::new(Mutex::new(HashSet::new())),
            message_queue: Arc::new(Mutex::new(Vec::new())),
            config,
            sender,
        };
        
        (protocol, receiver)
    }

    /// Add a peer to the network
    pub fn add_peer(&self, peer: PeerInfo) {
        let mut peers = self.peers.lock().unwrap();
        
        // Check if we're at max capacity
        if peers.len() >= self.config.max_peers {
            // Remove oldest peer
            if let Some(oldest) = peers.iter().min_by_key(|(_, info)| info.last_seen) {
                let addr = oldest.0.clone();
                peers.remove(&addr);
            }
        }
        
        peers.insert(peer.address.clone(), peer);
    }

    /// Remove a peer from the network
    pub fn remove_peer(&self, address: &str) {
        let mut peers = self.peers.lock().unwrap();
        peers.remove(address);
    }

    /// Get all peers
    pub fn get_peers(&self) -> Vec<PeerInfo> {
        let peers = self.peers.lock().unwrap();
        peers.values().cloned().collect()
    }

    /// Get validator peers only
    pub fn get_validator_peers(&self) -> Vec<PeerInfo> {
        let peers = self.peers.lock().unwrap();
        peers.values()
            .filter(|info| info.is_validator)
            .cloned()
            .collect()
    }

    /// Broadcast a message to all peers
    pub fn broadcast(&self, message: ValidationMessage, signature: Vec<u8>) -> Result<(), NetworkError> {
        let p2p_message = P2PMessage {
            message,
            sender: self.local_address.clone(),
            timestamp: Instant::now().elapsed().as_millis() as u64,
            signature,
        };

        // Add to message queue for gossip
        let mut queue = self.message_queue.lock().unwrap();
        queue.push(p2p_message);

        Ok(())
    }

    /// Send a message to a specific peer
    pub fn send_to_peer(&self, peer_address: &str, message: ValidationMessage, signature: Vec<u8>) -> Result<(), NetworkError> {
        let p2p_message = P2PMessage {
            message,
            sender: self.local_address.clone(),
            timestamp: Instant::now().elapsed().as_millis() as u64,
            signature,
        };

        // In production, this would send over the actual network
        // For now, we'll add to the sender channel
        self.sender.send(p2p_message).map_err(|_| NetworkError::NetworkError("Failed to send message".to_string()))?;

        Ok(())
    }

    /// Process an incoming message
    pub fn process_message(&self, p2p_message: P2PMessage) -> Result<NetworkEvent, NetworkError> {
        // Check for duplicate messages
        let message_id = self.compute_message_id(&p2p_message);
        let mut seen = self.seen_messages.lock().unwrap();
        
        if seen.contains(&message_id) {
            return Err(NetworkError::InvalidMessage); // Duplicate message
        }
        
        seen.insert(message_id);
        drop(seen);

        // Check message age
        let now = Instant::now().elapsed().as_millis() as u64;
        if now - p2p_message.timestamp > self.config.max_message_age_ms {
            return Err(NetworkError::InvalidMessage); // Stale message
        }

        // Update peer last seen
        let mut peers = self.peers.lock().unwrap();
        if let Some(peer) = peers.get_mut(&p2p_message.sender) {
            peer.last_seen = Instant::now();
        }
        drop(peers);

        // Process the validation message
        // This would normally call the network's process_message
        // For now, we'll return a placeholder event
        Ok(NetworkEvent::NotSelectedForValidation)
    }

    /// Gossip pending messages to peers
    pub fn gossip(&self) -> Result<usize, NetworkError> {
        let mut queue = self.message_queue.lock().unwrap();
        let messages = std::mem::take(&mut *queue);
        drop(queue);

        if messages.is_empty() {
            return Ok(0);
        }

        let peers = self.get_validator_peers();
        let fanout = self.config.fanout.min(peers.len());
        
        if fanout == 0 {
            return Ok(0);
        }

        let mut sent_count = 0;

        for message in messages {
            // Select random peers to gossip to
            let selected_peers: Vec<_> = peers
                .iter()
                .take(fanout)
                .collect();

            for peer in selected_peers {
                let _ = self.send_to_peer(&peer.address, message.message.clone(), message.signature.clone());
                sent_count += 1;
            }
        }

        Ok(sent_count)
    }

    /// Start the gossip protocol background task
    pub async fn start_gossip_task(&self) {
        let interval = Duration::from_millis(self.config.interval_ms);
        
        loop {
            tokio::time::sleep(interval).await;
            
            if let Err(e) = self.gossip() {
                eprintln!("Gossip error: {}", e);
            }
            
            // Clean up stale peers
            self.cleanup_stale_peers();
        }
    }

    /// Clean up peers that haven't been seen recently
    fn cleanup_stale_peers(&self) {
        let mut peers = self.peers.lock().unwrap();
        let timeout = Duration::from_secs(300); // 5 minutes
        
        peers.retain(|_, info| info.last_seen.elapsed() < timeout);
    }

    /// Compute a unique message ID for deduplication
    fn compute_message_id(&self, message: &P2PMessage) -> Vec<u8> {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        
        // Hash the message content
        if let Ok(serialized) = bincode::serialize(&message.message) {
            hasher.write(&serialized);
        }
        
        hasher.write(message.sender.as_bytes());
        hasher.write_u64(message.timestamp);
        
        format!("{:x}", hasher.finish()).into_bytes()
    }

    /// Get network statistics
    pub fn get_stats(&self) -> P2PStats {
        let peers = self.peers.lock().unwrap();
        let seen = self.seen_messages.lock().unwrap();
        let queue = self.message_queue.lock().unwrap();
        
        P2PStats {
            peer_count: peers.len(),
            validator_count: peers.values().filter(|p| p.is_validator).count(),
            seen_message_count: seen.len(),
            queued_message_count: queue.len(),
        }
    }
}

/// P2P network statistics
#[derive(Clone, Debug)]
pub struct P2PStats {
    pub peer_count: usize,
    pub validator_count: usize,
    pub seen_message_count: usize,
    pub queued_message_count: usize,
}

/// P2P protocol manager for coordinating multiple instances
pub struct P2PManager {
    protocols: HashMap<String, Arc<P2PProtocol>>,
}

impl P2PManager {
    /// Create a new P2P manager
    pub fn new() -> Self {
        Self {
            protocols: HashMap::new(),
        }
    }

    /// Register a protocol instance
    pub fn register(&mut self, address: String, protocol: Arc<P2PProtocol>) {
        self.protocols.insert(address, protocol);
    }

    /// Get a protocol by address
    pub fn get(&self, address: &str) -> Option<Arc<P2PProtocol>> {
        self.protocols.get(address).cloned()
    }

    /// Broadcast to all protocols
    pub fn broadcast_all(&self, message: ValidationMessage, signature: Vec<u8>) -> Result<(), NetworkError> {
        for protocol in self.protocols.values() {
            protocol.broadcast(message.clone(), signature.clone())?;
        }
        Ok(())
    }
}

impl Default for P2PManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_p2p_protocol_creation() {
        let config = GossipConfig::default();
        let (protocol, _receiver) = P2PProtocol::new("node1".to_string(), config);
        
        assert_eq!(protocol.get_peers().len(), 0);
    }

    #[test]
    fn test_peer_management() {
        let config = GossipConfig::default();
        let (protocol, _receiver) = P2PProtocol::new("node1".to_string(), config);
        
        let peer = PeerInfo {
            address: "node2".to_string(),
            public_key: vec![1; 32],
            last_seen: Instant::now(),
            is_validator: true,
        };
        
        protocol.add_peer(peer);
        assert_eq!(protocol.get_peers().len(), 1);
        
        protocol.remove_peer("node2");
        assert_eq!(protocol.get_peers().len(), 0);
    }

    #[test]
    fn test_validator_peer_filtering() {
        let config = GossipConfig::default();
        let (protocol, _receiver) = P2PProtocol::new("node1".to_string(), config);
        
        protocol.add_peer(PeerInfo {
            address: "validator1".to_string(),
            public_key: vec![1; 32],
            last_seen: Instant::now(),
            is_validator: true,
        });
        
        protocol.add_peer(PeerInfo {
            address: "non_validator".to_string(),
            public_key: vec![2; 32],
            last_seen: Instant::now(),
            is_validator: false,
        });
        
        let validators = protocol.get_validator_peers();
        assert_eq!(validators.len(), 1);
        assert_eq!(validators[0].address, "validator1");
    }

    #[test]
    fn test_duplicate_message_detection() {
        let config = GossipConfig::default();
        let (protocol, _receiver) = P2PProtocol::new("node1".to_string(), config);
        
        let message = P2PMessage {
            message: ValidationMessage::SelectionChallenge {
                block_hash: Hash::from_bytes([1u8; 32]),
                challenge_data: vec![1, 2, 3],
            },
            sender: "node2".to_string(),
            timestamp: Instant::now().elapsed().as_millis() as u64,
            signature: vec![1, 2, 3],
        };
        
        // First message should succeed
        assert!(protocol.process_message(message.clone()).is_ok());
        
        // Duplicate should fail
        assert!(matches!(protocol.process_message(message), Err(NetworkError::InvalidMessage)));
    }

    #[test]
    fn test_stale_message_rejection() {
        let config = GossipConfig {
            max_message_age_ms: 1000,
            ..Default::default()
        };
        let (protocol, _receiver) = P2PProtocol::new("node1".to_string(), config);
        
        let mut message = P2PMessage {
            message: ValidationMessage::SelectionChallenge {
                block_hash: Hash::from_bytes([1u8; 32]),
                challenge_data: vec![1, 2, 3],
            },
            sender: "node2".to_string(),
            timestamp: Instant::now().elapsed().as_millis() as u64 - 2000, // 2 seconds ago
            signature: vec![1, 2, 3],
        };
        
        assert!(matches!(protocol.process_message(message), Err(NetworkError::InvalidMessage)));
    }

    #[test]
    fn test_p2p_stats() {
        let config = GossipConfig::default();
        let (protocol, _receiver) = P2PProtocol::new("node1".to_string(), config);
        
        protocol.add_peer(PeerInfo {
            address: "validator1".to_string(),
            public_key: vec![1; 32],
            last_seen: Instant::now(),
            is_validator: true,
        });
        
        let stats = protocol.get_stats();
        assert_eq!(stats.peer_count, 1);
        assert_eq!(stats.validator_count, 1);
    }

    #[test]
    fn test_p2p_manager() {
        let mut manager = P2PManager::new();
        let config = GossipConfig::default();
        let (protocol, _receiver) = P2PProtocol::new("node1".to_string(), config);
        
        manager.register("node1".to_string(), Arc::new(protocol));
        
        assert!(manager.get("node1").is_some());
        assert!(manager.get("node2").is_none());
    }
}
