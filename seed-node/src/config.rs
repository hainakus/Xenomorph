/// Configuration for genome data sources used by the seed-node.
#[derive(Debug, Clone, PartialEq)]
pub struct GenomeSource {
    pub merkle_root: [u8; 32],
    pub uri: String,
    pub label: Option<String>,
}

/// Seed-node runtime configuration.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SeedNodeConfig {
    pub genome_default: Option<String>,
    pub genome_sources: Vec<GenomeSource>,
}
