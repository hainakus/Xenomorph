pub mod archive;
pub mod batch_generator;
pub mod downloader;
pub mod storage;

#[cfg(test)]
pub mod tests;

pub use archive::{
    base_from_bits, packed_bytes_for, GenomeArchive, XenomHeader, DEFAULT_FRAGMENT_SIZE,
    GENOME_FILE_HEADER_SIZE, GENOME_FILE_MAGIC,
};
pub use batch_generator::{GenomeBatchGenerator, GenomeSlice, GenomeTrainingBatch, pack_sequence};
pub use downloader::GenomeDownloader;
pub use storage::GenomeStorage;
