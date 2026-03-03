//! `genome-freeze` — converts GRCh38 FASTA files into a packed `.xenom` file.
//!
//! # Usage
//! ```text
//! genome-freeze --fasta chr1.fa chr2.fa ... --output grch38.xenom [--fragment-size 1048576] [--dataset-version 0]
//! ```
//!
//! # What it does
//! 1. Reads all FASTA files in order (chr1..22, X, Y recommended order for determinism)
//! 2. Strips `>header` lines and whitespace; uppercases all bases; N/ambiguous → A
//! 3. 2-bit packs: A=0b00, C=0b01, G=0b10, T=0b11 — 4 bases/byte, MSB-first
//! 4. Computes blake3 Merkle root over unpacked fragment leaves
//! 5. Writes `grch38.xenom` = 64-byte header + packed genome bytes
//!
//! The output file is deterministic given the same input order and content.
//! Its Merkle root must match `genome_merkle_root` in `Params` for nodes to
//! accept it.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use kaspa_hashes::Hash;
use kaspa_pow::genome_file::{pack_2bit, unpack_2bit, GenomeFileHeader, GENOME_FILE_FORMAT_VERSION};
use kaspa_pow::genome_pow::{build_merkle_root, fragment_leaf_hash, GENOME_BASE_SIZE};

#[derive(Parser)]
#[command(name = "genome-freeze", about = "Convert GRCh38 FASTA → packed .xenom genome dataset")]
struct Args {
    /// Input FASTA files (process in order: chr1..22, X, Y for determinism)
    #[arg(long, required = true, num_args = 1..)]
    fasta: Vec<PathBuf>,

    /// Output .xenom file path
    #[arg(long, default_value = "grch38.xenom")]
    output: PathBuf,

    /// Fragment size in unpacked bytes (must be divisible by 4)
    #[arg(long, default_value_t = 1_048_576)]
    fragment_size: u32,

    /// Dataset version (increment on hard-fork dataset change)
    #[arg(long, default_value_t = 0u32)]
    dataset_version: u32,

    /// Skip Merkle root computation (faster, for testing only)
    #[arg(long, default_value_t = false)]
    skip_merkle: bool,
}

fn main() {
    let args = Args::parse();

    if args.fragment_size % 4 != 0 {
        eprintln!("ERROR: --fragment-size must be divisible by 4");
        std::process::exit(1);
    }

    println!("genome-freeze v{}", env!("CARGO_PKG_VERSION"));
    println!("Fragment size: {} bytes unpacked ({} bytes packed)", args.fragment_size, args.fragment_size / 4);

    let t0 = Instant::now();

    // ── Phase 1: read, canonicalise, pack ─────────────────────────────────────
    println!("\nPhase 1: reading and 2-bit encoding FASTA files ...");

    let mut total_bases: u64 = 0;
    let mut packed_genome: Vec<u8> = Vec::with_capacity(GENOME_BASE_SIZE as usize / 4);
    let mut leftover_bases: Vec<u8> = Vec::with_capacity(4); // bases not yet packed (< 4)

    for fasta_path in &args.fasta {
        println!("  Reading {:?} ...", fasta_path);
        let file = match File::open(fasta_path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("ERROR: cannot open {:?}: {e}", fasta_path);
                std::process::exit(1);
            }
        };
        let reader = BufReader::with_capacity(8 * 1024 * 1024, file);

        for line in reader.lines() {
            let line = line.expect("I/O error reading FASTA");
            let trimmed = line.trim_end();
            if trimmed.is_empty() || trimmed.starts_with('>') || trimmed.starts_with(';') {
                continue; // skip FASTA headers and blank lines
            }
            for &b in trimmed.as_bytes() {
                leftover_bases.push(b);
                total_bases += 1;
                // flush once we have 4 bases
                if leftover_bases.len() == 4 {
                    let packed = pack_2bit(&leftover_bases);
                    packed_genome.extend_from_slice(&packed);
                    leftover_bases.clear();
                }
            }
        }
    }

    // flush remaining < 4 bases (pad with A)
    if !leftover_bases.is_empty() {
        let packed = pack_2bit(&leftover_bases); // pads to multiple of 4 internally
        packed_genome.extend_from_slice(&packed);
    }

    let total_packed_bytes = packed_genome.len() as u64;
    println!(
        "  Total bases: {}  packed bytes: {}  ({:.1} MB)  [{:.1}s]",
        total_bases,
        total_packed_bytes,
        total_packed_bytes as f64 / 1_048_576.0,
        t0.elapsed().as_secs_f64()
    );

    // ── Phase 2: Merkle root ───────────────────────────────────────────────────
    let merkle_root: [u8; 32];

    if args.skip_merkle {
        println!("\nPhase 2: skipping Merkle root (--skip-merkle)");
        merkle_root = [0u8; 32];
    } else {
        println!("\nPhase 2: computing Merkle root over {} fragments ...", num_fragments(total_packed_bytes, args.fragment_size));
        let t2 = Instant::now();

        let packed_frag = (args.fragment_size / 4) as usize;
        let nf = num_fragments(total_packed_bytes, args.fragment_size) as usize;
        let leaves: Vec<Hash> = (0..nf)
            .map(|idx| {
                let offset = idx * packed_frag;
                let end = (offset + packed_frag).min(packed_genome.len());
                let packed_slice = &packed_genome[offset..end];
                let mut unpacked = unpack_2bit(packed_slice);
                unpacked.truncate(args.fragment_size as usize);
                fragment_leaf_hash(idx as u64, &unpacked)
            })
            .collect();

        let root = build_merkle_root(&leaves);
        let root_slice: &[u8] = root.as_ref();
        merkle_root = root_slice.try_into().unwrap();

        println!(
            "  Merkle root: {}  [{:.1}s]",
            merkle_root.iter().map(|b| format!("{b:02x}")).collect::<String>(),
            t2.elapsed().as_secs_f64()
        );
    }

    // ── Phase 3: write .xenom file ────────────────────────────────────────────
    println!("\nPhase 3: writing {:?} ...", args.output);
    let t3 = Instant::now();

    let header = GenomeFileHeader {
        version: GENOME_FILE_FORMAT_VERSION,
        dataset_version: args.dataset_version,
        total_bases,
        total_packed_bytes,
        merkle_root,
    };

    let out_file = OpenOptions::new().write(true).create(true).truncate(true).open(&args.output).unwrap_or_else(|e| {
        eprintln!("ERROR: cannot create {:?}: {e}", args.output);
        std::process::exit(1);
    });
    let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, out_file);
    writer.write_all(&header.to_bytes()).expect("write header");
    writer.write_all(&packed_genome).expect("write genome data");
    writer.flush().expect("flush");

    println!(
        "  Written {:.1} MB in {:.1}s",
        (64 + total_packed_bytes) as f64 / 1_048_576.0,
        t3.elapsed().as_secs_f64()
    );

    // ── Summary ───────────────────────────────────────────────────────────────
    println!("\n✓ Done in {:.1}s", t0.elapsed().as_secs_f64());
    println!("  Output:          {:?}", args.output);
    println!("  Total bases:     {}", total_bases);
    println!("  Packed bytes:    {}", total_packed_bytes);
    println!("  Fragments:       {} × {} unpacked bytes", num_fragments(total_packed_bytes, args.fragment_size), args.fragment_size);
    println!("  Dataset version: {}", args.dataset_version);
    if !args.skip_merkle {
        println!(
            "  Merkle root:     {}",
            merkle_root.iter().map(|b| format!("{b:02x}")).collect::<String>()
        );
        println!("\n  Add to consensus Params:");
        println!("    genome_merkle_root: \"{}\"", merkle_root.iter().map(|b| format!("{b:02x}")).collect::<String>());
    }
}

fn num_fragments(total_packed_bytes: u64, fragment_size_bytes: u32) -> u64 {
    let packed_frag = (fragment_size_bytes / 4) as u64;
    if packed_frag == 0 {
        0
    } else {
        total_packed_bytes / packed_frag
    }
}
