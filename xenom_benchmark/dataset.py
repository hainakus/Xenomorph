"""Genome dataset loaders and samplers for MLM benchmarking."""

import gzip
import random
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .utils import BASES, BASE_SET, DEFAULT_FRAGMENT_SIZE, logger


@dataclass
class SequenceWindow:
    """A contiguous DNA window extracted from a genome archive."""

    sequence: str
    fragment_idx: int = 0
    start: int = 0
    fragment_size: int = DEFAULT_FRAGMENT_SIZE


class XenomArchive:
    """In-memory parser for a XENOGEN1 2-bit packed genome archive.

    The archive layout matches ``seed-node/src/genome/archive.rs``: a 64-byte
    header followed by 2-bit packed DNA (A=00, C=01, G=10, T=11, MSB-first).
    """

    MAGIC = b"XENOGEN1"
    BITS_TO_BASE = {0: "A", 1: "C", 2: "G", 3: "T"}
    BASE_TO_BITS = {v: k for k, v in BITS_TO_BASE.items()}

    def __init__(self, path: Union[str, Path], fragment_size: int = DEFAULT_FRAGMENT_SIZE):
        self.path = Path(path)
        self.fragment_size = fragment_size

        with self.path.open("rb") as f:
            header = f.read(64)
            if len(header) != 64:
                raise ValueError(f"{self.path}: file too small for 64-byte header")

            self.magic = header[:8]
            if self.magic != self.MAGIC:
                raise ValueError(
                    f"{self.path}: invalid magic {self.magic!r}, expected {self.MAGIC!r}"
                )

            self.version = struct.unpack_from("<I", header, 8)[0]
            self.dataset_version = struct.unpack_from("<I", header, 12)[0]
            self.total_bases = struct.unpack_from("<Q", header, 16)[0]
            self.total_packed_bytes = struct.unpack_from("<Q", header, 24)[0]
            self.merkle_root = header[32:64].hex()

            self.data = f.read()
            if len(self.data) != self.total_packed_bytes:
                raise ValueError(
                    f"{self.path}: packed data size mismatch "
                    f"(expected {self.total_packed_bytes}, got {len(self.data)})"
                )

    def num_fragments(self) -> int:
        """Return the number of full fragments in the archive."""
        packed_frag = self.fragment_size // 4
        if packed_frag == 0:
            return 0
        return self.total_packed_bytes // packed_frag

    def fragment_base_count(self, idx: int) -> int:
        """Return the number of bases in fragment `idx` (last fragment may be short)."""
        start = idx * self.fragment_size
        remaining = self.total_bases - start
        return min(self.fragment_size, remaining)

    def extract_sequence(self, fragment_idx: int, start: int, length: int) -> str:
        """Extract `length` bases starting at `start` in fragment `fragment_idx`."""
        fragment_bases = self.fragment_base_count(fragment_idx)
        if start + length > fragment_bases:
            raise ValueError(
                f"range {start}..{start + length} exceeds fragment size {fragment_bases}"
            )

        packed_frag = self.fragment_size // 4
        offset = fragment_idx * packed_frag
        end = min(offset + packed_frag, len(self.data))
        packed = self.data[offset:end]

        seq = []
        for i in range(length):
            base_offset = start + i
            byte_idx = base_offset // 4
            shift = 6 - 2 * (base_offset % 4)
            bits = (packed[byte_idx] >> shift) & 0b11
            seq.append(self.BITS_TO_BASE[bits])
        return "".join(seq)

    def random_windows(
        self, n: int, length: int, seed: int
    ) -> List[SequenceWindow]:
        """Sample `n` random non-overlapping windows from the archive."""
        rng = random.Random(seed)
        windows: List[SequenceWindow] = []
        attempts = 0
        max_attempts = n * 100

        while len(windows) < n and attempts < max_attempts:
            attempts += 1
            fragment_idx = rng.randrange(self.num_fragments())
            fragment_bases = self.fragment_base_count(fragment_idx)
            if fragment_bases < length:
                continue
            max_start = fragment_bases - length
            start = rng.randrange(0, max_start + 1)
            try:
                seq = self.extract_sequence(fragment_idx, start, length)
            except (ValueError, IndexError):
                continue
            if set(seq).issubset(BASE_SET):
                windows.append(SequenceWindow(seq, fragment_idx, start, self.fragment_size))

        if len(windows) < n:
            raise RuntimeError(
                f"could only extract {len(windows)}/{n} valid {length}-bp windows from {self.path}"
            )
        return windows


def load_fasta(path: Union[str, Path], min_len: int = 32) -> List[str]:
    """Return uppercase ACGT sequences from a FASTA (or .gz FASTA) file."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open

    seqs: List[str] = []
    current: List[str] = []

    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    s = "".join(current).upper()
                    if len(s) >= min_len and set(s).issubset(BASE_SET):
                        seqs.append(s)
                    current = []
            else:
                current.append(line)

    if current:
        s = "".join(current).upper()
        if len(s) >= min_len and set(s).issubset(BASE_SET):
            seqs.append(s)

    return seqs


def sample_fasta_windows(
    path: Union[str, Path], n: int, length: int, seed: int
) -> List[SequenceWindow]:
    """Sample `n` random `length`-bp windows from a FASTA file."""
    all_seqs = load_fasta(path)
    if not all_seqs:
        raise ValueError(f"no valid sequences found in {path}")

    concatenated = "".join(all_seqs)
    rng = random.Random(seed)
    windows: List[SequenceWindow] = []
    attempts = 0
    max_attempts = n * 100

    while len(windows) < n and attempts < max_attempts:
        attempts += 1
        max_start = max(0, len(concatenated) - length)
        start = 0
        if max_start == 0:
            window = concatenated
        else:
            start = rng.randrange(0, max_start + 1)
            window = concatenated[start : start + length]
        if len(window) == length and set(window).issubset(BASE_SET):
            windows.append(SequenceWindow(window, 0, start, len(concatenated)))

    if len(windows) < n:
        raise RuntimeError(f"could only extract {len(windows)}/{n} windows from {path}")
    return windows


def resolve_genome_source(path_or_name: Optional[Union[str, Path]]) -> Path:
    """Resolve a genome source path, falling back to common cache locations."""
    if path_or_name:
        candidate = Path(path_or_name).expanduser()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"genome source not found: {path_or_name}")

    candidates = [
        Path.home() / ".rusty-xenom" / "grch38.xenom",
        Path.home() / ".xenom-miner" / "genomes" / "grch38.xenom",
        Path.home() / ".xenom-miner" / "models" / "genomes" / "grch38.xenom",
        Path("grch38.xenom"),
    ]
    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        "no genome source provided and grch38.xenom not found in common locations"
    )


def load_windows(
    path: Union[str, Path],
    n: int,
    length: int,
    seed: int,
) -> List[SequenceWindow]:
    """Load `n` windows of `length` from a .xenom archive or .fasta file."""
    path = Path(path)
    if path.suffix.lower() == ".xenom":
        archive = XenomArchive(path)
        return archive.random_windows(n, length, seed)
    if path.suffix.lower() in (".fasta", ".fa", ".fna"):
        return sample_fasta_windows(path, n, length, seed)
    raise ValueError(f"unsupported genome source: {path} (use .xenom or .fasta)")


@dataclass
class GenomicRegion:
    """A named genomic interval (half-open, 0-based)."""

    name: str
    start: int
    end: int
    strand: str = "+"


@dataclass
class GenomeAnnotation:
    """Simple GFF/BED-style annotation container."""

    regions: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)

    def add(self, name: str, start: int, end: int) -> None:
        """Add a half-open interval to a region class."""
        self.regions.setdefault(name, []).append((start, end))

    def contains(self, name: str, pos: int) -> bool:
        """Return True if `pos` falls inside any interval of region `name`."""
        for start, end in self.regions.get(name, []):
            if start <= pos < end:
                return True
        return False


def load_gff_regions(path: Union[str, Path]) -> GenomeAnnotation:
    """Load a minimal subset of GFF/GTF features for region splitting.

    Supports common feature names: promoter, enhancer, exon, intron,
    CpG_island, intergenic, plus any user-defined feature type.
    """
    ann = GenomeAnnotation()
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            try:
                start = int(parts[3]) - 1  # GFF is 1-based
                end = int(parts[4])
            except ValueError:
                continue
            feature = parts[2].lower()
            ann.add(feature, start, end)
    return ann


def load_bed_regions(path: Union[str, Path]) -> GenomeAnnotation:
    """Load a BED file where the fourth column is the region name."""
    ann = GenomeAnnotation()
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            try:
                start = int(parts[1])
                end = int(parts[2])
            except ValueError:
                continue
            feature = parts[3].lower()
            ann.add(feature, start, end)
    return ann


def load_annotation(path: Optional[Union[str, Path]]) -> Optional[GenomeAnnotation]:
    """Load a GFF/GTF/BED annotation file, returning None if not provided."""
    if not path:
        return None
    path = Path(path)
    if path.suffix.lower() in (".gff", ".gff3", ".gtf"):
        return load_gff_regions(path)
    if path.suffix.lower() == ".bed":
        return load_bed_regions(path)
    raise ValueError(f"unsupported annotation format: {path}")
