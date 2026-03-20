#!/usr/bin/env python3
"""
Genome Variant Annotation Pipeline — GRCh38 (MVP)
GPU-accelerated (cupy/CUDA) when available, CPU (numpy) otherwise.
Uses Ensembl VEP REST API — no local bioinformatics tools required.

Pipeline:
  VCF → normalization → annotation → gene scoring → cohort grouping → dataset index

Usage:
  python genome_annotate.py --input DIR --output DIR [--max-variants 200]

Inputs (optional — uses built-in demo variants if none found):
  *.vcf or *.vcf.gz files in --input DIR

Outputs:
  annotated.vcf       — VCF with VEP consequence, IMPACT, gene, AF, ClinVar in INFO/CSQ
  analysis.json       — quality metrics and L2 score
  predictions.json    — score field for miner (mirrors analysis.json score)
  dataset_index.json  — searchable dataset index with gene scores and cohort summary
"""

import argparse
import gzip
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# ── Array backend (numpy / cupy) — set before pipeline stages ────────────────
# Honour USE_GPU env var or --gpu CLI flag (checked early via sys.argv).

if "--gpu" in sys.argv:
    os.environ.setdefault("USE_GPU", "1")

_GPU_BACKEND = "numpy/CPU"
try:
    if os.environ.get("USE_GPU", "0") not in ("0", "", "false", "False"):
        import cupy as _np          # type: ignore
        _np.array([1.0], dtype=_np.float32)  # smoke-test: raises if CUDA absent
        _GPU_BACKEND = f"cupy {_np.__version__}/CUDA"
    else:
        raise ImportError("GPU disabled")
except Exception:
    import numpy as _np             # type: ignore
    _GPU_BACKEND = f"numpy {_np.__version__}/CPU"

# ── Constants ──────────────────────────────────────────────────────────────────

ENSEMBL_REST = "https://rest.ensembl.org"
VEP_REGION_URL = f"{ENSEMBL_REST}/vep/human/region"
VEP_BATCH_SIZE = 200   # Ensembl POST /vep/human/region max
REQUEST_DELAY  = 1.1   # seconds between API calls (rate limit: ~15 req/s, be conservative)

# Well-known clinically relevant variants (GRCh38) used when no input VCFs are provided.
# These reliably annotate via Ensembl VEP with HIGH/MODERATE impact and ClinVar significance.
DEMO_VARIANTS = [
    # BRCA1 pathogenic splice — chr17:43094332 C>T (GRCh38)
    {"chrom": "17", "pos": 43094332, "id": "rs80357346", "ref": "C", "alt": "T"},
    # BRCA1 5382insC equivalent region
    {"chrom": "17", "pos": 43057051, "id": "rs80357713", "ref": "A", "alt": "T"},
    # BRCA2 pathogenic — chr13:32340300
    {"chrom": "13", "pos": 32340300, "id": "rs28897672", "ref": "A", "alt": "G"},
    # TP53 R248W hotspot — chr17:7675088
    {"chrom": "17", "pos": 7675088,  "id": "rs28934578", "ref": "G", "alt": "A"},
    # KRAS G12D — chr12:25245350
    {"chrom": "12", "pos": 25245350, "id": "rs121913529", "ref": "C", "alt": "T"},
    # CFTR F508del region — chr7:117548628
    {"chrom": "7",  "pos": 117548628, "id": "rs113993960", "ref": "CTT", "alt": "C"},
    # LDLR familial hypercholesterolaemia — chr19:11089435
    {"chrom": "19", "pos": 11089435, "id": "rs28942074", "ref": "G", "alt": "A"},
    # MTHFR C677T common variant — chr1:11796321
    {"chrom": "1",  "pos": 11796321, "id": "rs1801133",  "ref": "G", "alt": "A"},
    # APOE ε4 risk allele — chr19:45411941
    {"chrom": "19", "pos": 45411941, "id": "rs429358",   "ref": "T", "alt": "C"},
    # PCSK9 gain-of-function — chr1:55039879
    {"chrom": "1",  "pos": 55039879, "id": "rs11591147", "ref": "G", "alt": "T"},
    # HFE C282Y haemochromatosis — chr6:26091179
    {"chrom": "6",  "pos": 26091179, "id": "rs1800562",  "ref": "G", "alt": "A"},
    # FBN1 Marfan syndrome — chr15:48760590
    {"chrom": "15", "pos": 48760590, "id": "rs121913526", "ref": "G", "alt": "A"},
    # TTN truncating cardiomyopathy region — chr2:178524101
    {"chrom": "2",  "pos": 178524101, "id": "rs1437142407", "ref": "C", "alt": "T"},
    # MSH2 Lynch syndrome — chr2:47640227
    {"chrom": "2",  "pos": 47640227, "id": "rs63750527", "ref": "G", "alt": "A"},
    # PTEN hamartoma — chr10:89692905
    {"chrom": "10", "pos": 89692905, "id": "rs121909228", "ref": "C", "alt": "T"},
]

# ── VCF parsing ────────────────────────────────────────────────────────────────

def parse_vcf_file(path: Path, max_variants: int) -> list[dict]:
    """Parse a VCF (.vcf or .vcf.gz) — returns list of variant dicts."""
    variants = []
    opener = gzip.open if str(path).endswith(".gz") else open
    mode   = "rt"
    try:
        with opener(path, mode) as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 5:
                    continue
                chrom = parts[0].lstrip("chr")
                try:
                    pos = int(parts[1])
                except ValueError:
                    continue
                vid = parts[2] if parts[2] != "." else f"{chrom}:{pos}"
                ref = parts[3]
                alt = parts[4].split(",")[0]  # first ALT allele only
                variants.append({"chrom": chrom, "pos": pos, "id": vid, "ref": ref, "alt": alt})
                if len(variants) >= max_variants:
                    break
    except Exception as e:
        print(f"[WARN] Failed to parse {path}: {e}", file=sys.stderr)
    return variants

def collect_vcf_variants(input_dir: Path, max_variants: int) -> list[dict]:
    """Collect up to max_variants from all VCF files in input_dir."""
    variants: list[dict] = []
    for suffix in ("*.vcf", "*.vcf.gz"):
        for vcf in sorted(input_dir.glob(suffix)):
            batch = parse_vcf_file(vcf, max_variants - len(variants))
            variants.extend(batch)
            if len(variants) >= max_variants:
                break
        if len(variants) >= max_variants:
            break
    return variants[:max_variants]

# ── Stage 1: Normalization ─────────────────────────────────────────────────────

MAX_ALLELE_LEN = 50  # Ensembl VEP REST rejects very long alleles

def normalize_variant(v: dict) -> "dict | None":
    """Left-trim common prefix/suffix, reject symbolic/complex alleles."""
    ref = v["ref"].upper()
    alt = v["alt"].upper()
    # Reject symbolic alleles (<DEL>, <INS>, etc.) and missing alleles
    if alt.startswith("<") or "." in alt or "*" in alt or alt == "N":
        return None
    # Trim common suffix (right-align)
    while len(ref) > 1 and len(alt) > 1 and ref[-1] == alt[-1]:
        ref, alt = ref[:-1], alt[:-1]
    # Trim common prefix (left-align) — adjust POS
    offset = 0
    while len(ref) > 1 and len(alt) > 1 and ref[0] == alt[0]:
        ref, alt = ref[1:], alt[1:]
        offset += 1
    # Reject alleles that are still too long for the REST API
    if len(ref) > MAX_ALLELE_LEN or len(alt) > MAX_ALLELE_LEN:
        return None
    return {**v, "ref": ref, "alt": alt, "pos": v["pos"] + offset}

def normalize_variants(variants: list) -> list:
    """Normalize and deduplicate variant list."""
    seen: set = set()
    result = []
    skipped = 0
    for v in variants:
        n = normalize_variant(v)
        if n is None:
            skipped += 1
            continue
        key = (n["chrom"], n["pos"], n["ref"], n["alt"])
        if key in seen:
            skipped += 1
            continue
        seen.add(key)
        result.append(n)
    if skipped:
        print(f"[INFO] Normalization: kept {len(result)}, skipped {skipped} (symbolic/dup/toolong)",
              file=sys.stderr)
    return result

# ── Ensembl VEP REST annotation ────────────────────────────────────────────────

def _vep_region_line(v: dict) -> str:
    """Format a variant dict as a VEP region string: 'CHR START END REF/ALT +'"""
    start = v["pos"]
    # For deletions: end = start + len(ref) - 1
    end = start + max(len(v["ref"]), 1) - 1
    allele = f"{v['ref']}/{v['alt']}"
    return f"{v['chrom']} {start} {end} {allele} +"

def _post_json(url: str, payload: dict, retries: int = 3) -> list | None:
    """HTTP POST with JSON body to Ensembl REST — returns parsed JSON or None."""
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept":        "application/json",
        },
        method="POST",
    )
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            if e.code == 429:
                wait = 2 ** attempt
                print(f"[WARN] Ensembl rate limit (429) — sleeping {wait}s", file=sys.stderr)
                time.sleep(wait)
            else:
                print(f"[WARN] Ensembl HTTP {e.code}: {body[:200]}", file=sys.stderr)
                if attempt == retries:
                    return None
                time.sleep(REQUEST_DELAY)
        except Exception as e:
            print(f"[WARN] Ensembl request failed ({attempt}/{retries}): {e}", file=sys.stderr)
            if attempt == retries:
                return None
            time.sleep(REQUEST_DELAY)
    return None

def annotate_batch(variants: list[dict]) -> list[dict]:
    """Send one batch to VEP and return merged annotation dicts."""
    region_lines = [_vep_region_line(v) for v in variants]
    result = _post_json(VEP_REGION_URL, {
        "variants":          region_lines,
        "canonical":         True,
        "hgvs":              True,
        "af":                True,
        "af_gnomadg":        True,
        "af_gnomade":        True,
        "pubmed":            False,
        "vcf_string":        True,
        "clinical_significance": True,
    })
    if result is None:
        return []
    # Map input variants to annotation results by index
    annotated = []
    for ann in result:
        if not isinstance(ann, dict):
            continue
        # Extract canonical transcript consequence
        tc_list = ann.get("transcript_consequences", [])
        canon   = next((t for t in tc_list if t.get("canonical") == 1), tc_list[0] if tc_list else {})
        csq_terms = ",".join(canon.get("consequence_terms", ["intergenic_variant"]))
        impact    = canon.get("impact", "MODIFIER")
        gene      = canon.get("gene_symbol", ann.get("gene_id", "-"))
        hgvsc     = canon.get("hgvsc", "-")
        hgvsp     = canon.get("hgvsp", "-")
        af_gnomad = canon.get("af_gnomadg") or canon.get("af_gnomade") or ann.get("allele_string", "")
        # ClinVar significance from colocated variants
        clin_sig  = "-"
        for cv in ann.get("colocated_variants", []):
            cs = cv.get("clin_sig", [])
            if cs:
                clin_sig = ",".join(cs)
                break
        annotated.append({
            "input":    ann.get("input", ""),
            "allele":   ann.get("allele_string", ""),
            "impact":   impact,
            "csq":      csq_terms,
            "gene":     gene,
            "hgvsc":    hgvsc,
            "hgvsp":    hgvsp,
            "af":       af_gnomad,
            "clin_sig": clin_sig,
        })
    return annotated

def annotate_variants(variants: list[dict]) -> list[dict]:
    """Annotate all variants in batches, respecting rate limits."""
    annotated: list[dict] = []
    for i in range(0, len(variants), VEP_BATCH_SIZE):
        batch = variants[i : i + VEP_BATCH_SIZE]
        print(f"[INFO] Annotating variants {i+1}–{i+len(batch)} of {len(variants)} via Ensembl VEP…")
        batch_ann = annotate_batch(batch)
        annotated.extend(batch_ann)
        if i + VEP_BATCH_SIZE < len(variants):
            time.sleep(REQUEST_DELAY)
    return annotated

# ── Output writing ─────────────────────────────────────────────────────────────

VCF_HEADER = """\
##fileformat=VCFv4.2
##reference=GRCh38
##source=genome_annotate.py (Xenom L2 MVP)
##INFO=<ID=CSQ,Number=.,Type=String,Description="VEP consequence|impact|gene|HGVSc|HGVSp|AF|ClinVar">
##INFO=<ID=GENE,Number=1,Type=String,Description="Gene symbol (VEP canonical)">
##INFO=<ID=IMPACT,Number=1,Type=String,Description="VEP impact: HIGH|MODERATE|LOW|MODIFIER">
##INFO=<ID=CLIN_SIG,Number=.,Type=String,Description="ClinVar clinical significance">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
"""

def write_annotated_vcf(output_path: Path, variants: list[dict], annotations: list[dict]) -> None:
    """Write annotated VCF file."""
    ann_by_idx = {i: a for i, a in enumerate(annotations)}
    with open(output_path, "w") as fh:
        fh.write(VCF_HEADER)
        for i, v in enumerate(variants):
            ann = ann_by_idx.get(i, {})
            csq_str  = f"{ann.get('csq','-')}|{ann.get('impact','-')}|{ann.get('gene','-')}|{ann.get('hgvsc','-')}|{ann.get('hgvsp','-')}|{ann.get('af','-')}|{ann.get('clin_sig','-')}"
            info     = f"CSQ={csq_str};GENE={ann.get('gene','-')};IMPACT={ann.get('impact','-')};CLIN_SIG={ann.get('clin_sig','-')}"
            chrom    = v["chrom"] if v["chrom"].startswith("chr") else f"chr{v['chrom']}"
            fh.write(f"{chrom}\t{v['pos']}\t{v['id']}\t{v['ref']}\t{v['alt']}\t.\tPASS\t{info}\n")

# ── Scoring constants ──────────────────────────────────────────────────────────

IMPACT_WEIGHT = {"HIGH": 1.0, "MODERATE": 0.4, "LOW": 0.1, "MODIFIER": 0.01}
HIGH_IMPACT   = {"HIGH"}
MED_IMPACT    = {"HIGH", "MODERATE"}
PATHOGENIC    = {"pathogenic", "likely_pathogenic"}

# ── Stage 3: Gene scoring ───────────────────────────────────────────────────


def score_genes(annotations: list) -> dict:
    """Aggregate per-gene scores — GPU-vectorized weight accumulation via cupy/numpy."""
    if not annotations:
        return {}

    genes_list = [a.get("gene", "-") for a in annotations]
    weights  = _np.array([IMPACT_WEIGHT.get(a.get("impact", "MODIFIER"), 0.01) for a in annotations], dtype=_np.float32)
    is_high  = _np.array([a.get("impact") == "HIGH" for a in annotations], dtype=_np.bool_)
    is_path  = _np.array([
        any(s in PATHOGENIC for s in a.get("clin_sig", "-").split(","))
        for a in annotations
    ], dtype=_np.bool_)

    try:  # cupy → pull to host
        weights_h = weights.get().tolist()
        is_high_h = is_high.get().tolist()
        is_path_h = is_path.get().tolist()
    except AttributeError:  # already numpy
        weights_h = weights.tolist()
        is_high_h = is_high.tolist()
        is_path_h = is_path.tolist()

    genes: dict = {}
    for i, gene in enumerate(genes_list):
        if gene == "-":
            continue
        if gene not in genes:
            genes[gene] = {"variants": 0, "high_impact": 0, "pathogenic": 0, "raw_score": 0.0}
        g = genes[gene]
        g["variants"]  += 1
        g["raw_score"] += weights_h[i]
        if is_high_h[i]: g["high_impact"] += 1
        if is_path_h[i]: g["pathogenic"]  += 1

    result = {}
    for gene, g in genes.items():
        base  = min(g["raw_score"] / max(g["variants"], 1), 1.0)
        boost = min(g["pathogenic"] * 0.15, 0.30)
        result[gene] = {
            "variants":    g["variants"],
            "high_impact": g["high_impact"],
            "pathogenic":  g["pathogenic"],
            "score":       round(min(base + boost, 1.0), 4),
        }
    return result

# ── Stage 4: Cohort grouping ──────────────────────────────────────────────────

def group_cohorts(variants: list, annotations: list) -> dict:
    """Group variants by chromosome and aggregate stats per cohort."""
    chroms: dict = {}
    for v, ann in zip(variants[:len(annotations)], annotations):
        chrom = "chr" + v["chrom"].lstrip("chr")
        if chrom not in chroms:
            chroms[chrom] = {"variants": 0, "high_impact": 0, "pathogenic": 0, "genes": set()}
        c = chroms[chrom]
        c["variants"] += 1
        if ann.get("impact") == "HIGH":
            c["high_impact"] += 1
        if any(s in PATHOGENIC for s in ann.get("clin_sig", "-").split(",")):
            c["pathogenic"] += 1
        gene = ann.get("gene", "-")
        if gene != "-":
            c["genes"].add(gene)
    # Sort chromosomes numerically, convert gene sets to sorted lists
    def _chrom_key(k: str) -> tuple:
        s = k.lstrip("chr")
        return (0, int(s)) if s.isdigit() else (1, s)
    return {
        k: {**v, "genes": sorted(v["genes"])[:15]}
        for k, v in sorted(chroms.items(), key=lambda x: _chrom_key(x[0]))
    }

# ── Stage 5: Dataset index ───────────────────────────────────────────────────

def build_dataset_index(
    variants:    list,
    annotations: list,
    gene_scores: dict,
    cohorts:     dict,
    score:       float,
) -> dict:
    """Build a searchable dataset index for the result set."""
    top_genes = sorted(
        gene_scores.items(),
        key=lambda x: (x[1]["score"], x[1]["pathogenic"]),
        reverse=True,
    )[:20]
    top_pathogenic = [
        {"gene": ann.get("gene","-"), "csq": ann.get("csq","-"),
         "clin_sig": ann.get("clin_sig","-"), "hgvsp": ann.get("hgvsp","-")}
        for ann in annotations
        if any(s in PATHOGENIC for s in ann.get("clin_sig","-").split(","))
    ][:20]
    return {
        "schema_version":       "1.0",
        "reference":            "GRCh38",
        "pipeline":             "variant_annotation_grch38",
        "total_variants":       len(variants),
        "annotated_variants":   len(annotations),
        "genes_affected":       len(gene_scores),
        "chromosomes_affected": list(cohorts.keys()),
        "score":                score,
        "top_genes_by_score":   [{"gene": g, **s} for g, s in top_genes],
        "top_pathogenic":       top_pathogenic,
        "cohort_summary":       cohorts,
    }

def compute_score(variants: list, annotations: list, gene_scores: dict) -> "tuple[float, dict]":
    """Compute L2 score — GPU-vectorized count operations via cupy/numpy."""
    n_total = len(variants)
    if n_total == 0:
        return 0.1, {"error": "no_variants"}

    n_annotated = len(annotations)
    if n_annotated > 0:
        imp_arr  = _np.array([1 if a.get("impact") in HIGH_IMPACT else 0 for a in annotations], dtype=_np.int32)
        cod_arr  = _np.array([1 if a.get("impact") in MED_IMPACT  else 0 for a in annotations], dtype=_np.int32)
        path_arr = _np.array([
            1 if any(s in PATHOGENIC for s in a.get("clin_sig", "-").split(",")) else 0
            for a in annotations
        ], dtype=_np.int32)
        try:
            n_high       = int(imp_arr.sum())
            n_coding     = int(cod_arr.sum())
            n_pathogenic = int(path_arr.sum())
        except Exception:
            n_high       = int(sum(imp_arr.tolist()))
            n_coding     = int(sum(cod_arr.tolist()))
            n_pathogenic = int(sum(path_arr.tolist()))
    else:
        n_high = n_coding = n_pathogenic = 0
    n_genes      = len(gene_scores)

    annotation_rate  = n_annotated / n_total
    coding_rate      = n_coding / max(n_annotated, 1)
    pathogenic_score = min(n_pathogenic / max(n_annotated, 1) * 5, 1.0)
    diversity_score  = min(n_genes / 10, 1.0)
    # Gene-score quality: average of top-10 gene scores
    top10_avg = (sum(g["score"] for g in sorted(
        gene_scores.values(), key=lambda x: x["score"], reverse=True
    )[:10]) / 10) if gene_scores else 0.0

    raw = (annotation_rate  * 0.35
           + coding_rate    * 0.25
           + pathogenic_score * 0.20
           + diversity_score  * 0.10
           + top10_avg        * 0.10)
    score = round(0.10 + raw * 0.85, 6)
    score = max(0.10, min(0.95, score))

    metrics = {
        "reference":        "GRCh38",
        "pipeline":         "variant_annotation_grch38",
        "total_variants":   n_total,
        "annotated":        n_annotated,
        "annotation_rate":  round(annotation_rate, 4),
        "high_impact":      n_high,
        "coding_variants":  n_coding,
        "coding_rate":      round(coding_rate, 4),
        "pathogenic_count": n_pathogenic,
        "unique_genes":     n_genes,
        "top_genes":        sorted(gene_scores.keys(),
                                   key=lambda g: gene_scores[g]["score"], reverse=True)[:20],
        "score":            score,
    }
    return score, metrics

# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Genome variant annotation pipeline (GRCh38, Ensembl VEP REST)")
    parser.add_argument("--input",        required=True, help="Input directory (VCF files or empty for demo variants)")
    parser.add_argument("--output",       required=True, help="Output directory")
    parser.add_argument("--max-variants", type=int, default=200, help="Maximum variants to annotate (default: 200)")
    parser.add_argument("--gpu",          action="store_true", help="Enable GPU acceleration via cupy/CUDA (also honoured via USE_GPU=1 env var)")
    args = parser.parse_args()

    print(f"[INFO] Array backend: {_GPU_BACKEND}", file=sys.stderr)

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Collect + Normalize ──────────────────────────────────────────────
    raw_variants = collect_vcf_variants(input_dir, args.max_variants)
    if raw_variants:
        print(f"[INFO] Stage 1: Loaded {len(raw_variants)} raw variant(s) from {input_dir}")
        variants = normalize_variants(raw_variants)
        print(f"[INFO] Stage 1: {len(variants)} variants after normalization")
    else:
        print(f"[INFO] Stage 1: No VCF files found — using {len(DEMO_VARIANTS)} demo variants (GRCh38 known pathogenic)")
        variants = DEMO_VARIANTS[:args.max_variants]

    # ── Stage 2: Annotation via Ensembl VEP REST ──────────────────────────────────
    annotations = annotate_variants(variants)
    print(f"[INFO] Stage 2: Annotated {len(annotations)} of {len(variants)} variants")

    # ── Stage 3: Gene scoring ───────────────────────────────────────────────────
    gene_scores = score_genes(annotations)
    print(f"[INFO] Stage 3: Scored {len(gene_scores)} gene(s)")

    # ── Stage 4: Cohort grouping ────────────────────────────────────────────────
    cohorts = group_cohorts(variants, annotations)
    print(f"[INFO] Stage 4: Grouped into {len(cohorts)} chromosome cohort(s)")

    # ── Stage 5: Compute score + write all outputs ───────────────────────────────
    score, metrics = compute_score(variants, annotations, gene_scores)

    vcf_path = output_dir / "annotated.vcf"
    write_annotated_vcf(vcf_path, variants, annotations)
    print(f"[INFO] Wrote {vcf_path}")

    analysis = {**metrics, "gene_scores": gene_scores, "cohort_summary": cohorts}
    analysis_path = output_dir / "analysis.json"
    with open(analysis_path, "w") as fh:
        json.dump(analysis, fh, indent=2)
    print(f"[INFO] Wrote {analysis_path}")

    pred_path = output_dir / "predictions.json"
    with open(pred_path, "w") as fh:
        json.dump({"score": score, "task": "variant_annotation_grch38", **metrics}, fh, indent=2)
    print(f"[INFO] Wrote {pred_path}")

    idx = build_dataset_index(variants, annotations, gene_scores, cohorts, score)
    idx_path = output_dir / "dataset_index.json"
    with open(idx_path, "w") as fh:
        json.dump(idx, fh, indent=2)
    print(f"[INFO] Wrote {idx_path}")

    print(f"[RESULT] score={score:.4f}  annotated={len(annotations)}/{len(variants)}  "
          f"genes={len(gene_scores)}  cohorts={len(cohorts)}  "
          f"high_impact={metrics['high_impact']}  pathogenic={metrics['pathogenic_count']}")
    print(json.dumps({"score": score}))

if __name__ == "__main__":
    main()
