#!/usr/bin/env python3
"""
Dataset fingerprint and comparison utility.

Usage:
  - Compute fingerprint for one root:
      python dataset_fingerprint.py --root /path/to/dataset
  - Compare two roots:
      python dataset_fingerprint.py --root /path/to/datasetA --root2 /path/to/datasetB

By default, hashes file contents for common data types (images, videos, npy/npz, pt/pth, json/txt).
For very large files, uses chunked hashing to avoid excessive memory.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif",
    ".mp4", ".avi", ".mov", ".mkv",
    ".npy", ".npz", ".pt", ".pth", ".bin",
    ".json", ".txt", ".csv",
    ".lmdb"  # hashes LMDB files at the filesystem level
}


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def collect_files(root: Path, exts: List[str] = None, exclude_patterns: List[str] = None) -> List[Path]:
    if exts is None:
        exts = list(DEFAULT_EXTS)
    exts = {e.lower() for e in exts}
    exclude_patterns = exclude_patterns or []

    files = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if any(pattern in rel for pattern in exclude_patterns):
            continue
        if p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)


def fingerprint_root(root: Path, exts: List[str] = None, exclude_patterns: List[str] = None) -> Dict:
    files = collect_files(root, exts, exclude_patterns)
    per_file_hash: Dict[str, str] = {}
    total_bytes = 0
    per_ext_counts: Dict[str, int] = {}

    for f in files:
        rel = f.relative_to(root).as_posix()
        h = sha256_file(f)
        per_file_hash[rel] = h
        total_bytes += f.stat().st_size
        per_ext_counts[f.suffix.lower()] = per_ext_counts.get(f.suffix.lower(), 0) + 1

    # Build a stable combined fingerprint of the dataset file set + contents
    combined = hashlib.sha256()
    for rel in sorted(per_file_hash.keys()):
        combined.update(rel.encode("utf-8"))
        combined.update(b"::")
        combined.update(per_file_hash[rel].encode("utf-8"))
        combined.update(b"\n")

    return {
        "root": root.as_posix(),
        "total_files": len(files),
        "total_bytes": total_bytes,
        "per_ext_counts": per_ext_counts,
        "per_file_hash": per_file_hash,
        "dataset_sha256": combined.hexdigest(),
    }


def compare_fingerprints(a: Dict, b: Dict) -> Dict:
    a_set = set(a["per_file_hash"].keys())
    b_set = set(b["per_file_hash"].keys())
    only_in_a = sorted(a_set - b_set)
    only_in_b = sorted(b_set - a_set)

    common = sorted(a_set & b_set)
    content_mismatch: List[Tuple[str, str, str]] = []
    for rel in common:
        ha = a["per_file_hash"][rel]
        hb = b["per_file_hash"][rel]
        if ha != hb:
            content_mismatch.append((rel, ha, hb))

    identical = (
        len(only_in_a) == 0 and
        len(only_in_b) == 0 and
        len(content_mismatch) == 0 and
        a["dataset_sha256"] == b["dataset_sha256"]
    )

    return {
        "identical": identical,
        "root_a": a["root"],
        "root_b": b["root"],
        "dataset_sha256_a": a["dataset_sha256"],
        "dataset_sha256_b": b["dataset_sha256"],
        "only_in_a_count": len(only_in_a),
        "only_in_b_count": len(only_in_b),
        "content_mismatch_count": len(content_mismatch),
        "only_in_a": only_in_a,
        "only_in_b": only_in_b,
        "content_mismatch": content_mismatch,
    }


def main():
    parser = argparse.ArgumentParser(description="Dataset fingerprint and comparison utility")
    parser.add_argument("--root", required=True, help="Dataset root A")
    parser.add_argument("--root2", default=None, help="Dataset root B (optional)")
    parser.add_argument("--exts", nargs="*", default=None, help="File extensions to include (default: common data types)")
    parser.add_argument("--exclude", nargs="*", default=None, help="Exclude patterns in relative paths")
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    args = parser.parse_args()

    exts = args.exts if args.exts else None
    exclude_patterns = args.exclude if args.exclude else None

    fp_a = fingerprint_root(Path(args.root), exts, exclude_patterns)

    if args.root2:
        fp_b = fingerprint_root(Path(args.root2), exts, exclude_patterns)
        result = compare_fingerprints(fp_a, fp_b)
        if args.json:
            print(json.dumps({"fingerprint_a": fp_a, "fingerprint_b": fp_b, "comparison": result}, indent=2))
        else:
            print("=== Dataset A ===")
            print(json.dumps(fp_a, indent=2))
            print("\n=== Dataset B ===")
            print(json.dumps(fp_b, indent=2))
            print("\n=== Comparison ===")
            print(json.dumps(result, indent=2))
            if result["identical"]:
                print("\nDatasets are IDENTICAL by file set + content hash.")
            else:
                print("\nDatasets DIFFER. See details above.")
    else:
        if args.json:
            print(json.dumps(fp_a, indent=2))
        else:
            print(json.dumps(fp_a, indent=2))


if __name__ == "__main__":
    main()
