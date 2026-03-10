#!/usr/bin/env python3
"""
VPS Builder Pipeline: Download Features from R2 → Merge → Build FAISS Index → Upload

This worker downloads all feature .npy files and metadata .jsonl files
produced by VPS_Pipeline workers from R2, merges them into a single
features.npy + metadata.json, builds a FAISS IVFPQ index, uploads the
result to R2, and self-destructs.

Progress is logged to stdout in structured format:
    PROGRESS|builder|{step}|{detail}|{pct}|{status}
"""

import gc
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Fix OpenMP conflict — use all available cores
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
_ncpu = str(os.cpu_count() or 4)
os.environ["OMP_NUM_THREADS"] = _ncpu
os.environ["OPENBLAS_NUM_THREADS"] = _ncpu
os.environ["MKL_NUM_THREADS"] = _ncpu

from r2_storage import R2Client

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_DIM = 8448  # MegaLoc feature dimension
WORK_DIR = Path("/app/work")

# FAISS index settings (overridable via env vars)
INDEX_TYPE = os.environ.get("INDEX_TYPE", "ivfpq")
NLIST = int(os.environ.get("NLIST", "1024"))
M = int(os.environ.get("M", "32"))
NBITS = int(os.environ.get("NBITS", "8"))
TRAIN_SAMPLES = int(os.environ.get("TRAIN_SAMPLES", "1000000"))
NITER = int(os.environ.get("NITER", "100"))


# ═══════════════════════════════════════════════════════════════════════════════
# Environment Variables
# ═══════════════════════════════════════════════════════════════════════════════

def get_env(key: str, default: str = None) -> str:
    val = os.environ.get(key, default)
    if val is None:
        print(f"[FATAL] Missing required env var: {key}")
        sys.exit(1)
    return val


# ═══════════════════════════════════════════════════════════════════════════════
# Status Reporter
# ═══════════════════════════════════════════════════════════════════════════════

class StatusReporter:
    """Reports progress to R2 and stdout.

    Status key format: Status/INDEX_{city_name}_{instance_id}.json
    Flat structure avoids location-nested paths.
    """

    def __init__(self, r2: R2Client, city_name: str, instance_id: str):
        self.r2 = r2
        self.city_name = city_name
        self.instance_id = instance_id
        self.status_key = f"Status/INDEX_{city_name}_{instance_id}.json"
        self.start_time = time.time()
        self._last_report = 0

    def report(self, step: str, detail: str, pct: int, status: str = "RUNNING"):
        now = time.time()
        elapsed = now - self.start_time

        print(f"PROGRESS|builder|{step}|{detail}|{pct}|{status}")
        sys.stdout.flush()

        # Throttle R2 updates to every 10s
        if now - self._last_report < 10 and status == "RUNNING":
            return

        self._last_report = now
        self.r2.upload_json(self.status_key, {
            "worker": "builder",
            "step": step,
            "detail": detail,
            "pct": pct,
            "status": status,
            "elapsed_seconds": int(elapsed),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

    def report_final(self, status: str, detail: str = ""):
        self.report("done", detail, 100, status)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Discover feature files on R2
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_worker_npy(key: str) -> Optional[Tuple[int, int]]:
    """
    Parse worker_index and total_workers from a .npy key.

    Expected filename suffix: {city}_{worker_index}.{total_workers}.npy
    Returns (worker_index, total_workers) or None if the pattern doesn't match.
    """
    m = re.search(r'_(\d+)\.(\d+)\.npy$', key)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _parse_worker_jsonl(key: str) -> Optional[Tuple[int, int]]:
    """
    Parse worker_index and total_workers from a Metadata .jsonl key.

    Expected filename suffix: Metadata_{city}_{worker_index}.{total_workers}.jsonl
    """
    m = re.search(r'_(\d+)\.(\d+)\.jsonl$', key)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _parse_chunk_npy(key: str) -> Optional[int]:
    """Parse chunk number from a chunk-based .npy key.
    Expected: {city}_chunk_{NNNN}.npy → returns chunk number."""
    m = re.search(r'_chunk_(\d{4})\.npy$', key)
    if m:
        return int(m.group(1))
    return None


def _parse_chunk_jsonl(key: str) -> Optional[int]:
    """Parse chunk number from a Metadata .jsonl key.
    Expected: Metadata_{city}_chunk_{NNNN}.jsonl → returns chunk number."""
    m = re.search(r'_chunk_(\d{4})\.jsonl$', key)
    if m:
        return int(m.group(1))
    return None


def discover_feature_files(r2: R2Client, features_prefix: str) -> Tuple[List[str], List[str]]:
    """
    List all .npy and .jsonl files under the features prefix.

    Supports two naming conventions:
    - Chunk mode: {city}_chunk_{NNNN}.npy / Metadata_{city}_chunk_{NNNN}.jsonl
    - Worker mode (legacy): {city}_{worker}.{total}.npy / Metadata_{city}_{worker}.{total}.jsonl

    Files are returned sorted by index for deterministic merge order.
    """
    print(f"\n{'='*80}")
    print("STEP 1: Discovering feature files on R2")
    print(f"{'='*80}")
    print(f"Prefix: {features_prefix}")

    all_files = r2.list_files(features_prefix)
    size_map = {f['key']: f['size'] for f in all_files}

    raw_npy   = [f['key'] for f in all_files if f['key'].endswith('.npy')]
    raw_jsonl = [f['key'] for f in all_files if f['key'].endswith('.jsonl')]

    # ── Auto-detect mode: chunk vs worker ──
    chunk_npy_count = sum(1 for k in raw_npy if _parse_chunk_npy(k) is not None)
    worker_npy_count = sum(1 for k in raw_npy if _parse_worker_npy(k) is not None)

    use_chunk_mode = chunk_npy_count > worker_npy_count
    if use_chunk_mode:
        print(f"\nDetected CHUNK mode ({chunk_npy_count} chunk files)")
    else:
        print(f"\nDetected WORKER mode ({worker_npy_count} worker files)")

    if use_chunk_mode:
        # ── Chunk mode: sort by chunk number ──
        npy_by_chunk: Dict[int, str] = {}
        for key in raw_npy:
            chunk_num = _parse_chunk_npy(key)
            if chunk_num is None:
                print(f"  [WARN] Unrecognised .npy filename (skipping): {key}")
                continue
            npy_by_chunk[chunk_num] = key

        npy_files = [npy_by_chunk[i] for i in sorted(npy_by_chunk.keys())]

        # Match .jsonl files by chunk number
        jsonl_by_chunk: Dict[int, str] = {}
        for key in raw_jsonl:
            chunk_num = _parse_chunk_jsonl(key)
            if chunk_num is None:
                print(f"  [WARN] Unrecognised .jsonl filename (skipping): {key}")
                continue
            jsonl_by_chunk[chunk_num] = key

        jsonl_files = [jsonl_by_chunk[i] for i in sorted(npy_by_chunk.keys())
                       if i in jsonl_by_chunk]

        missing_jsonl = [i for i in sorted(npy_by_chunk.keys())
                         if i not in jsonl_by_chunk]
        if missing_jsonl:
            n = len(missing_jsonl)
            print(f"[WARN] Missing metadata (.jsonl) for {n} chunks")
            if n <= 20:
                print(f"  Chunks: {missing_jsonl}")

        print(f"\nFound {len(npy_files)} feature files (.npy)")
        print(f"Found {len(jsonl_files)} metadata files (.jsonl)")

        # Show summary for large counts, details for small
        if len(npy_files) <= 20:
            for key in npy_files:
                size_mb = size_map.get(key, 0) / (1024 * 1024)
                chunk_num = _parse_chunk_npy(key)
                print(f"  [chunk {chunk_num:04d}] {key} ({size_mb:.1f} MB)")
        else:
            total_size = sum(size_map.get(k, 0) for k in npy_files)
            print(f"  Total .npy size: {total_size / (1024**3):.2f} GB across {len(npy_files)} files")

    else:
        # ── Worker mode (legacy) ──
        npy_by_worker: Dict[int, str] = {}
        detected_total: Optional[int] = None
        inconsistent_totals: list = []

        for key in raw_npy:
            parsed = _parse_worker_npy(key)
            if parsed is None:
                print(f"  [WARN] Unrecognised .npy filename (skipping): {key}")
                continue
            w_idx, w_total = parsed
            if detected_total is None:
                detected_total = w_total
            elif w_total != detected_total:
                inconsistent_totals.append((key, w_total))
            npy_by_worker[w_idx] = key

        if inconsistent_totals:
            print(f"[WARN] Inconsistent total_workers across .npy files:")
            for key, t in inconsistent_totals:
                print(f"  {key} has total={t}, expected {detected_total}")

        if detected_total is not None:
            print(f"\nDetected NUM_WORKERS from filenames: {detected_total}")
            present_workers = sorted(npy_by_worker.keys())
            missing_workers = [i for i in range(1, detected_total + 1)
                               if i not in npy_by_worker]
            print(f"Workers present : {present_workers} ({len(present_workers)}/{detected_total})")
            if missing_workers:
                print(f"[WARN] Missing worker files for indices: {missing_workers}")
            else:
                print("All workers accounted for — proceeding with merge.")
        else:
            print("[WARN] Could not detect worker count from .npy filenames.")

        npy_files = [npy_by_worker[i] for i in sorted(npy_by_worker.keys())]

        jsonl_by_worker: Dict[int, str] = {}
        for key in raw_jsonl:
            parsed = _parse_worker_jsonl(key)
            if parsed is None:
                print(f"  [WARN] Unrecognised .jsonl filename (skipping): {key}")
                continue
            w_idx, _ = parsed
            jsonl_by_worker[w_idx] = key

        jsonl_files = [jsonl_by_worker[i] for i in sorted(npy_by_worker.keys())
                       if i in jsonl_by_worker]

        missing_jsonl = [i for i in sorted(npy_by_worker.keys())
                         if i not in jsonl_by_worker]
        if missing_jsonl:
            print(f"[WARN] Missing metadata (.jsonl) for worker indices: {missing_jsonl}")

        print(f"\nFound {len(npy_files)} feature files (.npy)")
        print(f"Found {len(jsonl_files)} metadata files (.jsonl)")

        for key in npy_files:
            size_mb = size_map.get(key, 0) / (1024 * 1024)
            parsed = _parse_worker_npy(key)
            tag = f"worker {parsed[0]}/{parsed[1]}" if parsed else "unknown worker"
            print(f"  [{tag}] {key} ({size_mb:.1f} MB)")

    if not npy_files:
        print("[FATAL] No feature files found!")
        sys.exit(1)
    if not jsonl_files:
        print("[FATAL] No metadata files found!")
        sys.exit(1)

    return npy_files, jsonl_files


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Download all feature files
# ═══════════════════════════════════════════════════════════════════════════════

def download_all_files(r2: R2Client, npy_keys: List[str], jsonl_keys: List[str],
                       reporter: StatusReporter) -> Tuple[List[Path], List[Path]]:
    """Download all feature and metadata files from R2.

    Uses parallel downloads (4 threads) when there are 50+ files.
    """
    print(f"\n{'='*80}")
    print("STEP 2: Downloading feature files from R2")
    print(f"{'='*80}")

    download_dir = WORK_DIR / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)

    all_keys = [(k, "npy") for k in npy_keys] + [(k, "jsonl") for k in jsonl_keys]
    total_files = len(all_keys)
    use_parallel = total_files >= 50

    if use_parallel:
        print(f"  Using parallel downloads (4 threads) for {total_files} files")

    def _download_one(key_type):
        key, ftype = key_type
        filename = key.split("/")[-1]
        local_path = download_dir / filename
        if local_path.exists():
            return local_path, True  # cached
        success = r2.download_file(key, str(local_path))
        if not success:
            print(f"  [ERROR] Failed to download {key}")
            return None, False
        return local_path, False

    npy_paths = []
    jsonl_paths = []
    downloaded = [0]

    if use_parallel:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=4) as pool:
            future_map = {pool.submit(_download_one, kt): kt for kt in all_keys}
            for future in as_completed(future_map):
                key, ftype = future_map[future]
                try:
                    local_path, cached = future.result()
                    if local_path is None:
                        sys.exit(1)
                    if ftype == "npy":
                        npy_paths.append(local_path)
                    else:
                        jsonl_paths.append(local_path)
                except Exception as e:
                    print(f"  [ERROR] Download exception for {key}: {e}")
                    sys.exit(1)

                downloaded[0] += 1
                if downloaded[0] % 50 == 0:
                    pct = int(downloaded[0] / total_files * 100)
                    reporter.report("download", f"{downloaded[0]}/{total_files} files", pct)
                    print(f"  Downloaded {downloaded[0]}/{total_files} files ({pct}%)")
    else:
        for key, ftype in all_keys:
            local_path, cached = _download_one((key, ftype))
            if local_path is None:
                sys.exit(1)
            if ftype == "npy":
                npy_paths.append(local_path)
            else:
                jsonl_paths.append(local_path)
            downloaded[0] += 1
            if not cached:
                pct = int(downloaded[0] / total_files * 100)
                reporter.report("download", f"Downloading ({downloaded[0]}/{total_files})", pct)

    # Sort by filename to ensure deterministic merge order
    npy_paths.sort(key=lambda p: p.name)
    jsonl_paths.sort(key=lambda p: p.name)

    reporter.report("download", f"Downloaded {total_files} files", 100)
    print(f"\n  Total: {len(npy_paths)} .npy files, {len(jsonl_paths)} .jsonl files")
    return npy_paths, jsonl_paths


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Merge features and metadata
# ═══════════════════════════════════════════════════════════════════════════════

def merge_features_and_metadata(npy_paths: List[Path], jsonl_paths: List[Path],
                                 reporter: StatusReporter) -> Tuple[Path, Path, int]:
    """
    Merge all worker feature files into one features.npy and metadata.json.

    Each worker's .npy is shape (N_worker, 8448).
    Each worker's .jsonl has lines: {"panoid": "...", "lat": ..., "lng": ..., "feature_index": ...}

    The merged metadata.json maps global index → {lat, lng} (same format as build_megaloc_index.py).
    """
    print(f"\n{'='*80}")
    print("STEP 3: Merging features and metadata")
    print(f"{'='*80}")

    # First pass: count total rows
    total_rows = 0
    worker_shapes = []
    for i, npy_path in enumerate(npy_paths):
        mmap = np.load(str(npy_path), mmap_mode='r')
        shape = mmap.shape
        worker_shapes.append(shape)
        total_rows += shape[0]
        print(f"  Worker file {npy_path.name}: {shape[0]:,} vectors")
        del mmap

    print(f"\nTotal vectors to merge: {total_rows:,}")
    total_size_gb = total_rows * FEATURE_DIM * 4 / (1024 ** 3)
    per_worker_gb = total_size_gb / len(npy_paths) if npy_paths else 0
    print(f"Merged file size: {total_size_gb:.2f} GB")
    print(f"Per-worker avg:   {per_worker_gb:.2f} GB")

    # Disk check — using streaming-delete strategy:
    # features.bin is a sparse file (blocks allocated only as written), and each
    # source .npy is deleted immediately after its data is copied in.  Peak extra
    # disk needed is therefore ~1 worker file, not the full merged size.
    import shutil
    disk_usage = shutil.disk_usage(str(WORK_DIR))
    free_gb = disk_usage.free / (1024 ** 3)
    needed_gb = per_worker_gb + 5  # 1 worker buffer + 5 GB headroom
    print(f"Disk free: {free_gb:.1f} GB  (need ~{needed_gb:.1f} GB with streaming-delete)")
    if free_gb < needed_gb:
        print(f"[FATAL] Not enough free disk even for one worker slice "
              f"({needed_gb:.1f} GB needed, {free_gb:.1f} GB free). "
              f"Provision at least {total_size_gb + per_worker_gb + 20:.0f} GB total disk.")
        sys.exit(1)

    # Create merged memmap as raw binary (no .npy header — avoids all np.lib.format issues)
    merged_features_path = WORK_DIR / "features.bin"
    print(f"\nCreating merged memmap: {merged_features_path}")
    reporter.report("merge", "Creating merged feature file", 0)

    merged = np.memmap(
        str(merged_features_path), dtype='float32', mode='w+',
        shape=(total_rows, FEATURE_DIM)
    )

    # Copy features
    write_offset = 0
    for i, npy_path in enumerate(npy_paths):
        pct = int(i / len(npy_paths) * 80)
        reporter.report("merge", f"Copying {npy_path.name}", pct)

        src = np.load(str(npy_path), mmap_mode='r')
        n_rows = src.shape[0]

        # Copy in chunks to avoid memory issues
        chunk_size = 50000
        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            merged[write_offset + start: write_offset + end] = src[start:end]

        write_offset += n_rows
        del src
        gc.collect()

        # Delete the source file now that it's safely in features.bin.
        # features.bin is a sparse file so its blocks were only allocated as
        # we wrote them — freeing the source keeps peak disk near the input size.
        worker_size_gb = n_rows * FEATURE_DIM * 4 / (1024 ** 3)
        try:
            npy_path.unlink()
            print(f"  [DISK] Freed {worker_size_gb:.2f} GB — deleted {npy_path.name}")
        except OSError as e:
            print(f"  [WARN] Could not delete {npy_path.name}: {e}")

        merged.flush()

    merged.flush()
    del merged
    gc.collect()
    print(f"  Merged features written: {total_rows:,} vectors")

    # Merge metadata
    print("\nMerging metadata...")
    reporter.report("merge", "Merging metadata", 85)

    global_metadata = {}
    global_index = 0

    for jsonl_path in jsonl_paths:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    global_metadata[str(global_index)] = {
                        'lat': float(entry.get('lat', 0)),
                        'lng': float(entry.get('lng', 0)),
                    }
                    global_index += 1
                except (json.JSONDecodeError, KeyError, ValueError):
                    # Still increment to keep alignment with features
                    global_metadata[str(global_index)] = {'lat': 0.0, 'lng': 0.0}
                    global_index += 1

    metadata_path = WORK_DIR / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(global_metadata, f)

    print(f"  Metadata entries: {len(global_metadata):,}")

    # Validate alignment
    if global_index != total_rows:
        print(f"[WARN] Metadata count ({global_index}) != feature count ({total_rows})")
        print(f"  Using min of both: {min(global_index, total_rows)}")
        total_rows = min(global_index, total_rows)

    reporter.report("merge", f"Merged {total_rows:,} vectors + metadata", 100)
    return merged_features_path, metadata_path, total_rows


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Build FAISS Index
# ═══════════════════════════════════════════════════════════════════════════════

def build_faiss_index(features_path: Path, n_vectors: int,
                      reporter: StatusReporter) -> Path:
    """
    Build FAISS IVFPQ index from merged features.

    Uses Inner Product metric (MegaLoc vectors are normalized).
    Reads features via direct file I/O to avoid mmap issues on large datasets.
    """
    import faiss

    print(f"\n{'='*80}")
    print("STEP 4: Building FAISS index")
    print(f"{'='*80}")
    print(f"Vectors: {n_vectors:,}, Dimension: {FEATURE_DIM}")
    print(f"Index type: {INDEX_TYPE}, nlist={NLIST}, m={M}, nbits={NBITS}")

    reporter.report("index", "Preparing index", 0)

    row_bytes = FEATURE_DIM * 4  # float32

    # Memory-map the raw binary features file (no .npy header to parse)
    features_mmap = np.memmap(
        str(features_path), dtype='float32', mode='r',
        shape=(n_vectors, FEATURE_DIM)
    )

    def read_features_slice(start_row, end_row):
        """Read a slice of features from the memmap."""
        return np.array(features_mmap[start_row:end_row], dtype=np.float32)

    def read_features_by_indices(indices):
        """Read specific rows by index from the memmap."""
        return np.array(features_mmap[indices], dtype=np.float32)

    # Validate M divides FEATURE_DIM
    m = M
    if FEATURE_DIM % m != 0:
        # Find nearest smaller M that divides FEATURE_DIM
        for candidate in range(m, 0, -1):
            if FEATURE_DIM % candidate == 0:
                m = candidate
                break
        print(f"  Auto-adjusted M: {M} -> {m} (must divide {FEATURE_DIM})")

    # Auto-adjust nlist if needed
    nlist = NLIST
    max_nlist = n_vectors // 39
    if nlist > max_nlist:
        nlist = max(64, 2 ** int(np.log2(max_nlist)))
        print(f"  Auto-adjusted nlist: {NLIST} -> {nlist}")

    # Create index with Inner Product metric
    metric = faiss.METRIC_INNER_PRODUCT

    if INDEX_TYPE == "ivfpq":
        quantizer = faiss.IndexFlatIP(FEATURE_DIM)
        index = faiss.IndexIVFPQ(quantizer, FEATURE_DIM, nlist, m, NBITS, metric)
        index.cp.niter = NITER
        index.cp.verbose = True  # Log clustering progress
        print(f"  Index: IVFPQ, nlist={nlist}, m={m}, nbits={NBITS}, niter={NITER}")
    elif INDEX_TYPE == "pq":
        index = faiss.IndexPQ(FEATURE_DIM, m, NBITS, metric)
        print(f"  Index: PQ, m={m}, nbits={NBITS}")
    else:
        raise ValueError(f"Unknown INDEX_TYPE: {INDEX_TYPE}")

    # Enable FAISS verbose output for training progress
    index.verbose = True

    # Training — cap samples to avoid OOM/segfault on large datasets
    # PQ only needs ~100K-200K samples for stable codebook training
    MAX_TRAIN = 200_000
    train_samples = min(n_vectors, TRAIN_SAMPLES, MAX_TRAIN)

    # Extra safety: cap based on memory (keep training data under 4 GB)
    max_by_mem = (4 * 1024**3) // (FEATURE_DIM * 4)  # 4 GB / (dim * sizeof(float32))
    if train_samples > max_by_mem:
        train_samples = int(max_by_mem)
        print(f"  Capped training samples to {train_samples:,} (4 GB memory limit)")

    print(f"\nSampling {train_samples:,} vectors for training...")
    reporter.report("index", f"Sampling {train_samples:,} training vectors", 5)

    rng = np.random.default_rng(42)
    if train_samples >= n_vectors:
        train_indices = np.arange(n_vectors)
    else:
        train_indices = np.sort(rng.choice(n_vectors, size=train_samples, replace=False))

    print("Reading training vectors from disk...")
    reporter.report("index", "Reading training data", 10)
    train_data = np.ascontiguousarray(read_features_by_indices(train_indices), dtype=np.float32)
    del train_indices
    gc.collect()

    print(f"Training data loaded: {train_data.nbytes / (1024 ** 2):.0f} MB")
    sys.stdout.flush()

    print(f"Training index ({INDEX_TYPE}) — this may take a while...")
    print(f"  {train_samples:,} vectors x {FEATURE_DIM} dim, {m} sub-quantizers")
    sys.stdout.flush()
    reporter.report("index", f"Training {INDEX_TYPE} index ({train_samples:,} vectors)", 15)

    train_start = time.time()
    index.train(train_data)
    train_elapsed = time.time() - train_start

    del train_data
    gc.collect()
    print(f"Training complete in {train_elapsed:.1f}s ({train_elapsed/60:.1f} min)")
    sys.stdout.flush()
    reporter.report("index", f"Training done ({train_elapsed/60:.1f} min)", 20)

    # Adding vectors
    add_batch_size = 10_000
    total_batches = (n_vectors + add_batch_size - 1) // add_batch_size
    print(f"\nAdding {n_vectors:,} vectors in batches of {add_batch_size:,} ({total_batches} batches)...")
    reporter.report("index", "Adding vectors", 25)
    add_start = time.time()

    for batch_idx, start in enumerate(range(0, n_vectors, add_batch_size)):
        end = min(start + add_batch_size, n_vectors)
        batch = read_features_slice(start, end)
        index.add(batch)
        del batch

        # Log every batch
        elapsed = time.time() - add_start
        pct = 25 + int((batch_idx + 1) / total_batches * 65)
        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == total_batches:
            rate = end / elapsed if elapsed > 0 else 0
            eta = (n_vectors - end) / rate if rate > 0 else 0
            print(f"  Batch {batch_idx+1}/{total_batches}: {end:,}/{n_vectors:,} vectors "
                  f"({elapsed:.0f}s elapsed, {rate:.0f} vec/s, ETA {eta:.0f}s)")
            sys.stdout.flush()
            reporter.report("index", f"Added {end:,}/{n_vectors:,} vectors (ETA {eta/60:.1f}m)", pct)

        if (batch_idx + 1) % 10 == 0:
            gc.collect()

    print(f"Index built with {index.ntotal} vectors")

    # Save index
    index_path = WORK_DIR / "megaloc.index"
    print(f"\nSaving index to {index_path}...")
    reporter.report("index", "Saving index file", 92)
    faiss.write_index(index, str(index_path))

    index_size_mb = os.path.getsize(str(index_path)) / (1024 * 1024)
    raw_size_mb = n_vectors * FEATURE_DIM * 4 / (1024 * 1024)
    print(f"  Index size: {index_size_mb:.2f} MB")
    print(f"  Raw features size: {raw_size_mb:.2f} MB")
    print(f"  Compression ratio: {raw_size_mb / index_size_mb:.1f}x")

    # Save config
    config = {
        'n_vectors': n_vectors,
        'dimension': FEATURE_DIM,
        'index_type': INDEX_TYPE,
        'nlist': nlist,
        'm': m,
        'nbits': NBITS,
        'index_file': 'megaloc.index',
        'metadata_file': 'metadata.json',
        'features_file': 'features.bin',
    }
    config_path = WORK_DIR / "config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    print(f"  Config saved to {config_path}")

    reporter.report("index", f"Index built: {index.ntotal:,} vectors, {index_size_mb:.0f} MB", 95)
    return index_path


# ═══════════════════════════════════════════════════════════════════════════════
# Step 5: Upload results to R2
# ═══════════════════════════════════════════════════════════════════════════════

def upload_results(r2: R2Client, features_prefix: str, reporter: StatusReporter):
    """Upload megaloc.index, metadata.json, and config.json to R2."""
    print(f"\n{'='*80}")
    print("STEP 5: Uploading results to R2")
    print(f"{'='*80}")

    upload_prefix = f"Index/{'/'.join(features_prefix.rstrip('/').split('/')[1:])}"

    files_to_upload = [
        (WORK_DIR / "megaloc.index", f"{upload_prefix}/megaloc.index"),
        (WORK_DIR / "metadata.json", f"{upload_prefix}/metadata.json"),
        (WORK_DIR / "config.json", f"{upload_prefix}/config.json"),
    ]

    for local_path, r2_key in files_to_upload:
        if not local_path.exists():
            print(f"  [SKIP] {local_path.name} not found")
            continue

        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"  Uploading {local_path.name} ({size_mb:.1f} MB) -> {r2_key}")
        reporter.report("upload", f"Uploading {local_path.name}", 95)

        # Retry indefinitely for the index file (it's precious)
        max_attempts = 10 if "index" in local_path.name else 3
        for attempt in range(1, max_attempts + 1):
            success = r2.upload_file(str(local_path), r2_key)
            if success:
                break
            print(f"  [RETRY] Upload failed, attempt {attempt}/{max_attempts}")
            time.sleep(min(60, 2 ** attempt))
        else:
            print(f"  [ERROR] Failed to upload {local_path.name} after {max_attempts} attempts")

    reporter.report("upload", "All files uploaded", 100)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 6: Upload logs & Self-Destruct
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_instance_id(r2: R2Client, city_name: str) -> str:
    """Detect our instance ID from R2 or env."""
    instance_id = os.environ.get("INSTANCE_ID", "")
    if instance_id:
        return instance_id

    # Try R2 flat lookup
    try:
        data = r2.download_json(f"Status/INDEX_{city_name}_lookup.json")
        if data and 'instance_id' in data:
            return str(data['instance_id'])
    except Exception:
        pass

    # Fallback: vastai CLI
    api_key = os.environ.get("VAST_API_KEY", "")
    if api_key:
        try:
            result = subprocess.run(
                ["vastai", "--api-key", api_key, "show", "instances", "--raw"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                instances = json.loads(result.stdout)
                if len(instances) == 1:
                    return str(instances[0].get("id", ""))
        except Exception:
            pass

    return ""


    # Log uploads to R2 removed — logs stay local for SSH debugging


def _cleanup_status(r2: R2Client, city_name: str, instance_id: str):
    """Delete the status and lookup files from R2 after completion."""
    status_key = f"Status/INDEX_{city_name}_{instance_id}.json"
    lookup_key = f"Status/INDEX_{city_name}_lookup.json"
    for key in [status_key, lookup_key]:
        try:
            r2.delete_file(key)
            print(f"[CLEANUP] Deleted {key}")
        except Exception as e:
            print(f"[WARN] Failed to delete {key}: {e}")


def self_destruct(r2: R2Client, city_name: str, instance_id: str):
    """Destroy this instance via vastai CLI."""
    api_key = os.environ.get("VAST_API_KEY", "")
    if not api_key:
        print("[WARN] No VAST_API_KEY — cannot self-destruct")
        return

    if not instance_id:
        instance_id = _detect_instance_id(r2, city_name)
    if not instance_id:
        print("[WARN] Could not detect instance ID — cannot self-destruct")
        return

    print(f"\n[SELF-DESTRUCT] Destroying instance {instance_id}...")
    for attempt in range(3):
        try:
            result = subprocess.run(
                ["vastai", "--api-key", api_key, "destroy", "instance", instance_id],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                print(f"[SELF-DESTRUCT] Instance {instance_id} destroyed.")
                return
            print(f"[SELF-DESTRUCT] Attempt {attempt+1} failed: {result.stderr}")
        except Exception as e:
            print(f"[SELF-DESTRUCT] Attempt {attempt+1} error: {e}")
        time.sleep(2 ** attempt)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

class _LogTee:
    """Tee stdout/stderr to a log file while preserving console output."""

    def __init__(self, log_path: Path):
        self.log_file = open(str(log_path), 'w', encoding='utf-8')
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def start(self):
        sys.stdout = self._TeeStream(self.stdout, self.log_file)
        sys.stderr = self._TeeStream(self.stderr, self.log_file)

    def stop(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.log_file.close()

    class _TeeStream:
        def __init__(self, original, log_file):
            self.original = original
            self.log_file = log_file

        def write(self, data):
            self.original.write(data)
            try:
                self.log_file.write(data)
                self.log_file.flush()
            except Exception:
                pass

        def flush(self):
            self.original.flush()
            try:
                self.log_file.flush()
            except Exception:
                pass


def _clean_work_dir():
    """Delete all files in WORK_DIR to free disk for the next city."""
    import shutil
    for child in WORK_DIR.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        except Exception as e:
            print(f"  [WARN] Could not delete {child}: {e}")


def _build_one_city(r2: R2Client, features_prefix: str, city_name: str,
                    instance_id: str, city_idx: int, total_cities: int):
    """Build index for a single city. Returns True on success."""
    tag = f"[{city_idx}/{total_cities}]"
    print(f"\n{'='*80}")
    print(f"  {tag} BUILDING INDEX: {city_name}")
    print(f"  Features: {features_prefix}")
    print(f"{'='*80}")

    reporter = StatusReporter(r2, city_name, instance_id)
    # Update status key to use batch-aware format
    reporter.status_key = f"Status/INDEX_{instance_id}.json"
    reporter.report("init", f"{tag} Starting {city_name}", 0)

    try:
        # Step 1: Discover files
        npy_keys, jsonl_keys = discover_feature_files(r2, features_prefix)

        # Step 2: Download
        npy_paths, jsonl_paths = download_all_files(r2, npy_keys, jsonl_keys, reporter)

        # Step 3: Merge
        features_path, metadata_path, total_vectors = merge_features_and_metadata(
            npy_paths, jsonl_paths, reporter
        )

        # Step 4: Build FAISS index
        index_path = build_faiss_index(features_path, total_vectors, reporter)

        # Step 5: Upload
        upload_results(r2, features_prefix, reporter)

        reporter.report("done", f"{tag} {city_name}: {total_vectors:,} vectors", 100,
                        "CITY_DONE")
        print(f"\n  {tag} DONE — {city_name}: {total_vectors:,} vectors")
        return True

    except Exception as e:
        reporter.report("done", f"{tag} {city_name} FAILED: {e}", 0, "CITY_FAILED")
        print(f"\n  {tag} FAILED — {city_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Clean up work dir for next city
        print(f"  {tag} Cleaning work dir for next city...")
        _clean_work_dir()
        WORK_DIR.mkdir(parents=True, exist_ok=True)


def main():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    log_path = WORK_DIR / "builder.log"
    tee = _LogTee(log_path)
    tee.start()

    print("=" * 80)
    print("  VPS BUILDER PIPELINE — Feature Merge + FAISS Index")
    print("=" * 80)

    # Init R2 and detect instance ID
    r2 = R2Client()

    # ── Batch mode: BATCH_CITIES env var (base64-encoded JSON list) ──
    batch_raw = os.environ.get("BATCH_CITIES", "")
    if batch_raw:
        try:
            import base64
            decoded = base64.b64decode(batch_raw).decode()
            cities = json.loads(decoded)
        except Exception as e:
            # Fallback: try plain JSON (for manual/testing use)
            try:
                cities = json.loads(batch_raw)
            except json.JSONDecodeError as e2:
                print(f"[FATAL] Invalid BATCH_CITIES (tried base64 and plain JSON): {e2}")
                sys.exit(1)
    else:
        # Single-city fallback (legacy)
        cities = [{
            "features_prefix": get_env("FEATURES_BUCKET_PREFIX"),
            "city_name": get_env("CITY_NAME"),
        }]

    # Detect instance ID using first city
    first_city = cities[0]["city_name"]
    instance_id = _detect_instance_id(r2, first_city)
    print(f"Instance ID: {instance_id or '(unknown)'}")
    print(f"Work dir: {WORK_DIR}")
    print(f"CPU cores: {os.cpu_count()} (OMP_NUM_THREADS={_ncpu})")
    print(f"Cities to build: {len(cities)}")
    for i, c in enumerate(cities, 1):
        print(f"  {i}. {c['city_name']} — {c['features_prefix']}")

    # Write batch status so monitor knows what's happening
    batch_reporter = StatusReporter(r2, first_city, instance_id)
    batch_reporter.status_key = f"Status/INDEX_{instance_id}.json"
    batch_reporter.report("init", f"Batch: {len(cities)} cities", 0)

    succeeded = []
    failed = []

    for i, city_info in enumerate(cities, 1):
        fp = city_info["features_prefix"]
        cn = city_info["city_name"]

        ok = _build_one_city(r2, fp, cn, instance_id, i, len(cities))
        if ok:
            succeeded.append(cn)
        else:
            failed.append(cn)

    # Final batch status
    summary = f"{len(succeeded)}/{len(cities)} cities built"
    if failed:
        summary += f" (failed: {', '.join(failed)})"

    if failed and not succeeded:
        batch_reporter.report("done", summary, 100, "FAILED")
    else:
        batch_reporter.report("done", summary, 100, "COMPLETED")

    print(f"\n{'='*80}")
    print(f"  BATCH COMPLETE — {summary}")
    print(f"{'='*80}")

    tee.stop()

    # Wait for monitor to pick up status, then clean up
    print("[INFO] Waiting 30s for monitor to see final status...")
    time.sleep(30)

    # Clean up status files
    for key in [
        f"Status/INDEX_{instance_id}.json",
        f"Status/INDEX_{first_city}_lookup.json",
    ]:
        try:
            r2.delete_file(key)
            print(f"[CLEANUP] Deleted {key}")
        except Exception:
            pass

    # Self-destruct
    self_destruct(r2, first_city, instance_id)


if __name__ == "__main__":
    main()
