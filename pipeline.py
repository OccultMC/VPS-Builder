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

# GPU settings — applies when faiss-gpu sees a CUDA device.
# USE_GPU=0 forces CPU even if a GPU is present (useful for debug).
# GPU_USE_FLOAT16=1 stores PQ codebook + IVF lists in fp16 (≈1/2 VRAM, no
# meaningful recall hit for 8-bit PQ codes).
# TRAIN_SAMPLES_GPU caps training-set rows when running on GPU. The full
# training matrix lives in VRAM during k-means; 8448-dim × 4 bytes ≈ 33 KB
# per row, so 500K rows ≈ 16.5 GB and fits a 24 GB card with headroom.
# Drop to 300K for 12 GB cards (≈10 GB), 800K for 48 GB.
USE_GPU = os.environ.get("USE_GPU", "1") == "1"
GPU_USE_FLOAT16 = os.environ.get("GPU_USE_FLOAT16", "1") == "1"
TRAIN_SAMPLES_GPU = int(os.environ.get("TRAIN_SAMPLES_GPU", "500000"))


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


def discover_feature_files(r2: R2Client, features_prefix: str) -> Tuple[List[str], List[str], Dict[str, int]]:
    """
    List all .npy and .jsonl files under the features prefix.

    Supports two naming conventions:
    - Chunk mode: {city}_chunk_{NNNN}.npy / Metadata_{city}_chunk_{NNNN}.jsonl
    - Worker mode (legacy): {city}_{worker}.{total}.npy / Metadata_{city}_{worker}.{total}.jsonl

    Files are returned as PAIRED lists sorted by index — only chunks/workers
    that have BOTH .npy AND .jsonl are included, to prevent metadata misalignment.
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

        # Match .jsonl files by chunk number
        jsonl_by_chunk: Dict[int, str] = {}
        for key in raw_jsonl:
            chunk_num = _parse_chunk_jsonl(key)
            if chunk_num is None:
                print(f"  [WARN] Unrecognised .jsonl filename (skipping): {key}")
                continue
            jsonl_by_chunk[chunk_num] = key

        # Only include chunks that have BOTH .npy AND .jsonl
        paired_chunks = sorted(set(npy_by_chunk.keys()) & set(jsonl_by_chunk.keys()))
        npy_files = [npy_by_chunk[i] for i in paired_chunks]
        jsonl_files = [jsonl_by_chunk[i] for i in paired_chunks]

        missing_jsonl = sorted(set(npy_by_chunk.keys()) - set(jsonl_by_chunk.keys()))
        missing_npy = sorted(set(jsonl_by_chunk.keys()) - set(npy_by_chunk.keys()))
        if missing_jsonl:
            n = len(missing_jsonl)
            print(f"[WARN] Skipping {n} chunks with missing .jsonl (would cause misalignment)")
            if n <= 20:
                print(f"  Chunks without metadata: {missing_jsonl}")
        if missing_npy:
            n = len(missing_npy)
            print(f"[WARN] Skipping {n} chunks with missing .npy")
            if n <= 20:
                print(f"  Chunks without features: {missing_npy}")

        print(f"\nFound {len(npy_files)} paired feature+metadata files")

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

        jsonl_by_worker: Dict[int, str] = {}
        for key in raw_jsonl:
            parsed = _parse_worker_jsonl(key)
            if parsed is None:
                print(f"  [WARN] Unrecognised .jsonl filename (skipping): {key}")
                continue
            w_idx, _ = parsed
            jsonl_by_worker[w_idx] = key

        # Only include workers that have BOTH .npy AND .jsonl
        paired_workers = sorted(set(npy_by_worker.keys()) & set(jsonl_by_worker.keys()))
        npy_files = [npy_by_worker[i] for i in paired_workers]
        jsonl_files = [jsonl_by_worker[i] for i in paired_workers]

        missing_jsonl = sorted(set(npy_by_worker.keys()) - set(jsonl_by_worker.keys()))
        if missing_jsonl:
            print(f"[WARN] Skipping workers with missing .jsonl (would cause misalignment): {missing_jsonl}")

        print(f"\nFound {len(npy_files)} paired feature+metadata files")

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

    return npy_files, jsonl_files, size_map


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Download all feature files
# ═══════════════════════════════════════════════════════════════════════════════

def download_and_merge_streaming(r2: R2Client, npy_keys: List[str], jsonl_keys: List[str],
                                  size_map: Dict[str, int], reporter: StatusReporter
                                  ) -> Tuple[Path, List[Path], int, List[int]]:
    """Pipelined download + memcpy into pre-allocated features.bin.

    Replaces the old two-step (download_all_files → merge feature copy) with
    a single streaming pass:

      * pre-compute per-chunk row count + global offset from R2 file sizes
        (no extra HEAD calls — list_objects_v2 already gave us sizes)
      * pre-allocate features.bin sparse to total_rows × 8448 × 4 bytes
      * N download workers each: download NPY + JSONL → memcpy NPY data into
        features.bin at known offset → delete source NPY
      * by the time downloads finish, features merge is also finished

    Tunables (env vars, all optional):
      R2_DL_PARALLEL              file workers (default 32, was 16)
      R2_DL_MULTIPART_CONCURRENCY streams per file (default 8, was 4)
      R2_DL_MULTIPART_CHUNK_MB    multipart chunk size MB (default 64, was 32)
      R2_POOL_SIZE                connection pool (default 256, was 64)

    Returns (features_path, jsonl_paths_in_input_order, total_rows, rows_per_chunk).
    """
    print(f"\n{'='*80}")
    print("STEP 2+3: Download + merge (streaming, pipelined)")
    print(f"{'='*80}")

    if len(npy_keys) != len(jsonl_keys):
        print(f"[FATAL] npy/jsonl key count mismatch")
        sys.exit(1)

    download_dir = WORK_DIR / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)

    # Compute row counts + offsets from sizes. NPY layout for shape (N, 8448) f32:
    #   small fixed header (~128 B) + N × 8448 × 4 bytes of data.
    # Determine header overhead from any chunk's full size — chunks share the
    # same descriptor and their headers are byte-identical for the same shape.
    row_size_bytes = FEATURE_DIM * 4
    rows_per_chunk: List[int] = []
    offsets: List[int] = []
    cum = 0
    header_bytes = 128  # standard NPY header for these shapes
    for k in npy_keys:
        sz = size_map.get(k, 0)
        if sz <= 0:
            print(f"[FATAL] missing size for {k}")
            sys.exit(1)
        n = (sz - header_bytes) // row_size_bytes
        if n <= 0 or (sz - header_bytes) % row_size_bytes != 0:
            # Header isn't 128 bytes — fall back to mmap'ing the file post-download
            # by marking with sentinel. But warn now.
            print(f"[WARN] non-standard NPY header for {k} (size={sz}); "
                  f"will fix offset post-download")
            n = -1
        rows_per_chunk.append(n)
        offsets.append(cum)
        if n > 0:
            cum += n
    if any(n < 0 for n in rows_per_chunk):
        # Resolve shapes the slow way: pre-fetch one chunk to learn the actual
        # header length, then re-derive.
        print("[INFO] resolving NPY header length from first chunk...")
        sample_key = npy_keys[0]
        sample_local = download_dir / "_header_probe.npy"
        r2.download_file(sample_key, str(sample_local))
        sample_arr = np.load(str(sample_local), mmap_mode='r')
        sample_n = sample_arr.shape[0]
        sample_size = sample_local.stat().st_size
        header_bytes = sample_size - sample_n * row_size_bytes
        del sample_arr
        sample_local.unlink()
        print(f"[INFO] resolved header={header_bytes} bytes, sample shape=({sample_n}, {FEATURE_DIM})")
        # re-derive
        rows_per_chunk = []
        offsets = []
        cum = 0
        for k in npy_keys:
            sz = size_map[k]
            n = (sz - header_bytes) // row_size_bytes
            rows_per_chunk.append(n)
            offsets.append(cum)
            cum += n
    total_rows = cum

    total_size_gb = total_rows * FEATURE_DIM * 4 / (1024 ** 3)
    print(f"  Chunks: {len(npy_keys)}  total rows: {total_rows:,}  merged size: {total_size_gb:.2f} GB")

    # Pre-allocate features.bin (sparse)
    features_path = WORK_DIR / "features.bin"
    print(f"  Allocating sparse features.bin ({total_size_gb:.2f} GB)...")
    with open(features_path, 'wb') as f:
        f.seek(total_rows * row_size_bytes - 1)
        f.write(b'\0')

    dl_workers = int(os.environ.get("R2_DL_PARALLEL", "32"))
    print(f"  Pipelined download: {dl_workers} file workers (multipart streams set in TransferConfig)")

    jsonl_paths_out: List[Optional[Path]] = [None] * len(npy_keys)
    completed = [0]

    def _process_chunk(i: int):
        npy_key = npy_keys[i]
        jsonl_key = jsonl_keys[i]
        offset_rows = offsets[i]
        expected_rows = rows_per_chunk[i]

        npy_local = download_dir / npy_key.split("/")[-1]
        jsonl_local = download_dir / jsonl_key.split("/")[-1]

        # JSONL is tiny — single GET via download_file is fine.
        if not jsonl_local.exists():
            ok = r2.download_file(jsonl_key, str(jsonl_local))
            if not ok:
                raise RuntimeError(f"jsonl download failed: {jsonl_key}")

        # NPY: download → memcpy at offset → delete
        if not npy_local.exists():
            ok = r2.download_file(npy_key, str(npy_local))
            if not ok:
                raise RuntimeError(f"npy download failed: {npy_key}")

        src = np.load(str(npy_local), mmap_mode='r')
        actual_n = src.shape[0]
        if actual_n != expected_rows:
            raise RuntimeError(
                f"row count mismatch chunk {i}: expected {expected_rows}, "
                f"got {actual_n} — offsets would misalign. Re-derive header_bytes."
            )

        # Write src bytes to features.bin at offset_rows × row_size_bytes
        write_bytes = row_size_bytes * actual_n
        with open(features_path, 'r+b') as out:
            out.seek(offset_rows * row_size_bytes)
            # tobytes copies — for huge arrays we could use src.view(np.uint8).tofile,
            # but tobytes is fine and explicit.
            out.write(src.tobytes())

        del src
        try:
            npy_local.unlink()
        except OSError:
            pass

        jsonl_paths_out[i] = jsonl_local
        n_done = completed[0] + 1
        completed[0] = n_done
        if n_done % 20 == 0 or n_done == len(npy_keys):
            pct = int(n_done / len(npy_keys) * 100)
            reporter.report("download+merge", f"{n_done}/{len(npy_keys)} chunks", pct)
            print(f"  done {n_done}/{len(npy_keys)} ({pct}%)  "
                  f"({write_bytes / (1024**2):.0f} MB written at offset "
                  f"{offset_rows * row_size_bytes / (1024**3):.2f} GB)")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=dl_workers) as pool:
        futs = [pool.submit(_process_chunk, i) for i in range(len(npy_keys))]
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                print(f"[FATAL] chunk worker failed: {e}")
                sys.exit(1)

    print(f"  Streaming download+merge complete: {total_rows:,} vectors -> {features_path}")
    reporter.report("download+merge", f"all {len(npy_keys)} chunks merged", 100)

    return features_path, [jsonl_paths_out[i] for i in range(len(npy_keys))], total_rows, rows_per_chunk


def download_all_files(r2: R2Client, npy_keys: List[str], jsonl_keys: List[str],
                       reporter: StatusReporter) -> Tuple[List[Path], List[Path]]:
    """Download all feature and metadata files from R2.

    Uses parallel downloads (4 threads) when there are 50+ files.
    IMPORTANT: npy_keys[i] and jsonl_keys[i] are PAIRED (same chunk/worker).
    The returned lists preserve this pairing order.
    """
    print(f"\n{'='*80}")
    print("STEP 2: Downloading feature files from R2")
    print(f"{'='*80}")

    if len(npy_keys) != len(jsonl_keys):
        print(f"[FATAL] npy/jsonl key count mismatch: {len(npy_keys)} npy vs {len(jsonl_keys)} jsonl")
        sys.exit(1)

    download_dir = WORK_DIR / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)

    all_keys = [(k, "npy") for k in npy_keys] + [(k, "jsonl") for k in jsonl_keys]
    total_files = len(all_keys)
    # Always parallel — even small batches benefit. Each worker uses
    # multipart download (4 streams per file) so 16 outer workers ≈ 64
    # concurrent TCP streams to R2, well within Cloudflare's per-account
    # parallelism budget but kind to flaky residential vast hosts.
    dl_workers = int(os.environ.get("R2_DL_PARALLEL", "16"))
    use_parallel = total_files > 1
    if use_parallel:
        print(f"  Parallel download: {dl_workers} file workers × multipart "
              f"({total_files} files)")

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

    # Download all files (order doesn't matter, we rebuild from keys)
    downloaded_map: Dict[str, Path] = {}  # R2 key → local path
    downloaded_count = [0]

    if use_parallel:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=dl_workers) as pool:
            future_map = {pool.submit(_download_one, kt): kt for kt in all_keys}
            for future in as_completed(future_map):
                key, ftype = future_map[future]
                try:
                    local_path, cached = future.result()
                    if local_path is None:
                        sys.exit(1)
                    downloaded_map[key] = local_path
                except Exception as e:
                    print(f"  [ERROR] Download exception for {key}: {e}")
                    sys.exit(1)

                downloaded_count[0] += 1
                if downloaded_count[0] % 50 == 0:
                    pct = int(downloaded_count[0] / total_files * 100)
                    reporter.report("download", f"{downloaded_count[0]}/{total_files} files", pct)
                    print(f"  Downloaded {downloaded_count[0]}/{total_files} files ({pct}%)")
    else:
        for key, ftype in all_keys:
            local_path, cached = _download_one((key, ftype))
            if local_path is None:
                sys.exit(1)
            downloaded_map[key] = local_path
            downloaded_count[0] += 1
            if not cached:
                pct = int(downloaded_count[0] / total_files * 100)
                reporter.report("download", f"Downloading ({downloaded_count[0]}/{total_files})", pct)

    # Rebuild paired lists in the SAME order as the input keys
    # This preserves the pairing from discover_feature_files
    npy_paths = [downloaded_map[k] for k in npy_keys]
    jsonl_paths = [downloaded_map[k] for k in jsonl_keys]

    reporter.report("download", f"Downloaded {total_files} files", 100)
    print(f"\n  Total: {len(npy_paths)} .npy files, {len(jsonl_paths)} .jsonl files")
    return npy_paths, jsonl_paths


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Merge features and metadata
# ═══════════════════════════════════════════════════════════════════════════════

def _merge_metadata_only(jsonl_paths: List[Path], rows_per_chunk: List[int],
                          total_rows: int, reporter: StatusReporter) -> Path:
    """Build the merged metadata.json after features.bin is already populated
    by download_and_merge_streaming(). Iterates the JSONL files in chunk order
    so global indices line up with feature offsets exactly.
    """
    print(f"\n{'='*80}")
    print("STEP 3b: Merging metadata (features already streamed)")
    print(f"{'='*80}")
    reporter.report("merge", "Merging metadata", 80)

    global_metadata: Dict[str, Dict] = {}
    global_index = 0
    alignment_errors = 0
    for file_idx, jsonl_path in enumerate(jsonl_paths):
        expected_rows = rows_per_chunk[file_idx]
        file_meta_count = 0
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    global_metadata[str(global_index)] = {
                        'panoid': str(entry.get('panoid', '')),
                        'lat': float(entry.get('lat', 0)),
                        'lng': float(entry.get('lng', 0)),
                        'fi_local': int(entry.get('feature_index', 0)),
                    }
                except (json.JSONDecodeError, KeyError, ValueError):
                    global_metadata[str(global_index)] = {
                        'panoid': '', 'lat': 0.0, 'lng': 0.0, 'fi_local': 0,
                    }
                global_index += 1
                file_meta_count += 1
        if file_meta_count != expected_rows:
            print(f"[ERROR] {jsonl_path.name}: {file_meta_count} meta lines "
                  f"vs {expected_rows} feature rows")
            alignment_errors += 1

    if alignment_errors > 0:
        print(f"[FATAL] {alignment_errors} chunks have meta/feature mismatch")
        sys.exit(1)
    if global_index != total_rows:
        print(f"[FATAL] metadata count {global_index} != feature count {total_rows}")
        sys.exit(1)

    metadata_path = WORK_DIR / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(global_metadata, f)
    print(f"  Metadata entries: {len(global_metadata):,}")
    return metadata_path


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

    # Merge metadata — process in PAIRS with per-file count validation
    print("\nMerging metadata (paired with features)...")
    reporter.report("merge", "Merging metadata", 85)

    global_metadata = {}
    global_index = 0
    alignment_errors = 0

    for file_idx, (npy_path, jsonl_path) in enumerate(zip(
        # Re-derive npy_paths order from worker_shapes (same iteration order as feature copy)
        # We know npy_paths[i] had worker_shapes[i][0] rows
        npy_paths, jsonl_paths
    )):
        expected_rows = worker_shapes[file_idx][0]
        file_meta_count = 0

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    global_metadata[str(global_index)] = {
                        'panoid': str(entry.get('panoid', '')),
                        'lat': float(entry.get('lat', 0)),
                        'lng': float(entry.get('lng', 0)),
                        # Local feature index within the source chunk (0..N-1).
                        # Lets the server identify which directional view of the
                        # pano the match came from, useful for debug + re-fetch.
                        'fi_local': int(entry.get('feature_index', 0)),
                    }
                    global_index += 1
                    file_meta_count += 1
                except (json.JSONDecodeError, KeyError, ValueError):
                    # Still increment to keep alignment with features
                    global_metadata[str(global_index)] = {
                        'panoid': '',
                        'lat': 0.0,
                        'lng': 0.0,
                        'fi_local': 0,
                    }
                    global_index += 1
                    file_meta_count += 1

        if file_meta_count != expected_rows:
            print(f"[ERROR] File {file_idx} count mismatch: "
                  f"{npy_path.name} has {expected_rows} vectors but "
                  f"{jsonl_path.name} has {file_meta_count} metadata lines!")
            alignment_errors += 1

    if alignment_errors > 0:
        print(f"\n[FATAL] {alignment_errors} file(s) have mismatched vector/metadata counts!")
        print(f"  The index would be corrupted. Fix the source data and re-run.")
        sys.exit(1)

    metadata_path = WORK_DIR / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(global_metadata, f)

    print(f"  Metadata entries: {len(global_metadata):,}")

    # Final alignment check
    if global_index != total_rows:
        print(f"[FATAL] Metadata count ({global_index}) != feature count ({total_rows})")
        print(f"  This indicates a pairing bug. Cannot build a valid index.")
        sys.exit(1)

    reporter.report("merge", f"Merged {total_rows:,} vectors + metadata", 100)
    return merged_features_path, metadata_path, total_rows


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Build FAISS Index
# ═══════════════════════════════════════════════════════════════════════════════

def build_faiss_index(features_path: Path, n_vectors: int,
                      reporter: StatusReporter,
                      r2: "R2Client" = None,
                      features_prefix: str = None) -> Path:
    """
    Build FAISS IVFPQ index from merged features.

    Uses Inner Product metric (MegaLoc vectors are normalized).
    Reads features via direct file I/O to avoid mmap issues on large datasets.

    Optional `r2` + `features_prefix` enable in-build region polygon lookup
    (downloads Shapefiles/{Country}.geojson from R2 and writes
    region_polygon_simplified into config.json).
    """
    import faiss

    print(f"\n{'='*80}")
    print("STEP 4: Building FAISS index")
    print(f"{'='*80}")
    print(f"Vectors: {n_vectors:,}, Dimension: {FEATURE_DIM}")

    # ── Small-index auto-switch ────────────────────────────────────────────
    # Below SMALL_INDEX_THRESHOLD vectors, IVFPQ adds little over pure PQ
    # but introduces two failure modes: k-means under-training (per-cell
    # samples thin out) and search-time recall loss from cell-scan misses.
    # Pure IndexPQ at this scale gives near-equivalent recall with smaller
    # build time, no convergence cliff, and the SAME M=256 sub-quantizer
    # vocabulary as the big indexes (so cross-city recall comparisons
    # stay valid). On-disk size: ~256 bytes per vector regardless of N.
    SMALL_INDEX_THRESHOLD = int(os.environ.get("SMALL_INDEX_THRESHOLD", "300000"))
    AUTO_TUNE_TYPE = os.environ.get("AUTO_TUNE_TYPE", "1") == "1"
    effective_index_type = INDEX_TYPE
    if AUTO_TUNE_TYPE and INDEX_TYPE == "ivfpq" and n_vectors < SMALL_INDEX_THRESHOLD:
        print(f"  [auto-switch] N={n_vectors:,} < {SMALL_INDEX_THRESHOLD:,} "
              f"— overriding INDEX_TYPE: ivfpq -> pq "
              f"(IVF k-means would starve at this scale; pure PQ is faster + "
              f"avoids convergence issues, M={M} preserved)")
        effective_index_type = "pq"

    print(f"Index type: {effective_index_type}, nlist={NLIST}, m={M}, nbits={NBITS}")

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

    # ── Auto-tune index params for this city's vector count ────────────────
    # AUTO_TUNE=1 (default) overrides NLIST / NITER / TRAIN_SAMPLES with values
    # sized for the dataset. Set AUTO_TUNE=0 to use raw env values.
    import math as _math
    AUTO_TUNE = os.environ.get("AUTO_TUNE", "1") == "1"

    nlist = NLIST
    niter_eff = NITER
    train_target_eff = TRAIN_SAMPLES  # default; refined in train-cap section

    if AUTO_TUNE and effective_index_type == "ivfpq":
        # ── nlist target: 8·√N (recall-optimised, "overkill") ────────────────
        # The classic FAISS rule of thumb is 4·√N for balanced build/search.
        # We bias to 8·√N — finer cells, higher search precision when the
        # index is queried with a matching nprobe. Constraints:
        #   * floor at 256 (too few cells = scan-heavy queries)
        #   * cap by training budget so per-cell train samples stays above
        #     the FAISS recommended floor (PER_CELL_FLOOR below)
        #   * cap at n_vectors // 39 (FAISS hard requirement)
        TRAIN_PER_CELL_TARGET = 1024     # quality knob — more = better k-means convergence
        TRAIN_PER_CELL_FLOOR  = 64       # FAISS recommends ≥30; we stay safely above
        # Train cap matches the host-RAM size of the training matrix:
        # 1.5M × 8448 × 4 bytes ≈ 50 GB, fits a 128 GB host with headroom for
        # the 200 GB+ features.bin memmap that lives alongside it.
        TRAIN_HARD_MAX = 1_500_000

        # Round to NEAREST power of 2 (not floor) — favours the "overkill"
        # direction so 8·√1M = 8000 maps to 8192 instead of 4096.
        target_nlist = 8 * _math.sqrt(max(n_vectors, 1))
        auto = 2 ** round(_math.log2(max(target_nlist, 256)))
        # Cap by FAISS hard requirement (≥39 per cell)
        auto = min(auto, max(256, n_vectors // 39))
        # Cap by training budget: ensure we can hand FAISS at least
        # PER_CELL_FLOOR samples per cell without blowing past TRAIN_HARD_MAX.
        # Use FLOOR (not round) here — must NEVER exceed the floor.
        train_budget = min(n_vectors, TRAIN_HARD_MAX)
        max_nlist_by_train = max(256, train_budget // TRAIN_PER_CELL_FLOOR)
        if auto > max_nlist_by_train:
            new_auto = 2 ** int(_math.log2(max_nlist_by_train))
            print(f"  [auto-tune] nlist {auto} -> {new_auto} "
                  f"(would starve k-means: {train_budget // auto} per cell < "
                  f"{TRAIN_PER_CELL_FLOOR} floor)")
            auto = new_auto
        if auto != NLIST:
            print(f"  [auto-tune] nlist {NLIST} -> {auto} for {n_vectors:,} vectors "
                  f"(target=8·√N={target_nlist:.0f})")
            nlist = auto

        # ── niter: keep the user's explicit value if reasonable ─────────────
        # Per user policy this is a fixed setting; only override if someone
        # sets a wasteful 100+ default that doesn't improve convergence.
        if NITER >= 100:
            niter_eff = 50
            print(f"  [auto-tune] niter {NITER} -> {niter_eff} (sufficient convergence)")
    else:
        # Legacy fallback: raw env nlist with the old too-large guard
        max_nlist = n_vectors // 39
        if nlist > max_nlist:
            nlist = max(64, 2 ** int(np.log2(max_nlist)))
            print(f"  Auto-adjusted nlist: {NLIST} -> {nlist}")

    # Create index with Inner Product metric
    metric = faiss.METRIC_INNER_PRODUCT

    if effective_index_type == "ivfpq":
        quantizer = faiss.IndexFlatIP(FEATURE_DIM)
        index = faiss.IndexIVFPQ(quantizer, FEATURE_DIM, nlist, m, NBITS, metric)
        index.cp.niter = niter_eff
        index.cp.verbose = True  # Log clustering progress
        print(f"  Index: IVFPQ, nlist={nlist}, m={m}, nbits={NBITS}, niter={niter_eff}")
    elif effective_index_type == "pq":
        index = faiss.IndexPQ(FEATURE_DIM, m, NBITS, metric)
        print(f"  Index: PQ, m={m}, nbits={NBITS}")
    else:
        raise ValueError(f"Unknown effective_index_type: {effective_index_type}")

    # Enable FAISS verbose output for training progress
    index.verbose = True

    # ── GPU acceleration ────────────────────────────────────────────────────
    # Training (k-means on the IVF coarse quantizer + PQ codebook fitting) and
    # .add() (PQ encoding of every vector) are the two CPU-bound phases on
    # this pipeline. Both port cleanly to faiss-gpu via index_cpu_to_gpu when
    # the index type is GPU-supported (IndexFlat*, IndexIVF*PQ). IndexPQ has
    # no GPU implementation in faiss-gpu — we skip the attempt for it.
    # The final write_index() needs a CPU index, so we clone back before save.
    GPU_SUPPORTED_TYPES = {"ivfpq"}
    gpu_index = None
    gpu_resources = None
    # gpu_requested tracks user intent (USE_GPU=1 + GPU visible) regardless of
    # whether the clone actually succeeded. We use it to size the training
    # set: a host picked for its GPU is not necessarily a host with 32+ GB
    # of system RAM, so the smaller TRAIN_SAMPLES_GPU cap is safer for the
    # CPU fallback too.
    gpu_requested = False
    if USE_GPU:
        try:
            n_gpus = faiss.get_num_gpus()
        except Exception as e:
            n_gpus = 0
            print(f"  [GPU] faiss.get_num_gpus() failed: {e}")
        if n_gpus > 0:
            gpu_requested = True
            if effective_index_type not in GPU_SUPPORTED_TYPES:
                print(f"\n[GPU] {n_gpus} CUDA device(s) detected but index type "
                      f"'{effective_index_type}' has no faiss-gpu implementation. "
                      f"Running on CPU (only 'ivfpq' is GPU-accelerated; "
                      f"'pq' is fast enough on CPU at small N).")
            else:
                print(f"\n[GPU] {n_gpus} CUDA device(s) detected — building index "
                      f"directly on GPU (useFloat16={GPU_USE_FLOAT16})")
                reporter.report("index", "Constructing GPU index", 12)
                try:
                    gpu_resources = faiss.StandardGpuResources()

                    # Build GpuIndexIVFPQ DIRECTLY rather than cloning the
                    # CPU index. Reason: GpuClonerOptions has no field for
                    # `interleavedLayout` — it lives on GpuIndexIVFPQConfig,
                    # which the cloner constructs internally with defaults.
                    # Without interleaved layout, faiss-gpu's
                    # IVFPQ::isSupportedPQCodeLength(M) only accepts the
                    # standard set {1,2,3,4,8,12,16,20,24,28,32,40,48,56,64,
                    # 96}, so any M outside that (e.g. M=256 for higher
                    # recall) crashes verifyPQSettings_() and silently falls
                    # back to CPU — which on multi-million-vector indexes
                    # is days of swap-thrashing k-means. Direct construction
                    # lets us set the config field that actually matters.
                    config = faiss.GpuIndexIVFPQConfig()
                    config.device = 0
                    config.useFloat16LookupTables = GPU_USE_FLOAT16
                    config.interleavedLayout = True

                    gpu_index = faiss.GpuIndexIVFPQ(
                        gpu_resources, FEATURE_DIM, nlist, m, NBITS,
                        metric, config,
                    )
                    # Training params live on the GPU index itself, not on
                    # `index.cp` like the CPU variant.
                    gpu_index.cp.niter = niter_eff
                    gpu_index.cp.verbose = True
                    gpu_index.verbose = True
                    # Keep the CPU `index` allocated as the safety-net for
                    # the GPU-OOM fallback path below. It costs ~quantizer
                    # size in host RAM (negligible vs the train/add tensors)
                    # and means we don't have to reconstruct on failure.
                except Exception as e:
                    print(f"  [GPU] direct GpuIndexIVFPQ construction failed "
                          f"({type(e).__name__}: {e}) — falling back to CPU")
                    import traceback
                    traceback.print_exc()
                    gpu_index = None
        else:
            print("\n[GPU] USE_GPU=1 but no CUDA devices visible — running on CPU")

    # Training set sizing.
    # AUTO_TUNE: target = nlist × TRAIN_PER_CELL_TARGET (quality-biased),
    # capped at TRAIN_HARD_MAX (host-RAM limit) and N (can't sample more
    # than exist). This pairs with the nlist computation above so per-cell
    # training stays comfortably above FAISS's PER_CELL_FLOOR.
    if AUTO_TUNE and effective_index_type == "ivfpq":
        train_target_eff = min(
            nlist * 1024,         # 1024/cell — generous, stable k-means
            1_500_000,            # ~50 GB host RAM at 8448 dim fp32 (TRAIN_HARD_MAX)
            n_vectors,            # can't sample more than exist
        )
    train_cap = TRAIN_SAMPLES_GPU if gpu_requested else TRAIN_SAMPLES
    train_samples = min(n_vectors // 3 or n_vectors, train_target_eff, train_cap)

    if gpu_requested:
        train_bytes_gb = train_samples * FEATURE_DIM * 4 / (1024 ** 3)
        target = "GPU" if gpu_index is not None else "CPU (GPU-host fallback)"
        print(f"  [{target}] Training cap = {train_cap:,} rows "
              f"({train_bytes_gb:.1f} GB float32 input)")

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

    print(f"Training index ({effective_index_type}) — this may take a while...")
    print(f"  {train_samples:,} vectors x {FEATURE_DIM} dim, {m} sub-quantizers")
    sys.stdout.flush()
    reporter.report("index", f"Training {effective_index_type} index ({train_samples:,} vectors)", 15)

    # If GPU clone is set, train on it; the CPU `index` is updated implicitly
    # because faiss's GpuIndex wraps the same parameter buffer (after the
    # train completes we'll mirror back via index_gpu_to_cpu before save).
    train_target = gpu_index if gpu_index is not None else index

    train_start = time.time()
    try:
        train_target.train(train_data)
    except Exception as e:
        # Most likely failure: VRAM OOM on the k-means step. Fall back to CPU
        # using the SAME training set — re-sampling to a larger size on the
        # CPU path can OOM the host RAM (8448-dim × 4 bytes × N samples).
        if gpu_index is not None:
            print(f"  [GPU] train() failed ({e}) — falling back to CPU "
                  f"with the same {train_samples:,}-row training set. "
                  f"(Lower TRAIN_SAMPLES_GPU if this keeps happening.)")
            del gpu_index
            gpu_index = None
            gc.collect()
            index.train(train_data)
        else:
            raise
    train_elapsed = time.time() - train_start

    del train_data
    gc.collect()
    print(f"Training complete in {train_elapsed:.1f}s ({train_elapsed/60:.1f} min)")
    sys.stdout.flush()
    reporter.report("index", f"Training done ({train_elapsed/60:.1f} min)", 20)

    # Adding vectors. With GPU active, batches are pushed to VRAM, encoded,
    # and the resulting PQ codes append directly to the GPU index. Per-batch
    # transfer dominates at the same wall-clock cost as the CPU path BUT the
    # encoding itself is ~10-20× faster, which dominates for big indices.
    add_target = gpu_index if gpu_index is not None else index
    add_batch_size = 50_000 if gpu_index is not None else 10_000
    total_batches = (n_vectors + add_batch_size - 1) // add_batch_size
    print(f"\nAdding {n_vectors:,} vectors in batches of {add_batch_size:,} ({total_batches} batches)...")
    reporter.report("index", "Adding vectors", 25)
    add_start = time.time()

    for batch_idx, start in enumerate(range(0, n_vectors, add_batch_size)):
        end = min(start + add_batch_size, n_vectors)
        batch = read_features_slice(start, end)
        add_target.add(batch)
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

    final_target = gpu_index if gpu_index is not None else index
    # Capture gpu actually-used flag before we delete gpu_index for the save step
    gpu_used_for_build = gpu_index is not None
    print(f"Index built with {final_target.ntotal} vectors")

    # Move GPU index back to CPU so faiss.write_index can serialize it.
    if gpu_index is not None:
        print("\n[GPU] Moving index back to CPU for serialization...")
        reporter.report("index", "Cloning index GPU->CPU", 91)
        index = faiss.index_gpu_to_cpu(gpu_index)
        del gpu_index
        gpu_index = None
        # Release GPU memory before the (large) save step.
        if gpu_resources is not None:
            try:
                gpu_resources.noTempMemory()
            except Exception:
                pass
            gpu_resources = None
        gc.collect()

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

    # Save config — capture FINAL values actually used (post-auto-tune),
    # not the env defaults. Downstream search code reads this file to
    # configure the same index for queries.
    config = {
        'n_vectors': n_vectors,
        'dimension': FEATURE_DIM,
        'index_type': effective_index_type,
        'nlist': nlist,
        'm': m,
        'nbits': NBITS,
        'niter': niter_eff if effective_index_type == "ivfpq" else None,
        'metric': 'inner_product',
        'train_samples_used': train_samples,
        'train_data_mb': round(train_samples * FEATURE_DIM * 4 / (1024 ** 2)),
        'index_size_mb': round(index_size_mb, 2),
        'raw_size_mb': round(raw_size_mb, 2),
        'compression_ratio': round(raw_size_mb / index_size_mb, 1),
        'training_seconds': round(train_elapsed, 1),
        'gpu_used': gpu_used_for_build,
        'gpu_requested': gpu_requested,
        'auto_tuned': AUTO_TUNE,
        'index_file': 'megaloc.index',
        'metadata_file': 'metadata.json',
        'features_file': 'features.bin',
    }

    # ── Geo enrichment for the inference server's "Search this city" feature ──
    # Computes the city centroid (mean lat/lng) plus a small set of evenly-
    # spaced sample points. The sample points let the offline polygon
    # backfill (zelesis-inference/backfill_polygons.py) attribute this city
    # to its urban polygon without re-downloading metadata.json.
    try:
        metadata_path = WORK_DIR / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                md = json.load(f)
            items = list(md.values()) if isinstance(md, dict) else list(md)
            lats = [v.get("lat") for v in items if isinstance(v, dict) and isinstance(v.get("lat"), (int, float))]
            lngs = [v.get("lng") for v in items if isinstance(v, dict) and isinstance(v.get("lng"), (int, float))]
            if lats:
                config["center_lat"] = sum(lats) / len(lats)
                config["center_lng"] = sum(lngs) / len(lngs)
                config["center_n"] = len(lats)
            # 20 evenly-spaced sample (lat, lng) tuples. Enough variety for
            # the polygon backfill to find a hit even if the city's edge
            # bleeds across an admin boundary.
            sample_pairs = []
            step = max(1, len(items) // 20)
            for v in items[::step][:20]:
                if isinstance(v, dict):
                    lat, lng = v.get("lat"), v.get("lng")
                    if isinstance(lat, (int, float)) and isinstance(lng, (int, float)):
                        sample_pairs.append([lat, lng])
            if sample_pairs:
                config["sample_points"] = sample_pairs
            print(f"  Geo enrichment: center=({config.get('center_lat')}, {config.get('center_lng')}) "
                  f"sample_points={len(sample_pairs)}")
    except Exception as e:
        print(f"  Geo enrichment skipped: {e}")

    # ── Region polygon (downloaded from R2 Shapefiles/, resolved at build) ─────
    # The features_prefix looks like "Features/{COUNTRY}/{STATE}/{CITY}".
    # Country dictates which shapefile to grab; the resolver only emits the
    # simplified polygon (we explicitly skip the full one because it's 30×
    # larger and unused server-side). Doing this here removes the need to
    # ever run zelesis-inference/backfill_polygons.py.
    #
    # Keys in R2 (uploaded once by the local maintainer):
    #   AU/* -> Shapefiles/Australia.geojson
    #   US/* -> Shapefiles/United_States_Urban_By_County.geojson
    if r2 is None or not features_prefix:
        # Skip silently when called without R2 context (e.g. legacy code paths).
        pass
    try:
        if r2 is None or not features_prefix:
            raise RuntimeError("no r2/features_prefix")
        country = None
        if isinstance(features_prefix, str):
            parts = [p for p in features_prefix.strip("/").split("/") if p]
            if len(parts) >= 2 and parts[0] == "Features":
                country = parts[1]
        shape_keys = {
            "AU": "Shapefiles/Australia.geojson",
            "US": "Shapefiles/United_States_Urban_By_County.geojson",
        }
        shape_key = shape_keys.get(country)
        if shape_key and config.get("sample_points"):
            shape_local = WORK_DIR / Path(shape_key).name
            if not shape_local.exists():
                print(f"  Downloading shapefile {shape_key}...")
                r2.download_file(shape_key, str(shape_local))
            from region_attribution import RegionResolver
            us_path = str(shape_local) if country == "US" else "/__missing_us__"
            au_path = str(shape_local) if country == "AU" else "/__missing_au__"
            resolver = RegionResolver(us_path=us_path, au_path=au_path)
            pts = [tuple(p) for p in config["sample_points"]]
            match = resolver.find_union(pts)
            if match:
                config.update(match)
                print(f"  Region polygon: {match['region_name']} "
                      f"(simplified, source={match['region_source']})")
            else:
                print(f"  Region polygon: no match for {country} sample points")
        else:
            print(f"  Region polygon skipped (country={country!r})")
    except Exception as e:
        print(f"  Region polygon lookup skipped: {e}")

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

    # Map the features-prefix family to the matching Index family so that
    # provider-separated runs land in distinct buckets:
    #   Features/<...>        -> Index/<...>          (Google Street View)
    #   Features_Apple/<...>  -> Index_Apple/<...>    (Apple Look Around)
    # We replace just the *first* path segment (which is always Features*),
    # preserving any suffix (anything after "Features").
    parts = features_prefix.rstrip('/').split('/')
    first = parts[0] or 'Features'
    if first.startswith('Features'):
        index_first = 'Index' + first[len('Features'):]
    else:
        index_first = 'Index'
    upload_prefix = '/'.join([index_first] + parts[1:])

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
        # Step 1: Discover files (also returns size_map for offset pre-compute)
        npy_keys, jsonl_keys, size_map = discover_feature_files(r2, features_prefix)

        # Step 2+3 fused: pipelined download + memcpy into pre-allocated
        # features.bin. By the time the last byte is on disk, the merged
        # feature file is also done — no separate "STEP 3 merge features"
        # phase. Metadata.json is still merged in chunk order below.
        features_path, jsonl_paths, total_vectors, _rows_per_chunk = \
            download_and_merge_streaming(r2, npy_keys, jsonl_keys, size_map, reporter)

        # Step 3b: merge metadata only (features already merged above).
        # Reuse merge_features_and_metadata's metadata branch by passing in
        # the already-merged feature path so it skips the copy phase.
        metadata_path = _merge_metadata_only(
            jsonl_paths, _rows_per_chunk, total_vectors, reporter
        )

        # Step 4: Build FAISS index
        index_path = build_faiss_index(
            features_path, total_vectors, reporter,
            r2=r2, features_prefix=features_prefix,
        )

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


def _run_redis_queue(r2: R2Client, instance_id: str):
    """Redis queue mode: claim cities one at a time until queue is empty."""
    from redis_queue import BuildQueue

    redis_url = os.environ["REDIS_URL"]
    redis_token = os.environ["REDIS_TOKEN"]
    build_job = os.environ["BUILD_JOB"]

    bq = BuildQueue(redis_url, redis_token)

    print(f"Redis queue mode — job: {build_job}")
    progress = bq.get_progress(build_job)
    print(f"  Total cities: {progress['total_cities']}, "
          f"todo: {progress['todo']}, active: {progress['active']}, "
          f"done: {progress['done']}, failed: {progress['failed']}")

    worker_id = instance_id or f"worker_{int(time.time())}"
    succeeded = []
    failed = []

    # Reclaim any stale tasks from crashed workers before starting
    reclaimed = bq.reclaim_stale(build_job)
    if reclaimed:
        print(f"  Reclaimed {len(reclaimed)} stale cities: {reclaimed}")

    batch_reporter = StatusReporter(r2, "queue", instance_id)
    batch_reporter.status_key = f"Status/INDEX_{instance_id}.json"

    while True:
        city = bq.claim_city(build_job, worker_id)
        if city is None:
            print("\n[QUEUE] No more cities — queue empty")
            break

        city_id = city["city_id"]
        fp = city["features_prefix"]
        cn = city["city_name"]
        total_done = len(succeeded) + len(failed) + 1

        print(f"\n[QUEUE] Claimed {city_id}: {cn} ({fp})")
        bq.report_worker(build_job, worker_id, "BUILDING",
                         city_id=city_id, cities_done=len(succeeded),
                         current_city=cn)

        # Use progress for total count display
        prog = bq.get_progress(build_job)
        total_cities = prog["total_cities"]

        # Heartbeat thread keeps our active claim alive during long builds
        hb = bq.heartbeat_loop(build_job, worker_id, city_id)
        hb.start()
        ok = _build_one_city(r2, fp, cn, instance_id, total_done, total_cities)
        hb.stop()

        if ok:
            bq.complete_city(build_job, city_id, worker_id)
            succeeded.append(cn)
        else:
            bq.fail_city(build_job, city_id, worker_id, f"{cn} build failed")
            failed.append(cn)

        # Heartbeat / worker status update
        bq.report_worker(build_job, worker_id, "IDLE",
                         city_id=city_id, cities_done=len(succeeded),
                         current_city="")

    # Final summary
    summary = f"Worker {worker_id}: {len(succeeded)} built, {len(failed)} failed"
    bq.report_worker(build_job, worker_id, "DONE",
                     cities_done=len(succeeded), current_city="")

    batch_reporter.report("done", summary, 100, "COMPLETED")
    print(f"\n{'='*80}")
    print(f"  {summary}")
    print(f"{'='*80}")

    return succeeded, failed


def _run_batch(r2: R2Client, instance_id: str):
    """Batch mode: BATCH_CITIES env var or single-city fallback."""
    batch_raw = os.environ.get("BATCH_CITIES", "")
    if batch_raw:
        try:
            import base64
            decoded = base64.b64decode(batch_raw).decode()
            cities = json.loads(decoded)
        except Exception:
            try:
                cities = json.loads(batch_raw)
            except json.JSONDecodeError as e2:
                print(f"[FATAL] Invalid BATCH_CITIES (tried base64 and plain JSON): {e2}")
                sys.exit(1)
    else:
        cities = [{
            "features_prefix": get_env("FEATURES_BUCKET_PREFIX"),
            "city_name": get_env("CITY_NAME"),
        }]

    print(f"Batch mode — {len(cities)} cities")
    for i, c in enumerate(cities, 1):
        print(f"  {i}. {c['city_name']} — {c['features_prefix']}")

    batch_reporter = StatusReporter(r2, cities[0]["city_name"], instance_id)
    batch_reporter.status_key = f"Status/INDEX_{instance_id}.json"
    batch_reporter.report("init", f"Batch: {len(cities)} cities", 0)

    succeeded = []
    failed = []

    for i, city_info in enumerate(cities, 1):
        ok = _build_one_city(r2, city_info["features_prefix"],
                             city_info["city_name"], instance_id,
                             i, len(cities))
        if ok:
            succeeded.append(city_info["city_name"])
        else:
            failed.append(city_info["city_name"])

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

    return succeeded, failed


def main():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    log_path = WORK_DIR / "builder.log"
    tee = _LogTee(log_path)
    tee.start()

    print("=" * 80)
    print("  VPS BUILDER PIPELINE — Feature Merge + FAISS Index")
    print("=" * 80)

    r2 = R2Client()

    # Detect instance ID
    first_city = os.environ.get("CITY_NAME", "builder")
    instance_id = _detect_instance_id(r2, first_city)
    print(f"Instance ID: {instance_id or '(unknown)'}")
    print(f"Work dir: {WORK_DIR}")
    print(f"CPU cores: {os.cpu_count()} (OMP_NUM_THREADS={_ncpu})")

    # ── Choose mode: Redis queue vs batch/single ──
    use_redis = all(os.environ.get(k) for k in ("REDIS_URL", "REDIS_TOKEN", "BUILD_JOB"))

    if use_redis:
        succeeded, failed = _run_redis_queue(r2, instance_id)
    else:
        succeeded, failed = _run_batch(r2, instance_id)

    tee.stop()

    # Wait for monitor, then clean up
    print("[INFO] Waiting 30s for monitor to see final status...")
    time.sleep(30)

    for key in [
        f"Status/INDEX_{instance_id}.json",
        f"Status/INDEX_{first_city}_lookup.json",
    ]:
        try:
            r2.delete_file(key)
            print(f"[CLEANUP] Deleted {key}")
        except Exception:
            pass

    self_destruct(r2, first_city, instance_id)


if __name__ == "__main__":
    main()
