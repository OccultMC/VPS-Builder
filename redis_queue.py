"""
Redis Build Queue (Upstash)

Manages a shared city queue for distributed index building.
Multiple builder workers pull cities from the same queue.

Redis key schema:
    build:{job}:todo     → List   [city_0001, city_0002, ...]
    build:{job}:active   → Hash   {city_id: "worker_id|timestamp"}
    build:{job}:done     → Set    {city_0001, city_0003, ...}
    build:{job}:failed   → Set    {city_0002, ...}
    build:{job}:meta     → Hash   {total_cities, created_at}
    build:{job}:cmap     → Hash   {city_id: "features_prefix|city_name"}
    build:{job}:ws:{wid} → Hash   {status, city_id, cities_done, current_city, ts}
"""
import time
import logging
import threading
from typing import List, Dict, Optional

from upstash_redis import Redis

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL = 120  # 2 minutes between heartbeats
STALE_TIMEOUT = 600       # 10 minutes without heartbeat = dead worker


class _HeartbeatThread:
    """Background thread that pings Redis every HEARTBEAT_INTERVAL seconds."""

    def __init__(self, bq, job, worker_id, city_id):
        self._bq = bq
        self._job = job
        self._worker_id = worker_id
        self._city_id = city_id
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        while not self._stop.wait(HEARTBEAT_INTERVAL):
            try:
                self._bq.heartbeat(self._job, self._worker_id, self._city_id)
            except Exception as e:
                logger.warning(f"Heartbeat failed: {e}")


class BuildQueue:
    """Redis-backed work queue for distributed index building."""

    def __init__(self, redis_url: str, redis_token: str):
        self.redis = Redis(url=redis_url, token=redis_token)

    # ── Key helpers ───────────────────────────────────────────────────────

    def _todo_key(self, job: str) -> str:
        return f"build:{job}:todo"

    def _active_key(self, job: str) -> str:
        return f"build:{job}:active"

    def _done_key(self, job: str) -> str:
        return f"build:{job}:done"

    def _failed_key(self, job: str) -> str:
        return f"build:{job}:failed"

    def _meta_key(self, job: str) -> str:
        return f"build:{job}:meta"

    def _cmap_key(self, job: str) -> str:
        return f"build:{job}:cmap"

    def _worker_key(self, job: str, worker_id: str) -> str:
        return f"build:{job}:ws:{worker_id}"

    # ── Job lifecycle ─────────────────────────────────────────────────────

    def init_job(self, job: str, cities: List[Dict]):
        """
        Initialize a build job with a list of cities.

        cities: [{"features_prefix": "...", "city_name": "..."}, ...]
        """
        todo_key = self._todo_key(job)
        active_key = self._active_key(job)
        done_key = self._done_key(job)
        failed_key = self._failed_key(job)
        meta_key = self._meta_key(job)
        cmap_key = self._cmap_key(job)

        # Clean previous job data
        self.redis.delete(todo_key, active_key, done_key, failed_key, meta_key, cmap_key)

        # Build city IDs and metadata map
        city_ids = []
        cmap = {}
        for i, city in enumerate(cities, 1):
            city_id = f"city_{i:04d}"
            city_ids.append(city_id)
            cmap[city_id] = f"{city['features_prefix']}|{city['city_name']}"

        # Push all cities to todo
        if city_ids:
            self.redis.rpush(todo_key, *city_ids)

        # Store city metadata map
        if cmap:
            self.redis.hset(cmap_key, values=cmap)

        # Set job metadata
        self.redis.hset(meta_key, values={
            "total_cities": str(len(cities)),
            "created_at": str(time.time()),
        })

        logger.info(f"Initialized build job {job}: {len(cities)} cities")

    def claim_city(self, job: str, worker_id: str) -> Optional[Dict]:
        """
        Atomically pop a city from the todo list and mark it active.

        Returns {"city_id": ..., "features_prefix": ..., "city_name": ...}
        or None if queue is empty.
        """
        todo_key = self._todo_key(job)
        active_key = self._active_key(job)

        city_id = self.redis.lpop(todo_key)
        if city_id is None:
            return None

        # Mark as active
        self.redis.hset(active_key, city_id, f"{worker_id}|{time.time()}")

        # Get city metadata
        meta = self.get_city_meta(job, city_id)
        if meta:
            meta["city_id"] = city_id
            return meta

        return {"city_id": city_id, "features_prefix": "", "city_name": city_id}

    def complete_city(self, job: str, city_id: str, worker_id: str):
        """Move a city from active to done."""
        self.redis.hdel(self._active_key(job), city_id)
        self.redis.sadd(self._done_key(job), city_id)
        logger.info(f"Completed {city_id} (worker {worker_id})")

    def fail_city(self, job: str, city_id: str, worker_id: str, error: str):
        """Move a failed city to the failed set (don't retry index builds)."""
        self.redis.hdel(self._active_key(job), city_id)
        self.redis.sadd(self._failed_key(job), city_id)
        logger.warning(f"Failed {city_id} (worker {worker_id}): {error}")

    def heartbeat(self, job: str, worker_id: str, city_id: str):
        """Update timestamp for an active city (proves worker is alive)."""
        self.redis.hset(
            self._active_key(job), city_id, f"{worker_id}|{time.time()}"
        )

    def heartbeat_loop(self, job: str, worker_id: str, city_id: str):
        """Return a HeartbeatThread that pings Redis every 2 min while building.

        Usage:
            hb = bq.heartbeat_loop(job, worker_id, city_id)
            hb.start()
            ... do work ...
            hb.stop()
        """
        return _HeartbeatThread(self, job, worker_id, city_id)

    # ── Stale recovery ────────────────────────────────────────────────────

    def reclaim_stale(self, job: str, timeout: int = STALE_TIMEOUT) -> List[str]:
        """Move cities stuck in active for too long back to todo."""
        active_key = self._active_key(job)
        todo_key = self._todo_key(job)

        all_active = self.redis.hgetall(active_key)
        if not all_active:
            return []

        now = time.time()
        reclaimed = []

        for city_id, value in all_active.items():
            try:
                parts = value.rsplit("|", 1)
                claimed_at = float(parts[-1])
            except (ValueError, IndexError):
                claimed_at = 0.0

            if now - claimed_at > timeout:
                self.redis.hdel(active_key, city_id)
                self.redis.rpush(todo_key, city_id)
                reclaimed.append(city_id)
                worker = parts[0] if len(parts) > 1 else "unknown"
                logger.warning(
                    f"Reclaimed stale {city_id} "
                    f"(was held by {worker} for {now - claimed_at:.0f}s)"
                )

        return reclaimed

    # ── Queries ────────────────────────────────────────────────────────────

    def get_city_meta(self, job: str, city_id: str) -> Optional[Dict]:
        """Get features_prefix and city_name for a city_id."""
        val = self.redis.hget(self._cmap_key(job), city_id)
        if not val:
            return None
        parts = val.split("|", 1)
        if len(parts) < 2:
            return None
        return {"features_prefix": parts[0], "city_name": parts[1]}

    def get_progress(self, job: str) -> Dict:
        """Return current job progress."""
        meta = self.redis.hgetall(self._meta_key(job)) or {}
        todo = self.redis.llen(self._todo_key(job)) or 0
        active = self.redis.hlen(self._active_key(job)) or 0
        done = self.redis.scard(self._done_key(job)) or 0
        failed = self.redis.scard(self._failed_key(job)) or 0

        return {
            "total_cities": int(meta.get("total_cities", "0")),
            "todo": todo,
            "active": active,
            "done": done,
            "failed": failed,
        }

    def is_complete(self, job: str) -> bool:
        """True if nothing left in todo or active."""
        todo = self.redis.llen(self._todo_key(job)) or 0
        active = self.redis.hlen(self._active_key(job)) or 0
        return todo == 0 and active == 0

    # ── Worker status ─────────────────────────────────────────────────────

    def report_worker(self, job: str, worker_id: str, status: str,
                      city_id: str = "", cities_done: int = 0,
                      current_city: str = ""):
        """Write per-worker status for monitoring."""
        key = self._worker_key(job, worker_id)
        self.redis.hset(key, values={
            "s": status,
            "cid": city_id,
            "cd": str(cities_done),
            "cc": current_city,
            "ts": f"{time.time():.1f}",
        })

    def get_all_workers(self, job: str) -> Dict[str, Dict]:
        """Get status of all workers for this job."""
        prefix = f"build:{job}:ws:"
        result = {}
        cursor = 0
        while True:
            cursor, keys = self.redis.scan(cursor, match=f"{prefix}*", count=100)
            for key in keys:
                worker_id = key[len(prefix):]
                data = self.redis.hgetall(key)
                if data:
                    result[worker_id] = {
                        "status": data.get("s", "UNKNOWN"),
                        "city_id": data.get("cid", ""),
                        "cities_done": int(data.get("cd", "0")),
                        "current_city": data.get("cc", ""),
                        "ts": float(data.get("ts", "0")),
                    }
            if cursor == 0:
                break
        return result

    # ── Cleanup ────────────────────────────────────────────────────────────

    def cleanup(self, job: str):
        """Delete all Redis keys for a completed job."""
        prefix = f"build:{job}:ws:"
        cursor = 0
        while True:
            cursor, keys = self.redis.scan(cursor, match=f"{prefix}*", count=100)
            if keys:
                self.redis.delete(*keys)
            if cursor == 0:
                break

        self.redis.delete(
            self._todo_key(job),
            self._active_key(job),
            self._done_key(job),
            self._failed_key(job),
            self._meta_key(job),
            self._cmap_key(job),
        )
        logger.info(f"Cleaned up build job {job}")
