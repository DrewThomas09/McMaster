"""Tiny in-memory sliding-window rate limiter (per client key)."""

from __future__ import annotations

import threading
import time
from collections import deque


class RateLimiter:
    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def bucket(host: str) -> str:
        """IPv6 clients get a /64 bucket (one subscriber), IPv4 the address itself."""
        if ":" in host:
            try:
                import ipaddress

                ip = ipaddress.ip_address(host)
                if ip.version == 6 and ip.ipv4_mapped:  # "::ffff:1.2.3.4" behind a "::" listener
                    return str(ip.ipv4_mapped)
                return str(ipaddress.ip_network(f"{host}/64", strict=False))
            except ValueError:
                return host
        return host

    def allow(self, key: str, cost: int = 1) -> bool:
        if self.per_minute <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            q = self._hits.setdefault(key, deque())
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) + cost > self.per_minute:
                if not q:
                    del self._hits[key]
                return False
            q.extend([now] * max(1, cost))
            if len(self._hits) > 50_000:  # forget idle clients instead of growing forever
                for k in [k for k, d in self._hits.items() if not d or now - d[-1] > 60]:
                    del self._hits[k]
            return True
