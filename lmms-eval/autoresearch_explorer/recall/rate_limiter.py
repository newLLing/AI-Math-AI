"""
Global Rate Limiter for External API Calls
============================================
Ensures all outbound HTTP requests (arXiv, CrossRef, OpenReview, etc.)
are globally serialized with a minimum interval, preventing 429 due to
concurrent / overlapping requests from the same IP.

Key features:
  - Process-wide singleton: all recall modules share one limiter
  - Respects Retry-After headers from 429 responses
  - Circuit breaker: pauses a domain after consecutive 429s
"""

import time
import threading
import logging
from typing import Optional, Dict


class GlobalRateLimiter:
    """
    Process-wide rate limiter for external academic APIs.
    """

    # Minimum interval between ANY two external HTTP requests (seconds)
    DEFAULT_MIN_INTERVAL = 10.0

    # Circuit breaker settings
    CB_FAIL_THRESHOLD = 5          # consecutive 429s before opening
    CB_COOLDOWN_SECONDS = 300      # 5 minutes pause after opening

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._lock = threading.Lock()
        self._last_request_time: float = 0.0
        self._domain_stats: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def wait(self, domain: str = "global", min_interval: Optional[float] = None):
        """
        Block until it is safe to issue the next HTTP request.

        Args:
            domain: API domain name (e.g. 'arxiv', 'crossref', 'openreview')
            min_interval: override the default minimum interval (seconds)
        """
        interval = min_interval if min_interval is not None else self.DEFAULT_MIN_INTERVAL

        with self._lock:
            # Circuit breaker check
            if self._is_circuit_open(domain):
                self.logger.warning(
                    f"[RateLimiter] Circuit OPEN for {domain}, "
                    f"skipping request for {self.CB_COOLDOWN_SECONDS}s"
                )
                raise CircuitBreakerOpen(domain)

            elapsed = time.time() - self._last_request_time
            if elapsed < interval:
                wait_time = interval - elapsed
                self.logger.info(
                    f"[RateLimiter] {domain}: waiting {wait_time:.1f}s "
                    f"(min_interval={interval:.1f}s)"
                )
                # Release lock while sleeping so other threads can plan
                self._lock.release()
                try:
                    time.sleep(wait_time)
                finally:
                    self._lock.acquire()

            self._last_request_time = time.time()

    def record_success(self, domain: str = "global"):
        """Call after a successful request to reset failure counters."""
        with self._lock:
            stats = self._domain_stats.get(domain)
            if stats and stats.get("consecutive_429", 0) > 0:
                self.logger.info(
                    f"[RateLimiter] {domain}: success, reset consecutive_429"
                )
                stats["consecutive_429"] = 0

    def record_429(self, domain: str = "global",
                   retry_after: Optional[float] = None) -> float:
        """
        Call after receiving a 429 response.

        Args:
            domain: API domain
            retry_after: value from Retry-After header (seconds)

        Returns:
            Recommended sleep time before next request.
        """
        with self._lock:
            stats = self._domain_stats.setdefault(domain, {
                "consecutive_429": 0,
                "last_429_time": 0.0,
                "circuit_open_until": 0.0,
            })
            stats["consecutive_429"] += 1
            stats["last_429_time"] = time.time()

            # Use Retry-After if provided, otherwise exponential backoff
            if retry_after is not None and retry_after > 0:
                wait = retry_after
                self.logger.warning(
                    f"[RateLimiter] {domain}: 429 with Retry-After={wait:.0f}s"
                )
            else:
                wait = min(120, 10 * (2 ** (stats["consecutive_429"] - 1)))
                self.logger.warning(
                    f"[RateLimiter] {domain}: 429 (no Retry-After), "
                    f"backoff={wait:.0f}s, consecutive={stats['consecutive_429']}"
                )

            # Circuit breaker
            if stats["consecutive_429"] >= self.CB_FAIL_THRESHOLD:
                stats["circuit_open_until"] = time.time() + self.CB_COOLDOWN_SECONDS
                self.logger.error(
                    f"[RateLimiter] {domain}: CIRCUIT OPENED for "
                    f"{self.CB_COOLDOWN_SECONDS}s after {stats['consecutive_429']} "
                    f"consecutive 429s"
                )

            return wait

    def get_retry_after_from_response(self, response) -> Optional[float]:
        """
        Extract Retry-After value from a requests.Response object.
        Handles both integer seconds and HTTP-date formats.
        """
        raw = response.headers.get("Retry-After") or response.headers.get("retry-after")
        if not raw:
            return None
        try:
            # Try integer seconds first
            return float(raw)
        except ValueError:
            # Could be HTTP-date; for simplicity, treat as unknown
            self.logger.debug(f"[RateLimiter] Retry-After is HTTP-date: {raw}")
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_circuit_open(self, domain: str) -> bool:
        stats = self._domain_stats.get(domain)
        if not stats:
            return False
        return time.time() < stats.get("circuit_open_until", 0.0)


class CircuitBreakerOpen(Exception):
    """Raised when the circuit breaker is open for a domain."""
    def __init__(self, domain: str):
        self.domain = domain
        super().__init__(f"Circuit breaker open for {domain}")


# ------------------------------------------------------------------------------
# Process-wide singleton instance
# ------------------------------------------------------------------------------

_limiter_instance: Optional[GlobalRateLimiter] = None
_limiter_lock = threading.Lock()


def get_rate_limiter(logger: Optional[logging.Logger] = None) -> GlobalRateLimiter:
    """Get the process-wide singleton GlobalRateLimiter."""
    global _limiter_instance
    if _limiter_instance is None:
        with _limiter_lock:
            if _limiter_instance is None:
                _limiter_instance = GlobalRateLimiter(logger=logger)
    return _limiter_instance
