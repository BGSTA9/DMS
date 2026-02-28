# =============================================================================
# core/thread_manager.py
#
# ThreadManager — centralized lifecycle management for the DMS application.
#
# Responsibilities:
#   • Register all stoppable components in one place
#   • Provide a single stop_all() call for clean shutdown
#   • Catch and log any shutdown errors so the app always exits cleanly
#   • Report startup timing for each component (useful for the report)
#
# Design pattern: Registry + Observer
#   Components register themselves. On shutdown, ThreadManager calls
#   stop() on each in reverse-registration order (LIFO — last started,
#   first stopped).
# =============================================================================

import time
from typing import Callable, List, Tuple
from core.logger import get_logger

log = get_logger(__name__)


class ThreadManager:
    """
    Centralized lifecycle registry for all stoppable DMS components.

    Usage:
        tm = ThreadManager()
        tm.register("DMSCore",   dms.start,  dms.stop)
        tm.register("UIManager", None,        ui.quit)
        tm.start_all()
        # ... main loop ...
        tm.stop_all()
    """

    def __init__(self):
        # List of (name, start_fn, stop_fn) tuples in registration order
        self._components: List[Tuple[str, Callable, Callable]] = []
        self._started:    List[str] = []
        self._t0 = time.time()

    def register(
        self,
        name:     str,
        start_fn: Callable = None,
        stop_fn:  Callable = None,
    ) -> None:
        """
        Register a component for lifecycle management.

        Args:
            name:     Human-readable component name (for logging)
            start_fn: Callable to start the component (or None)
            stop_fn:  Callable to stop/cleanup the component (or None)
        """
        self._components.append((name, start_fn, stop_fn))
        log.debug(f"Registered component: {name}")

    def start_all(self) -> None:
        """Start all registered components in registration order."""
        log.info("=" * 50)
        log.info("  Starting DMS Application")
        log.info("=" * 50)

        for name, start_fn, _ in self._components:
            if start_fn is not None:
                t0 = time.perf_counter()
                try:
                    start_fn()
                    elapsed = (time.perf_counter() - t0) * 1000
                    log.info(f"  ✓  {name:<25} started  ({elapsed:.0f}ms)")
                    self._started.append(name)
                except Exception as e:
                    log.error(f"  ✗  {name} failed to start: {e}", exc_info=True)
                    raise
            else:
                self._started.append(name)
                log.info(f"  ✓  {name:<25} registered (no start fn)")

        total = (time.time() - self._t0) * 1000
        log.info(f"  All components ready in {total:.0f}ms")
        log.info("=" * 50)

    def stop_all(self) -> None:
        """
        Stop all registered components in reverse order (LIFO).
        Errors during shutdown are logged but never re-raised.
        """
        log.info("Shutting down DMS Application …")

        # Reverse order — last started, first stopped
        for name, _, stop_fn in reversed(self._components):
            if stop_fn is not None:
                try:
                    stop_fn()
                    log.info(f"  ✓  {name} stopped.")
                except Exception as e:
                    log.error(f"  ✗  {name} shutdown error: {e}", exc_info=True)

        log.info("DMS Application shut down cleanly.")