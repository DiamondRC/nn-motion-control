"""
A tiny console spinner for long synchronous steps (rollouts, probes).

The spinner animates on a daemon thread while the caller's blocking work
runs on the main thread, then clears its line. Off a TTY it prints the
message once and animates nothing, so piped/CI output stays clean and
deterministic.
"""

from __future__ import annotations

import itertools
import sys
import threading
import time
from contextlib import contextmanager


@contextmanager
def spinner(message: str, interval: float = 0.1):
    """
    Animate 'message' with a spinner and a live elapsed timer until the
    block exits.

    Off a TTY it prints the message once (start) and its duration (end), no
    animation.
    """

    start = time.perf_counter()

    if not sys.stderr.isatty():
        sys.stderr.write(f"{message} ...\n")
        sys.stderr.flush()
        yield
        sys.stderr.write(f"  ({message}: {time.perf_counter() - start:.1f}s)\n")
        sys.stderr.flush()

        return

    stop = threading.Event()

    def _spin() -> None:
        for frame in itertools.cycle("|/-\\"):
            if stop.is_set():
                break
            elapsed = time.perf_counter() - start
            line = f"\r{frame} {message} ({elapsed:.1f}s)"
            sys.stderr.write(line)
            sys.stderr.flush()
            stop.wait(interval)
        sys.stderr.write("\r" + " " * (len(message) + 20) + "\r")
        sys.stderr.flush()

    thread = threading.Thread(target=_spin, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
