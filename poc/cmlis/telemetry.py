"""Telemetry collection.

Samples `numastat` (remote NUMA traffic) and basic per-CPU utilization via
psutil. On non-Linux systems, only psutil metrics are available.

Linux-only extras: `perf stat` (LLC cache misses) and `mpstat` (iowait).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field


@dataclass
class TelemetrySample:
    t: float
    cpu_percent: list[float]
    numa_local_mb: dict[int, float] = field(default_factory=dict)
    numa_remote_mb: dict[int, float] = field(default_factory=dict)
    perf_cache_misses: int | None = None  # LLC-load-misses count
    perf_cache_refs: int | None = None  # cache-references count
    perf_llc_miss_rate: float | None = None  # misses/refs ratio
    mpstat_iowait_pct: float | None = None  # mean iowait across CPUs


@dataclass
class TelemetrySummary:
    duration_s: float
    samples: int
    mean_cpu_percent: float
    remote_numa_fraction: float | None  # None if unavailable
    mean_llc_miss_rate: float | None = None  # mean across samples
    peak_iowait_pct: float | None = None  # max iowait sample
    perf_available: bool = False
    mpstat_available: bool = False

    def as_dict(self) -> dict:
        return {
            "duration_s": self.duration_s,
            "samples": self.samples,
            "mean_cpu_percent": self.mean_cpu_percent,
            "remote_numa_fraction": self.remote_numa_fraction,
            "mean_llc_miss_rate": self.mean_llc_miss_rate,
            "peak_iowait_pct": self.peak_iowait_pct,
            "perf_available": self.perf_available,
            "mpstat_available": self.mpstat_available,
        }


def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _read_numastat() -> tuple[dict[int, float], dict[int, float]]:
    """Return (local_mb_per_node, remote_mb_per_node)."""
    if not _have("numastat"):
        return {}, {}
    try:
        r = subprocess.run(["numastat"], capture_output=True, text=True, timeout=5)
    except (subprocess.SubprocessError, OSError):
        return {}, {}

    local: dict[int, float] = {}
    remote: dict[int, float] = {}
    for line in r.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        key = parts[0]
        if key == "numa_hit":
            for i, v in enumerate(parts[1:]):
                local[i] = float(v)
        elif key == "numa_miss":
            for i, v in enumerate(parts[1:]):
                remote[i] = float(v)
    return local, remote


def _read_perf_stat(pid: int | None) -> tuple[int | None, int | None]:
    """Run perf stat and return (llc_load_misses, cache_references).

    Linux-only. Returns (None, None) on other platforms, on permission errors,
    or if perf is not installed.
    """
    if sys.platform != "linux":
        return None, None
    if not _have("perf"):
        return None, None
    try:
        cmd = ["perf", "stat", "-e", "cache-misses,cache-references,LLC-load-misses"]
        if pid is not None:
            cmd += ["-p", str(pid)]
        cmd += ["sleep", "0.4"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        # perf writes its output to stderr
        output = r.stdout + r.stderr
    except Exception:
        return None, None

    llc_load_misses: int | None = None
    cache_references: int | None = None

    for line in output.splitlines():
        # Lines look like: "     1,234,567      cache-misses"
        parts = line.split()
        if len(parts) < 2:
            continue
        # Strip commas from the number
        raw_num = parts[0].replace(",", "")
        try:
            count = int(raw_num)
        except ValueError:
            continue
        label = parts[1].lower()
        if "llc-load-misses" in label or label == "llc-load-misses":
            llc_load_misses = count
        elif label in ("cache-misses", "cache-misses:u", "cache-misses:k"):
            # Only use cache-misses if llc-load-misses is not available
            if llc_load_misses is None:
                llc_load_misses = count
        elif "cache-references" in label:
            cache_references = count

    return llc_load_misses, cache_references


def _read_mpstat() -> float | None:
    """Run mpstat and return mean iowait% across all CPUs from the Average block.

    Linux-only. Returns None on other platforms or if mpstat is not installed.
    """
    if sys.platform != "linux":
        return None
    if not _have("mpstat"):
        return None
    try:
        r = subprocess.run(
            ["mpstat", "-P", "ALL", "1", "1"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = r.stdout
    except Exception:
        return None

    # Find the Average: section and parse %iowait column
    lines = output.splitlines()

    # Locate the header line under Average: to find column index
    iowait_col: int | None = None
    iowait_values: list[float] = []
    in_average = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        parts = stripped.split()
        if not parts:
            continue

        # Header line contains "%iowait" — look for it near "Average:"
        if "Average:" in parts[0] and "%iowait" in stripped:
            # This is the header row for Average block
            in_average = True
            try:
                iowait_col = parts.index("%iowait")
            except ValueError:
                return None
            continue

        if in_average and iowait_col is not None and parts[0] == "Average:":
            # Data rows: Average:  cpu_id  ...  %iowait  ...
            # Skip the "all" row to get per-CPU lines, or use all rows
            if len(parts) > iowait_col:
                try:
                    iowait_values.append(float(parts[iowait_col]))
                except ValueError:
                    continue

    if not iowait_values:
        return None
    return sum(iowait_values) / len(iowait_values)


class TelemetryCollector:
    """Background sampler. Start/stop around an inference run."""

    def __init__(self, interval_s: float = 0.5, pid: int | None = None):
        self.interval_s = interval_s
        self._pid = pid
        self.samples: list[TelemetrySample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t_start: float | None = None

    def _loop(self) -> None:
        import psutil

        while not self._stop.is_set():
            local, remote = _read_numastat()
            llc_misses, cache_refs = _read_perf_stat(self._pid)
            iowait = _read_mpstat()

            llc_miss_rate: float | None = None
            if llc_misses is not None and cache_refs is not None and cache_refs > 0:
                llc_miss_rate = llc_misses / cache_refs

            self.samples.append(
                TelemetrySample(
                    t=time.perf_counter(),
                    cpu_percent=psutil.cpu_percent(interval=None, percpu=True),
                    numa_local_mb=local,
                    numa_remote_mb=remote,
                    perf_cache_misses=llc_misses,
                    perf_cache_refs=cache_refs,
                    perf_llc_miss_rate=llc_miss_rate,
                    mpstat_iowait_pct=iowait,
                )
            )
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self.samples.clear()
        self._stop.clear()
        self._t_start = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> TelemetrySummary:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        duration = time.perf_counter() - (self._t_start or time.perf_counter())

        cpu_avg = 0.0
        if self.samples:
            flat = [v for s in self.samples for v in s.cpu_percent]
            cpu_avg = sum(flat) / len(flat) if flat else 0.0

        remote_frac: float | None = None
        if len(self.samples) >= 2:
            first, last = self.samples[0], self.samples[-1]
            if first.numa_local_mb and last.numa_local_mb:
                d_local = sum(
                    last.numa_local_mb.get(k, 0) - first.numa_local_mb.get(k, 0) for k in last.numa_local_mb
                )
                d_remote = sum(
                    last.numa_remote_mb.get(k, 0) - first.numa_remote_mb.get(k, 0)
                    for k in last.numa_remote_mb
                )
                total = d_local + d_remote
                if total > 0:
                    remote_frac = d_remote / total

        # LLC miss rate mean
        miss_rates = [s.perf_llc_miss_rate for s in self.samples if s.perf_llc_miss_rate is not None]
        mean_llc_miss_rate = sum(miss_rates) / len(miss_rates) if miss_rates else None

        # Peak iowait
        iowait_vals = [s.mpstat_iowait_pct for s in self.samples if s.mpstat_iowait_pct is not None]
        peak_iowait_pct = max(iowait_vals) if iowait_vals else None

        # Availability flags
        perf_available = any(s.perf_cache_misses is not None for s in self.samples)
        mpstat_available = any(s.mpstat_iowait_pct is not None for s in self.samples)

        return TelemetrySummary(
            duration_s=duration,
            samples=len(self.samples),
            mean_cpu_percent=cpu_avg,
            remote_numa_fraction=remote_frac,
            mean_llc_miss_rate=mean_llc_miss_rate,
            peak_iowait_pct=peak_iowait_pct,
            perf_available=perf_available,
            mpstat_available=mpstat_available,
        )
