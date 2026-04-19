"""Telemetry collection.

Samples NUMA locality, CPU utilization, cache behavior, and iowait around
an inference run. On Linux, NUMA and perf sampling can be scoped to a
specific process PID. On other systems, unavailable metrics are reported
explicitly instead of being folded into zero values.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field


@dataclass
class TelemetrySample:
    t: float
    pid: int | None
    cpu_percent: list[float]
    numa_local_mb: dict[int, float] = field(default_factory=dict)
    numa_remote_mb: dict[int, float] = field(default_factory=dict)
    numa_scope: str = "unavailable"
    perf_cache_misses: int | None = None
    perf_cache_refs: int | None = None
    perf_llc_miss_rate: float | None = None
    mpstat_iowait_pct: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class TelemetrySummary:
    duration_s: float
    samples: int
    mean_cpu_percent: float
    remote_numa_fraction: float | None
    numa_available: bool = False
    numa_scope: str = "unavailable"
    mean_llc_miss_rate: float | None = None
    peak_iowait_pct: float | None = None
    perf_available: bool = False
    mpstat_available: bool = False

    def as_dict(self) -> dict:
        return {
            "duration_s": self.duration_s,
            "samples": self.samples,
            "mean_cpu_percent": self.mean_cpu_percent,
            "remote_numa_fraction": self.remote_numa_fraction,
            "numa_available": self.numa_available,
            "numa_scope": self.numa_scope,
            "mean_llc_miss_rate": self.mean_llc_miss_rate,
            "peak_iowait_pct": self.peak_iowait_pct,
            "perf_available": self.perf_available,
            "mpstat_available": self.mpstat_available,
        }


def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _read_numastat(pid: int | None = None) -> tuple[dict[int, float], dict[int, float], str]:
    """Return (local_mb_per_node, remote_mb_per_node, scope)."""
    if sys.platform != "linux" or not _have("numastat"):
        return {}, {}, "unavailable"

    cmd = ["numastat"]
    scope = "system"
    if pid is not None:
        cmd += ["-p", str(pid)]
        scope = "process"

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    except (subprocess.SubprocessError, OSError):
        return {}, {}, "unavailable"

    local: dict[int, float] = {}
    remote: dict[int, float] = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        key = parts[0]
        if key == "numa_hit":
            for index, value in enumerate(parts[1:]):
                local[index] = float(value)
        elif key == "numa_miss":
            for index, value in enumerate(parts[1:]):
                remote[index] = float(value)
    if not local and not remote:
        return {}, {}, "unavailable"
    return local, remote, scope


def _read_perf_stat(pid: int | None) -> tuple[int | None, int | None]:
    """Run perf stat and return (llc_load_misses, cache_references)."""
    if sys.platform != "linux" or not _have("perf"):
        return None, None
    try:
        cmd = ["perf", "stat", "-e", "cache-misses,cache-references,LLC-load-misses"]
        if pid is not None:
            cmd += ["-p", str(pid)]
        cmd += ["sleep", "0.4"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        output = result.stdout + result.stderr
    except Exception:
        return None, None

    llc_load_misses: int | None = None
    cache_references: int | None = None
    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        raw_num = parts[0].replace(",", "")
        try:
            count = int(raw_num)
        except ValueError:
            continue
        label = parts[1].lower()
        if "llc-load-misses" in label or label == "llc-load-misses":
            llc_load_misses = count
        elif label in ("cache-misses", "cache-misses:u", "cache-misses:k"):
            if llc_load_misses is None:
                llc_load_misses = count
        elif "cache-references" in label:
            cache_references = count

    return llc_load_misses, cache_references


def _read_mpstat() -> float | None:
    """Run mpstat and return mean iowait% across all CPUs from the Average block."""
    if sys.platform != "linux" or not _have("mpstat"):
        return None
    try:
        result = subprocess.run(
            ["mpstat", "-P", "ALL", "1", "1"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout
    except Exception:
        return None

    iowait_col: int | None = None
    iowait_values: list[float] = []
    in_average = False
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if not parts:
            continue
        if "Average:" in parts[0] and "%iowait" in stripped:
            in_average = True
            try:
                iowait_col = parts.index("%iowait")
            except ValueError:
                return None
            continue
        if in_average and iowait_col is not None and parts[0] == "Average:" and len(parts) > iowait_col:
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
            local, remote, numa_scope = _read_numastat(self._pid)
            llc_misses, cache_refs = _read_perf_stat(self._pid)
            iowait = _read_mpstat()

            llc_miss_rate: float | None = None
            if llc_misses is not None and cache_refs is not None and cache_refs > 0:
                llc_miss_rate = llc_misses / cache_refs

            self.samples.append(
                TelemetrySample(
                    t=time.perf_counter(),
                    pid=self._pid,
                    cpu_percent=psutil.cpu_percent(interval=None, percpu=True),
                    numa_local_mb=local,
                    numa_remote_mb=remote,
                    numa_scope=numa_scope,
                    perf_cache_misses=llc_misses,
                    perf_cache_refs=cache_refs,
                    perf_llc_miss_rate=llc_miss_rate,
                    mpstat_iowait_pct=iowait,
                )
            )
            self._stop.wait(self.interval_s)

    def start(self, pid: int | None = None) -> None:
        self.samples.clear()
        self._stop.clear()
        self._pid = pid if pid is not None else self._pid
        self._t_start = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> TelemetrySummary:
        if self._thread is None:
            return TelemetrySummary(
                duration_s=0.0,
                samples=0,
                mean_cpu_percent=0.0,
                remote_numa_fraction=None,
                numa_available=False,
                numa_scope="unavailable",
                perf_available=False,
                mpstat_available=False,
            )

        self._stop.set()
        self._thread.join(timeout=2.0)
        duration = time.perf_counter() - (self._t_start or time.perf_counter())
        self._thread = None

        cpu_avg = 0.0
        if self.samples:
            flat = [value for sample in self.samples for value in sample.cpu_percent]
            cpu_avg = sum(flat) / len(flat) if flat else 0.0

        numa_available = any(sample.numa_scope != "unavailable" for sample in self.samples)
        numa_scope = "unavailable"
        for sample in self.samples:
            if sample.numa_scope != "unavailable":
                numa_scope = sample.numa_scope
                break

        remote_frac: float | None = None
        if len(self.samples) >= 2 and numa_available:
            first, last = self.samples[0], self.samples[-1]
            d_local = sum(last.numa_local_mb.get(key, 0) - first.numa_local_mb.get(key, 0) for key in last.numa_local_mb)
            d_remote = sum(last.numa_remote_mb.get(key, 0) - first.numa_remote_mb.get(key, 0) for key in last.numa_remote_mb)
            total = d_local + d_remote
            if total > 0:
                remote_frac = d_remote / total

        miss_rates = [sample.perf_llc_miss_rate for sample in self.samples if sample.perf_llc_miss_rate is not None]
        mean_llc_miss_rate = sum(miss_rates) / len(miss_rates) if miss_rates else None

        iowait_vals = [sample.mpstat_iowait_pct for sample in self.samples if sample.mpstat_iowait_pct is not None]
        peak_iowait_pct = max(iowait_vals) if iowait_vals else None

        perf_available = any(sample.perf_cache_misses is not None for sample in self.samples)
        mpstat_available = any(sample.mpstat_iowait_pct is not None for sample in self.samples)

        return TelemetrySummary(
            duration_s=duration,
            samples=len(self.samples),
            mean_cpu_percent=cpu_avg,
            remote_numa_fraction=remote_frac,
            numa_available=numa_available,
            numa_scope=numa_scope,
            mean_llc_miss_rate=mean_llc_miss_rate,
            peak_iowait_pct=peak_iowait_pct,
            perf_available=perf_available,
            mpstat_available=mpstat_available,
        )
