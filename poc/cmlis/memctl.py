"""Memory Control Layer.

Builds the command prefix that binds a process to a specific NUMA node and
physical core set. On Linux, uses `numactl --cpunodebind --membind` plus
`taskset -c`. On other platforms, returns an empty prefix and records that
enforcement was skipped.
"""

from __future__ import annotations

import platform
import shutil
from dataclasses import dataclass


@dataclass
class BindingPlan:
    numa_node: int
    cpus: list[int]
    enforced: bool
    prefix: list[str]
    notes: list[str]


def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def build_binding(numa_node: int, cpus: list[int]) -> BindingPlan:
    """Build a platform-appropriate command prefix to enforce NUMA binding."""
    system = platform.system()
    notes: list[str] = []

    if system != "Linux":
        notes.append(f"{system}: NUMA enforcement unavailable; returning no-op prefix.")
        return BindingPlan(numa_node, cpus, enforced=False, prefix=[], notes=notes)

    prefix: list[str] = []
    if _have("numactl"):
        prefix += ["numactl", f"--cpunodebind={numa_node}", f"--membind={numa_node}"]
    else:
        notes.append("numactl not installed; skipping NUMA memory bind.")

    if _have("taskset") and cpus:
        cpu_list = ",".join(str(c) for c in cpus)
        prefix += ["taskset", "-c", cpu_list]
    elif not cpus:
        notes.append("no CPU list provided; skipping taskset.")
    else:
        notes.append("taskset not installed; skipping affinity.")

    return BindingPlan(
        numa_node=numa_node,
        cpus=cpus,
        enforced=bool(prefix),
        prefix=prefix,
        notes=notes,
    )


def drop_caches_cmd() -> list[str]:
    """Return the command to drop OS caches between runs (requires root)."""
    return ["sh", "-c", "echo 3 > /proc/sys/vm/drop_caches && echo 1 > /proc/sys/vm/compact_memory"]


def disable_numa_balancing_cmd() -> list[str]:
    return ["sh", "-c", "echo 0 > /proc/sys/kernel/numa_balancing"]
