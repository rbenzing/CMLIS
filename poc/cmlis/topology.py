"""Hardware topology discovery.

Detects NUMA nodes, physical cores, and L3 cache layout. On Linux, parses
`numactl --hardware` and `lscpu`. On Windows/macOS, falls back to `psutil`
with best-effort inference and emits a warning that NUMA features are
unavailable for live enforcement.
"""

from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field


@dataclass
class NumaNode:
    node_id: int
    cpus: list[int]
    memory_mb: int


@dataclass
class Topology:
    system: str
    sockets: int
    physical_cores: int
    logical_cores: int
    total_memory_mb: int
    numa_nodes: list[NumaNode] = field(default_factory=list)
    l3_cache_kb: int | None = None
    numa_available: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def summary(self) -> str:
        lines = [
            f"System: {self.system}",
            f"Sockets: {self.sockets}  Physical cores: {self.physical_cores}  Logical: {self.logical_cores}",
            f"Total memory: {self.total_memory_mb} MB",
            f"L3 cache: {self.l3_cache_kb} KB" if self.l3_cache_kb else "L3 cache: unknown",
            f"NUMA nodes: {len(self.numa_nodes)} ({'available' if self.numa_available else 'simulated'})",
        ]
        for n in self.numa_nodes:
            lines.append(f"  node {n.node_id}: {len(n.cpus)} cpus, {n.memory_mb} MB")
        for note in self.notes:
            lines.append(f"note: {note}")
        return "\n".join(lines)


def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _run(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return r.stdout
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return ""


def _parse_numactl_hardware(text: str) -> list[NumaNode]:
    nodes: dict[int, NumaNode] = {}
    for m in re.finditer(r"node (\d+) cpus:\s*([\d ]+)", text):
        nid = int(m.group(1))
        cpus = [int(x) for x in m.group(2).split()]
        nodes[nid] = NumaNode(node_id=nid, cpus=cpus, memory_mb=0)
    for m in re.finditer(r"node (\d+) size:\s*(\d+)\s*MB", text):
        nid = int(m.group(1))
        if nid in nodes:
            nodes[nid].memory_mb = int(m.group(2))
    return [nodes[k] for k in sorted(nodes)]


def _parse_lscpu_l3(text: str) -> int | None:
    m = re.search(r"L3 cache:\s*([\d.]+)\s*([KMG])iB", text)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2)
    mult = {"K": 1, "M": 1024, "G": 1024 * 1024}[unit]
    return int(val * mult)


def _parse_lscpu_sockets(text: str) -> int | None:
    m = re.search(r"^Socket\(s\):\s*(\d+)", text, re.MULTILINE)
    return int(m.group(1)) if m else None


def discover() -> Topology:
    """Discover topology on the current machine."""
    import psutil

    system = platform.system()
    logical = psutil.cpu_count(logical=True) or 0
    physical = psutil.cpu_count(logical=False) or logical
    total_mem_mb = int(psutil.virtual_memory().total / (1024 * 1024))

    topo = Topology(
        system=system,
        sockets=1,
        physical_cores=physical,
        logical_cores=logical,
        total_memory_mb=total_mem_mb,
    )

    if system == "Linux" and _have("numactl"):
        nh = _run(["numactl", "--hardware"])
        nodes = _parse_numactl_hardware(nh)
        if nodes:
            topo.numa_nodes = nodes
            topo.numa_available = True
        if _have("lscpu"):
            lscpu_out = _run(["lscpu"])
            topo.l3_cache_kb = _parse_lscpu_l3(_run(["lscpu", "-C=L3"]) or lscpu_out)
            sockets = _parse_lscpu_sockets(lscpu_out)
            topo.sockets = sockets if sockets is not None else len(nodes) if nodes else 1
    else:
        topo.notes.append(f"NUMA tools unavailable on {system}; treating machine as single simulated node.")
        topo.numa_nodes = [NumaNode(node_id=0, cpus=list(range(logical)), memory_mb=total_mem_mb)]

    return topo


def main() -> None:
    t = discover()
    print(t.summary())
    print()
    print(json.dumps(t.to_dict(), indent=2))


if __name__ == "__main__":
    main()
