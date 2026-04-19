"""Heuristic routing layer.

Rules-based classifier that inspects prompt length and context state to
pick an execution strategy. The current PoC supports expert limiting and
NUMA-node selection, but long-context KV placement is still planning
metadata rather than a wired runtime feature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .memctl import BindingPlan
    from .topology import Topology


class WorkloadClass(str, Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"
    MIXED = "mixed"


@dataclass
class RoutingDecision:
    workload: WorkloadClass
    active_experts: int
    kv_cache_chunks: int
    prefer_numa_node: int
    threads: int
    rationale: str
    notes: list[str] = field(default_factory=list)

    def as_flags(self) -> list[str]:
        """llama.cpp flags reflecting the routing decision."""
        flags = ["-t", str(self.threads)]
        if self.active_experts > 0:
            flags.extend(["--override-kv", f"llama.expert_used_count=int:{self.active_experts}"])
        return flags


SHORT_MAX = 1024
MEDIUM_MAX = 4096


def classify(input_tokens: int) -> WorkloadClass:
    if input_tokens <= SHORT_MAX:
        return WorkloadClass.SHORT
    if input_tokens <= MEDIUM_MAX:
        return WorkloadClass.MEDIUM
    return WorkloadClass.LONG


def decide(
    input_tokens: int,
    cores_per_node: int,
    numa_nodes: int = 1,
    route_index: int = 0,
    mixture_of_experts: bool = True,
    kv_cache_runtime_supported: bool = False,
) -> RoutingDecision:
    """Produce a routing decision for a single inference job."""
    cls = classify(input_tokens)
    threads = max(1, cores_per_node - 1)
    prefer_numa_node = route_index % max(1, numa_nodes)
    notes: list[str] = []

    if cls is WorkloadClass.SHORT:
        active = 2 if mixture_of_experts else 0
        chunks = 1
        rationale = "short prompt: cap active experts to stay in L3"
    elif cls is WorkloadClass.MEDIUM:
        active = 0
        chunks = 2
        rationale = "medium prompt: distribute runs across NUMA nodes; KV chunking is planning metadata"
    else:
        active = 0
        chunks = 4
        rationale = "long prompt: distribute runs across NUMA nodes; KV placement is not yet implemented"

    if chunks > 1 and not kv_cache_runtime_supported:
        notes.append("kv_cache_chunks is planning metadata only; runtime KV placement is not yet supported.")

    return RoutingDecision(
        workload=cls,
        active_experts=active,
        kv_cache_chunks=chunks,
        prefer_numa_node=prefer_numa_node,
        threads=threads,
        rationale=rationale,
        notes=notes,
    )


def validate_binding(decision: RoutingDecision, binding: BindingPlan, topo: Topology) -> list[str]:
    """Validate that a binding plan matches the intended topology."""
    issues: list[str] = []
    if not topo.numa_nodes:
        return issues

    expected = topo.numa_nodes[decision.prefer_numa_node % len(topo.numa_nodes)]
    if binding.numa_node != expected.node_id:
        issues.append(f"binding targets NUMA node {binding.numa_node}, expected {expected.node_id}")

    if binding.cpus:
        expected_cpus = set(expected.cpus)
        actual_cpus = set(binding.cpus)
        if not actual_cpus.issubset(expected_cpus):
            issues.append(
                f"binding CPUs {sorted(actual_cpus)} do not fit inside NUMA node {expected.node_id} CPUs {sorted(expected_cpus)}"
            )

    if topo.system == "Linux" and topo.numa_available and not binding.enforced:
        issues.append("NUMA topology is available but binding enforcement is disabled.")

    return issues
