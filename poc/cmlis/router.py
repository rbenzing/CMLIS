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


# Quantization types that carry high memory pressure per token.
# For these, reduce threads by 1 extra to leave headroom for OS I/O.
_HIGH_MEM_QUANTS = frozenset({"F32", "F16", "BF16", "Q8_0", "Q8_1", "Q8_K", "Q6_K"})

# Low-precision quantizations where we can safely allow more active experts
# on SHORT workloads (smaller working set fits more easily in L3).
_LOW_MEM_QUANTS = frozenset({"Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ1_S", "IQ1_M"})


def decide(
    input_tokens: int,
    cores_per_node: int,
    numa_nodes: int = 1,
    route_index: int = 0,
    mixture_of_experts: bool = True,
    kv_cache_runtime_supported: bool = False,
    quant_type: str | None = None,
) -> RoutingDecision:
    """Produce a routing decision for a single inference job.

    *quant_type* (e.g. "Q5_K_M") adjusts thread count and expert limits when
    the quantization precision is known from the GGUF metadata.
    """
    cls = classify(input_tokens)
    threads = max(1, cores_per_node - 1)
    prefer_numa_node = route_index % max(1, numa_nodes)
    notes: list[str] = []

    # Quant-aware thread adjustment
    if quant_type and quant_type in _HIGH_MEM_QUANTS:
        threads = max(1, threads - 1)
        notes.append(f"high-precision quant ({quant_type}): reduced threads by 1 to leave OS memory headroom.")

    if cls is WorkloadClass.SHORT:
        chunks = 1
        # Low-precision quants have smaller per-expert footprint; allow 4 experts
        if mixture_of_experts and quant_type and quant_type in _LOW_MEM_QUANTS:
            active = 4
            rationale = f"short prompt + low-precision quant ({quant_type}): allow 4 experts (smaller footprint)"
        else:
            active = 2 if mixture_of_experts else 0
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

    if quant_type:
        notes.append(f"quant: {quant_type}")

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
