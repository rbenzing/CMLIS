"""Heuristic routing layer.

Rules-based classifier that inspects prompt length and context state to
pick an execution strategy. Matches SPEC.md §1.3 and MODEL.md §2.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class WorkloadClass(str, Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"
    MIXED = "mixed"


@dataclass
class RoutingDecision:
    workload: WorkloadClass
    active_experts: int  # for MoE models; 0 = "all"
    kv_cache_chunks: int  # 1 = monolithic, >1 = chunked placement
    prefer_numa_node: int  # which node this job binds to
    threads: int
    rationale: str

    def as_flags(self) -> list[str]:
        """llama.cpp flags reflecting the routing decision."""
        flags = ["-t", str(self.threads)]
        if self.active_experts > 0:
            flags.extend(["--override-kv", f"llama.expert_used_count=int:{self.active_experts}"])
        return flags


# Boundaries from METHODOLOGY.md §4 (512 / 2048 / 8192 tokens)
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
    numa_node: int = 0,
    mixture_of_experts: bool = True,
) -> RoutingDecision:
    """Produce a routing decision for a single inference job.

    Heuristic policy:
      - Short prompts: aggressive expert limiting for cache residency.
      - Medium: balanced — all experts, chunked KV.
      - Long: full experts, heavier KV chunking for NUMA-local placement.
    """
    cls = classify(input_tokens)
    threads = max(1, cores_per_node - 1)  # leave one core for OS/telemetry

    if cls is WorkloadClass.SHORT:
        active = 2 if mixture_of_experts else 0
        chunks = 1
        rationale = "short prompt: cap active experts to stay in L3"
    elif cls is WorkloadClass.MEDIUM:
        active = 0
        chunks = 2
        rationale = "medium prompt: all experts, chunk KV across node"
    else:
        active = 0
        chunks = 4
        rationale = "long prompt: all experts, aggressive KV chunking"

    return RoutingDecision(
        workload=cls,
        active_experts=active,
        kv_cache_chunks=chunks,
        prefer_numa_node=numa_node,
        threads=threads,
        rationale=rationale,
    )
