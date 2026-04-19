"""Perplexity measurement module.

Downloads WikiText-2 test set and runs llama.cpp --perplexity to measure
PPL degradation relative to naive baseline. Success criterion (SPEC.md):
degradation <= 0.1. Simulation is only used when explicitly requested.
"""

from __future__ import annotations

import random
import re
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import engine, memctl, router, topology

_WIKITEXT2_URL = (
    "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt"
)
_CACHE_PATH = Path.home() / ".cache" / "cmlis" / "wikitext2-test.txt"

PPL_CONFIGS = ("naive", "full")

_PPL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+)")


@dataclass
class PplResult:
    config: str
    ppl: float
    ppl_stderr: float | None
    simulated: bool
    exit_code: int = 0
    message: str = ""


def _download_wikitext2() -> Path:
    """Download WikiText-2 test set to cache if not present; return path."""
    if _CACHE_PATH.exists():
        return _CACHE_PATH
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(_WIKITEXT2_URL, _CACHE_PATH)
    return _CACHE_PATH


def _parse_ppl(output: str) -> tuple[float, float | None]:
    """Parse PPL and stderr from llama.cpp --perplexity output."""
    m = _PPL_RE.search(output)
    if m:
        return float(m.group(1)), float(m.group(2))
    return 0.0, None


def _simulate_ppl(config: str, seed: int, reason: str = "simulation mode requested") -> PplResult:
    """Return synthetic PPL values for simulation mode."""
    rng = random.Random(seed)
    base = 4.23
    offset = 0.02 if config == "full" else 0.0
    noise = rng.uniform(-0.005, 0.005)
    ppl_value = base + offset + noise
    return PplResult(
        config=config,
        ppl=round(ppl_value, 4),
        ppl_stderr=0.01,
        simulated=True,
        exit_code=0,
        message=reason,
    )


def _error_result(config: str, message: str) -> PplResult:
    return PplResult(
        config=config,
        ppl=0.0,
        ppl_stderr=None,
        simulated=False,
        exit_code=2,
        message=message,
    )


def _route_for_ppl(config: str, topo: topology.Topology) -> router.RoutingDecision:
    """Build routing decision for PPL measurement (medium context, 2048 tokens)."""
    node = topo.numa_nodes[0] if topo.numa_nodes else None
    cores_per_node = len(node.cpus) if node else (topo.physical_cores or 4)
    if config == "naive":
        return router.RoutingDecision(
            workload=router.WorkloadClass.MEDIUM,
            active_experts=0,
            kv_cache_chunks=1,
            prefer_numa_node=0,
            threads=topo.physical_cores or cores_per_node,
            rationale="naive: default llama.cpp",
        )
    return router.decide(2048, cores_per_node=cores_per_node, numa_nodes=max(1, len(topo.numa_nodes)))


def _plan_for_ppl(
    config: str, decision: router.RoutingDecision, topo: topology.Topology
) -> memctl.BindingPlan:
    """Build binding plan for PPL measurement."""
    if config == "naive":
        return memctl.BindingPlan(
            numa_node=0, cpus=[], enforced=False, prefix=[], notes=["naive: no binding"]
        )
    node = topo.numa_nodes[decision.prefer_numa_node % max(1, len(topo.numa_nodes))]
    return memctl.build_binding(node.node_id, node.cpus)


def run_ppl(
    configs: list[str] | None = None,
    model_path: str | None = None,
    binary: str | None = None,
    simulate: bool = False,
    seed: int = 42,
    timeout: float = 3600.0,
) -> list[PplResult]:
    """Run perplexity measurement for each config."""
    if configs is None:
        configs = list(PPL_CONFIGS)

    if simulate:
        return [_simulate_ppl(cfg, seed) for cfg in configs]

    resolved_binary = engine.resolve_binary(binary)
    if not resolved_binary:
        message = "llama.cpp binary not found; pass --binary, set LLAMA_CPP_BIN, or use --simulate"
        return [_error_result(cfg, message) for cfg in configs]
    if not model_path:
        message = "model path is required for a real PPL run; pass --model or use --simulate"
        return [_error_result(cfg, message) for cfg in configs]
    if not Path(model_path).exists():
        message = f"model not found at {model_path!r}; pass a valid --model or use --simulate"
        return [_error_result(cfg, message) for cfg in configs]

    try:
        wikitext_path = _download_wikitext2()
    except OSError as exc:
        message = f"failed to prepare WikiText-2 dataset: {exc}"
        return [_error_result(cfg, message) for cfg in configs]

    topo = topology.discover()
    results: list[PplResult] = []
    for cfg in configs:
        decision = _route_for_ppl(cfg, topo)
        binding = _plan_for_ppl(cfg, decision, topo)
        cmd = list(binding.prefix) + [
            resolved_binary,
            "-m",
            model_path,
            "--perplexity",
            "-f",
            str(wikitext_path),
            "-t",
            str(decision.threads),
            "--seed",
            str(seed),
        ]

        try:
            completed = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            combined = completed.stdout + "\n" + completed.stderr
            ppl_value, ppl_err = _parse_ppl(combined)
            results.append(
                PplResult(
                    config=cfg,
                    ppl=ppl_value,
                    ppl_stderr=ppl_err,
                    simulated=False,
                    exit_code=completed.returncode,
                    message="real execution completed"
                    if completed.returncode == 0
                    else "real execution failed",
                )
            )
        except subprocess.TimeoutExpired:
            results.append(
                PplResult(
                    config=cfg,
                    ppl=0.0,
                    ppl_stderr=None,
                    simulated=False,
                    exit_code=124,
                    message=f"timeout after {timeout}s",
                )
            )

    return results


def check_degradation(results: list[PplResult]) -> dict[str, dict]:
    """Compare each config's PPL against the naive baseline.

    Returns a dict keyed by config with keys: ppl, delta, passes.
    delta = config_ppl - naive_ppl; passes = delta <= 0.1.
    """
    naive_ppl: float | None = None
    for result in results:
        if result.config == "naive":
            naive_ppl = result.ppl
            break

    out: dict[str, dict] = {}
    for result in results:
        delta = (result.ppl - naive_ppl) if naive_ppl is not None else 0.0
        out[result.config] = {
            "ppl": result.ppl,
            "delta": round(delta, 6),
            "passes": delta <= 0.1,
        }
    return out
