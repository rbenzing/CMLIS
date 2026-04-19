"""Benchmark harness.

Runs the three configurations from METHODOLOGY.md §5 across one or more
workloads, collects tokens/sec, and computes a Welch two-sample t-test
between naive and full-CMLIS.
"""

from __future__ import annotations

import json
import math
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import engine, memctl, router, telemetry, topology

CONFIGS = ("naive", "numa", "full")


def _try_privileged(cmd: list[str], label: str) -> None:
    """Run a privileged system command; warn to stderr if it fails (e.g. no root)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            print(f"WARNING: {label} failed (rc={r.returncode}): {r.stderr.strip()}", file=sys.stderr)
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"WARNING: {label} unavailable: {exc}", file=sys.stderr)


# Workload table from METHODOLOGY.md §4.
# N must be ≥ 30 for CLT validity; METHODOLOGY requires 50/50/30.
WORKLOADS = {
    "short": {"input": 512, "output": 256, "default_n": 50, "min_n": 30},
    "medium": {"input": 2048, "output": 512, "default_n": 50, "min_n": 30},
    "long": {"input": 8192, "output": 256, "default_n": 30, "min_n": 30},
    "mixed": {"input": 1024, "output": 256, "default_n": 50, "min_n": 30},
}


@dataclass
class RunResult:
    config: str
    workload: str
    rep: int
    tokens_per_second: float
    wall_seconds: float
    simulated: bool
    remote_numa_fraction: float | None
    command: list[str]


@dataclass
class ConfigStats:
    config: str
    workload: str
    n: int
    mean_tps: float
    stdev_tps: float
    min_tps: float
    max_tps: float
    cv_pct: float  # coefficient of variation (stdev/mean * 100)
    variance_ok: bool  # True if cv_pct < 5 (SPEC §5)


@dataclass
class BenchReport:
    topology_summary: str
    swap_mb_at_start: float = 0.0  # SPEC §5: must be 0 for clean run
    results: list[RunResult] = field(default_factory=list)
    stats: list[ConfigStats] = field(default_factory=list)
    significance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "topology_summary": self.topology_summary,
            "swap_mb_at_start": self.swap_mb_at_start,
            "results": [asdict(r) for r in self.results],
            "stats": [asdict(s) for s in self.stats],
            "significance": self.significance,
        }


def _make_prompt(input_tokens: int, seed: int) -> str:
    """Synthetic deterministic prompt of approximately `input_tokens` tokens."""
    import random

    rng = random.Random(seed)
    words = [
        "memory",
        "cache",
        "numa",
        "cpu",
        "token",
        "latency",
        "bandwidth",
        "node",
        "core",
        "socket",
        "inference",
        "kernel",
        "throughput",
        "cold",
        "warm",
        "prefetch",
        "vector",
        "matrix",
        "weight",
        "layer",
    ]
    n = max(10, int(input_tokens * 0.75))  # rough token-to-word ratio
    return " ".join(rng.choice(words) for _ in range(n))


def _plan(config: str, decision: router.RoutingDecision, topo: topology.Topology) -> memctl.BindingPlan:
    """Materialise the binding plan for a given config."""
    if config == "naive":
        return memctl.BindingPlan(
            numa_node=0, cpus=[], enforced=False, prefix=[], notes=["naive: no binding"]
        )
    node = topo.numa_nodes[decision.prefer_numa_node % max(1, len(topo.numa_nodes))]
    return memctl.build_binding(node.node_id, node.cpus)


def _route(config: str, input_tokens: int, topo: topology.Topology) -> router.RoutingDecision:
    node = topo.numa_nodes[0] if topo.numa_nodes else None
    cores_per_node = len(node.cpus) if node else (topo.physical_cores or 4)

    if config == "naive":
        # All cores, all experts, no routing logic.
        return router.RoutingDecision(
            workload=router.classify(input_tokens),
            active_experts=0,
            kv_cache_chunks=1,
            prefer_numa_node=0,
            threads=topo.physical_cores or cores_per_node,
            rationale="naive: default llama.cpp",
        )
    if config == "numa":
        # NUMA binding only — router is bypassed.
        return router.RoutingDecision(
            workload=router.classify(input_tokens),
            active_experts=0,
            kv_cache_chunks=1,
            prefer_numa_node=0,
            threads=cores_per_node,
            rationale="numa-only: binding without routing",
        )
    # full CMLIS
    return router.decide(input_tokens, cores_per_node=cores_per_node, numa_node=0)


def _welch_t(a: list[float], b: list[float]) -> tuple[float, float, int]:
    """Welch two-sample t-statistic and two-sided p-value (scipy t-distribution)."""
    if len(a) < 2 or len(b) < 2:
        return 0.0, 1.0, 0
    m1, m2 = statistics.mean(a), statistics.mean(b)
    v1, v2 = statistics.variance(a), statistics.variance(b)
    n1, n2 = len(a), len(b)
    se = math.sqrt(v1 / n1 + v2 / n2)
    if se == 0:
        return 0.0, 1.0, n1 + n2 - 2
    t = (m1 - m2) / se
    # Welch-Satterthwaite degrees of freedom
    num = (v1 / n1 + v2 / n2) ** 2
    den = (v1 * v1) / (n1 * n1 * (n1 - 1)) + (v2 * v2) / (n2 * n2 * (n2 - 1))
    df = int(num / den) if den > 0 else n1 + n2 - 2
    # Two-sided p-value from t-distribution (correct for small N).
    from scipy.stats import t as t_dist

    p = float(2 * t_dist.sf(abs(t), df))
    return t, p, df


def run_bench(
    workloads: list[str],
    configs: list[str],
    reps: int | None = None,
    model_path: str | None = None,
    binary: str | None = None,
    simulate: bool = False,
    collect_telemetry: bool = True,
    seed: int = 42,
) -> BenchReport:
    import psutil

    topo = topology.discover()
    swap_mb = psutil.swap_memory().used / (1024 * 1024)
    report = BenchReport(topology_summary=topo.summary(), swap_mb_at_start=round(swap_mb, 1))
    if swap_mb > 0:
        print(
            f"WARNING: OS swap in use ({swap_mb:.0f} MB). Results may be unreliable (SPEC §5).",
            file=sys.stderr,
        )

    # Disable kernel NUMA page migration so binding stays effective.
    _try_privileged(memctl.disable_numa_balancing_cmd(), "disable NUMA balancing")

    for wl in workloads:
        if wl not in WORKLOADS:
            raise ValueError(f"unknown workload: {wl}")
        spec = WORKLOADS[wl]
        n = reps if reps is not None else spec["default_n"]
        if n < spec["min_n"]:
            print(
                f"WARNING: {wl} workload has n={n} reps, below minimum {spec['min_n']} for statistical validity.",
                file=sys.stderr,
            )

        for cfg in configs:
            if cfg not in CONFIGS:
                raise ValueError(f"unknown config: {cfg}")
            decision = _route(cfg, spec["input"], topo)
            binding = _plan(cfg, decision, topo)

            # Drop OS page cache between config batches for clean DRAM fetch metrics.
            _try_privileged(memctl.drop_caches_cmd(), "drop OS caches")

            for i in range(n):
                prompt = _make_prompt(spec["input"], seed + i)
                tel: telemetry.TelemetryCollector | None = None
                if collect_telemetry:
                    tel = telemetry.TelemetryCollector(interval_s=0.5)
                    tel.start()
                run = engine.run(
                    decision,
                    binding,
                    prompt,
                    output_tokens=spec["output"],
                    model_path=model_path,
                    binary=binary,
                    simulate=simulate,
                    seed=seed + i,
                    config=cfg,
                )
                summary = tel.stop() if tel else None
                report.results.append(
                    RunResult(
                        config=cfg,
                        workload=wl,
                        rep=i,
                        tokens_per_second=run.tokens_per_second,
                        wall_seconds=run.wall_seconds,
                        simulated=run.simulated,
                        remote_numa_fraction=summary.remote_numa_fraction if summary else None,
                        command=run.command,
                    )
                )

    # Per-config stats
    grouped: dict[tuple[str, str], list[float]] = {}
    for r in report.results:
        grouped.setdefault((r.config, r.workload), []).append(r.tokens_per_second)
    for (cfg, wl), vals in grouped.items():
        mean = statistics.mean(vals)
        stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
        cv_pct = (stdev / mean * 100) if mean > 0 else 0.0
        report.stats.append(
            ConfigStats(
                config=cfg,
                workload=wl,
                n=len(vals),
                mean_tps=mean,
                stdev_tps=stdev,
                min_tps=min(vals),
                max_tps=max(vals),
                cv_pct=round(cv_pct, 2),
                variance_ok=cv_pct < 5.0,
            )
        )

    # Significance: full vs naive per workload
    for wl in workloads:
        naive = [r.tokens_per_second for r in report.results if r.config == "naive" and r.workload == wl]
        full = [r.tokens_per_second for r in report.results if r.config == "full" and r.workload == wl]
        if naive and full:
            t, p, df = _welch_t(full, naive)
            uplift = (
                (statistics.mean(full) / statistics.mean(naive) - 1.0) * 100
                if statistics.mean(naive) > 0
                else 0.0
            )
            report.significance[wl] = {
                "uplift_pct": round(uplift, 2),
                "t": round(t, 3),
                "p": round(p, 4),
                "df": df,
                "meets_25pct": uplift >= 25.0,
                "significant_p01": p < 0.01,
            }
    return report


def save_report(report: BenchReport, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"cmlis-bench-{stamp}.json"
    path.write_text(json.dumps(report.to_dict(), indent=2))
    return path
