"""Benchmark harness.

Runs the three configurations from METHODOLOGY.md section 5 across one or
more workloads, collects tokens/sec, and computes a Welch two-sample t-test
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


@dataclass
class ControlResult:
    ok: bool
    label: str
    detail: str = ""


@dataclass
class RunResult:
    config: str
    workload: str
    rep: int
    tokens_per_second: float
    wall_seconds: float
    simulated: bool
    exit_code: int
    measurement_valid: bool
    status: str
    message: str
    stderr_summary: str
    binding_validation: list[str]
    remote_numa_fraction: float | None
    telemetry_summary: dict | None
    telemetry_samples: list[dict]
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
    cv_pct: float
    variance_ok: bool


@dataclass
class BenchReport:
    topology_summary: str
    swap_mb_at_start: float = 0.0
    results: list[RunResult] = field(default_factory=list)
    stats: list[ConfigStats] = field(default_factory=list)
    significance: dict = field(default_factory=dict)
    validity: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "topology_summary": self.topology_summary,
            "swap_mb_at_start": self.swap_mb_at_start,
            "results": [asdict(result) for result in self.results],
            "stats": [asdict(stat) for stat in self.stats],
            "significance": self.significance,
            "validity": self.validity,
        }


class BenchValidationError(RuntimeError):
    def __init__(self, report: BenchReport, failures: list[str]):
        self.report = report
        self.failures = failures
        super().__init__("benchmark validity gates failed")


def _try_privileged(cmd: list[str], label: str) -> ControlResult:
    """Run a privileged system command; warn to stderr if it fails."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            detail = result.stderr.strip()
            print(f"WARNING: {label} failed (rc={result.returncode}): {detail}", file=sys.stderr)
            return ControlResult(ok=False, label=label, detail=detail)
        return ControlResult(ok=True, label=label)
    except (subprocess.SubprocessError, OSError) as exc:
        detail = str(exc)
        print(f"WARNING: {label} unavailable: {detail}", file=sys.stderr)
        return ControlResult(ok=False, label=label, detail=detail)


WORKLOADS = {
    "short": {"input": 512, "output": 256, "default_n": 50, "min_n": 30},
    "medium": {"input": 2048, "output": 512, "default_n": 50, "min_n": 30},
    "long": {"input": 8192, "output": 256, "default_n": 30, "min_n": 30},
    "mixed": {"input": 1024, "output": 256, "default_n": 50, "min_n": 30},
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
    n = max(10, int(input_tokens * 0.75))
    return " ".join(rng.choice(words) for _ in range(n))


def _plan(config: str, decision: router.RoutingDecision, topo: topology.Topology) -> memctl.BindingPlan:
    """Materialize the binding plan for a given config."""
    if config == "naive":
        return memctl.BindingPlan(
            numa_node=0, cpus=[], enforced=False, prefix=[], notes=["naive: no binding"]
        )
    node = topo.numa_nodes[decision.prefer_numa_node % max(1, len(topo.numa_nodes))]
    return memctl.build_binding(node.node_id, node.cpus)


def _route(config: str, input_tokens: int, topo: topology.Topology, route_index: int = 0) -> router.RoutingDecision:
    node = topo.numa_nodes[0] if topo.numa_nodes else None
    cores_per_node = len(node.cpus) if node else (topo.physical_cores or 4)
    numa_nodes = max(1, len(topo.numa_nodes))
    prefer_numa_node = route_index % numa_nodes

    if config == "naive":
        return router.RoutingDecision(
            workload=router.classify(input_tokens),
            active_experts=0,
            kv_cache_chunks=1,
            prefer_numa_node=0,
            threads=topo.physical_cores or cores_per_node,
            rationale="naive: default llama.cpp",
        )
    if config == "numa":
        return router.RoutingDecision(
            workload=router.classify(input_tokens),
            active_experts=0,
            kv_cache_chunks=1,
            prefer_numa_node=prefer_numa_node,
            threads=cores_per_node,
            rationale="numa-only: rotate placements across discovered NUMA nodes",
        )
    return router.decide(input_tokens, cores_per_node=cores_per_node, numa_nodes=numa_nodes, route_index=route_index)


def _welch_t(a: list[float], b: list[float]) -> tuple[float, float, int]:
    """Welch two-sample t-statistic and two-sided p-value."""
    if len(a) < 2 or len(b) < 2:
        return 0.0, 1.0, 0
    mean_a, mean_b = statistics.mean(a), statistics.mean(b)
    var_a, var_b = statistics.variance(a), statistics.variance(b)
    n_a, n_b = len(a), len(b)
    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se == 0:
        return 0.0, 1.0, n_a + n_b - 2
    t_stat = (mean_a - mean_b) / se
    numerator = (var_a / n_a + var_b / n_b) ** 2
    denominator = (var_a * var_a) / (n_a * n_a * (n_a - 1)) + (var_b * var_b) / (
        n_b * n_b * (n_b - 1)
    )
    df = int(numerator / denominator) if denominator > 0 else n_a + n_b - 2
    from scipy.stats import t as t_dist

    p_value = float(2 * t_dist.sf(abs(t_stat), df))
    return t_stat, p_value, df


def _run_status(run: engine.EngineRun) -> str:
    if run.simulated:
        return "simulated"
    if run.exit_code != 0:
        return "failed"
    if not run.measurement_valid:
        return "invalid_measurement"
    return "ok"


def _stderr_summary(stderr: str) -> str:
    return stderr.strip().replace("\n", " ")[:200]


def _build_validity(
    report: BenchReport,
    gate_failures: list[str],
    strict: bool,
    allow_invalid: bool,
) -> None:
    invalid_runs = [result for result in report.results if not result.measurement_valid]
    failed_runs = [result for result in report.results if result.exit_code != 0]
    binding_issue_runs = [result for result in report.results if result.binding_validation]
    report.validity = {
        "total_runs": len(report.results),
        "valid_runs": len(report.results) - len(invalid_runs),
        "invalid_runs": len(invalid_runs),
        "failed_runs": len(failed_runs),
        "binding_issue_runs": len(binding_issue_runs),
        "all_measurements_valid": not invalid_runs,
        "strict_mode": strict,
        "allow_invalid": allow_invalid,
        "gate_failures": gate_failures,
        "passed_gates": not gate_failures,
    }


def run_bench(
    workloads: list[str],
    configs: list[str],
    reps: int | None = None,
    model_path: str | None = None,
    binary: str | None = None,
    simulate: bool = False,
    collect_telemetry: bool = True,
    seed: int = 42,
    strict: bool = False,
    allow_invalid: bool = False,
) -> BenchReport:
    import psutil

    topo = topology.discover()
    swap_mb = psutil.swap_memory().used / (1024 * 1024)
    report = BenchReport(topology_summary=topo.summary(), swap_mb_at_start=round(swap_mb, 1))
    gate_failures: list[str] = []
    seen_failures: set[str] = set()

    if swap_mb > 0:
        print(
            f"WARNING: OS swap in use ({swap_mb:.0f} MB). Results may be unreliable (SPEC section 5).",
            file=sys.stderr,
        )
        if not simulate:
            gate_failures.append(f"swap in use at bench start: {round(swap_mb, 1)} MB")

    if topo.system == "Linux" and not simulate:
        numa_control = _try_privileged(memctl.disable_numa_balancing_cmd(), "disable NUMA balancing")
        if not numa_control.ok:
            gate_failures.append(f"{numa_control.label} failed: {numa_control.detail or 'unknown error'}")
    else:
        _try_privileged(memctl.disable_numa_balancing_cmd(), "disable NUMA balancing")

    for workload in workloads:
        if workload not in WORKLOADS:
            raise ValueError(f"unknown workload: {workload}")
        spec = WORKLOADS[workload]
        n = reps if reps is not None else spec["default_n"]
        if n < spec["min_n"]:
            message = (
                f"{workload} workload has n={n} reps, below minimum {spec['min_n']} for statistical validity."
            )
            print(f"WARNING: {message}", file=sys.stderr)
            if not simulate:
                gate_failures.append(message)

        for config in configs:
            if config not in CONFIGS:
                raise ValueError(f"unknown config: {config}")
            cache_control = _try_privileged(memctl.drop_caches_cmd(), "drop OS caches")
            if topo.system == "Linux" and not simulate and not cache_control.ok:
                failure = f"{cache_control.label} failed: {cache_control.detail or 'unknown error'}"
                if failure not in seen_failures:
                    seen_failures.add(failure)
                    gate_failures.append(failure)

            for rep in range(n):
                decision = _route(config, spec["input"], topo, route_index=rep)
                binding = _plan(config, decision, topo)
                binding_validation = router.validate_binding(decision, binding, topo)
                prompt = _make_prompt(spec["input"], seed + rep)
                collector: telemetry.TelemetryCollector | None = None
                summary: telemetry.TelemetrySummary | None = None
                if collect_telemetry:
                    collector = telemetry.TelemetryCollector(interval_s=0.5)
                run = engine.run(
                    decision,
                    binding,
                    prompt,
                    output_tokens=spec["output"],
                    model_path=model_path,
                    binary=binary,
                    simulate=simulate,
                    seed=seed + rep,
                    config=config,
                    on_start=collector.start if collector is not None else None,
                )
                summary = collector.stop() if collector else None
                report.results.append(
                    RunResult(
                        config=config,
                        workload=workload,
                        rep=rep,
                        tokens_per_second=run.tokens_per_second,
                        wall_seconds=run.wall_seconds,
                        simulated=run.simulated,
                        exit_code=run.exit_code,
                        measurement_valid=run.measurement_valid,
                        status=_run_status(run),
                        message=run.message,
                        stderr_summary=_stderr_summary(run.stderr),
                        binding_validation=binding_validation,
                        remote_numa_fraction=summary.remote_numa_fraction if summary else None,
                        telemetry_summary=summary.as_dict() if summary else None,
                        telemetry_samples=[sample.as_dict() for sample in collector.samples] if collector else [],
                        command=run.command,
                    )
                )

    grouped: dict[tuple[str, str], list[float]] = {}
    for result in report.results:
        if result.measurement_valid:
            grouped.setdefault((result.config, result.workload), []).append(result.tokens_per_second)
    for (config, workload), values in grouped.items():
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        cv_pct = (stdev / mean * 100) if mean > 0 else 0.0
        report.stats.append(
            ConfigStats(
                config=config,
                workload=workload,
                n=len(values),
                mean_tps=mean,
                stdev_tps=stdev,
                min_tps=min(values),
                max_tps=max(values),
                cv_pct=round(cv_pct, 2),
                variance_ok=cv_pct < 5.0,
            )
        )

    for workload in workloads:
        naive = [
            result.tokens_per_second
            for result in report.results
            if result.config == "naive" and result.workload == workload and result.measurement_valid
        ]
        full = [
            result.tokens_per_second
            for result in report.results
            if result.config == "full" and result.workload == workload and result.measurement_valid
        ]
        if naive and full:
            t_stat, p_value, df = _welch_t(full, naive)
            uplift = (
                (statistics.mean(full) / statistics.mean(naive) - 1.0) * 100
                if statistics.mean(naive) > 0
                else 0.0
            )
            report.significance[workload] = {
                "uplift_pct": round(uplift, 2),
                "t": round(t_stat, 3),
                "p": round(p_value, 4),
                "df": df,
                "meets_25pct": uplift >= 25.0,
                "significant_p01": p_value < 0.01,
            }

    _build_validity(report, gate_failures, strict=strict, allow_invalid=allow_invalid)
    if strict and gate_failures and not allow_invalid:
        raise BenchValidationError(report=report, failures=gate_failures)
    return report


def save_report(report: BenchReport, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"cmlis-bench-{stamp}.json"
    path.write_text(json.dumps(report.to_dict(), indent=2))
    return path
