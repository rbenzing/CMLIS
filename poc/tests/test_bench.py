import argparse
from pathlib import Path

import psutil

from cmlis import bench, cli, engine, telemetry, topology


def _valid_run(simulated: bool = True, tps: float = 5.0) -> engine.EngineRun:
    return engine.EngineRun(
        simulated=simulated,
        command=["<simulated>"] if simulated else ["fake"],
        stdout="ok",
        stderr="",
        exit_code=0,
        wall_seconds=1.0,
        tokens_generated=256,
        tokens_per_second=tps,
        message="simulation mode requested" if simulated else "real execution completed",
        measurement_valid=True,
    )


def test_simulated_bench_produces_stats():
    report = bench.run_bench(
        workloads=["short"],
        configs=["naive", "numa", "full"],
        reps=4,
        simulate=True,
        collect_telemetry=False,
    )
    assert len(report.results) == 12
    assert len(report.stats) == 3
    by = {(stat.config, stat.workload): stat for stat in report.stats}
    assert by[("full", "short")].mean_tps > by[("naive", "short")].mean_tps
    assert "short" in report.significance


def test_welch_t_handles_tiny_samples():
    t_stat, p_value, df = bench._welch_t([1.0], [2.0])
    assert p_value == 1.0
    assert df == 0


def test_failed_runs_are_excluded_from_stats(monkeypatch):
    calls = {"count": 0}

    def fake_run(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return engine.EngineRun(
                simulated=False,
                command=["fake"],
                stdout="",
                stderr="boom",
                exit_code=1,
                wall_seconds=0.1,
                tokens_generated=0,
                tokens_per_second=0.0,
                message="real execution failed",
                measurement_valid=False,
            )
        return _valid_run()

    monkeypatch.setattr(bench.engine, "run", fake_run)
    report = bench.run_bench(
        workloads=["short"],
        configs=["naive"],
        reps=2,
        simulate=True,
        collect_telemetry=False,
    )
    assert len(report.results) == 2
    assert report.results[0].status == "failed"
    assert report.validity["failed_runs"] == 1
    assert report.validity["valid_runs"] == 1
    assert len(report.stats) == 1
    assert report.stats[0].n == 1


def test_missing_timing_output_is_invalid_measurement(monkeypatch):
    def fake_run(*args, **kwargs):
        return engine.EngineRun(
            simulated=False,
            command=["fake"],
            stdout="",
            stderr="",
            exit_code=0,
            wall_seconds=1.0,
            tokens_generated=0,
            tokens_per_second=0.0,
            message="real execution missing timing output",
            measurement_valid=False,
        )

    monkeypatch.setattr(bench.engine, "run", fake_run)
    report = bench.run_bench(
        workloads=["short"],
        configs=["naive", "full"],
        reps=1,
        simulate=True,
        collect_telemetry=False,
    )
    assert report.validity["invalid_runs"] == 2
    assert report.stats == []
    assert report.significance == {}


def test_strict_real_bench_rejects_low_reps(monkeypatch):
    monkeypatch.setattr(psutil, "swap_memory", lambda: type("Swap", (), {"used": 0})())
    monkeypatch.setattr(bench.engine, "run", lambda *args, **kwargs: _valid_run(simulated=False))
    try:
        bench.run_bench(
            workloads=["short"],
            configs=["naive"],
            reps=2,
            simulate=False,
            collect_telemetry=False,
            strict=True,
        )
    except bench.BenchValidationError as exc:
        assert any("below minimum" in failure for failure in exc.failures)
        assert exc.report.validity["passed_gates"] is False
    else:
        raise AssertionError("expected BenchValidationError")


def test_strict_real_bench_rejects_swap(monkeypatch):
    monkeypatch.setattr(psutil, "swap_memory", lambda: type("Swap", (), {"used": 64 * 1024 * 1024})())
    monkeypatch.setattr(bench.engine, "run", lambda *args, **kwargs: _valid_run(simulated=False))
    try:
        bench.run_bench(
            workloads=["short"],
            configs=["naive"],
            reps=30,
            simulate=False,
            collect_telemetry=False,
            strict=True,
        )
    except bench.BenchValidationError as exc:
        assert any("swap in use" in failure for failure in exc.failures)
    else:
        raise AssertionError("expected BenchValidationError")


def test_strict_linux_bench_rejects_control_failures(monkeypatch):
    monkeypatch.setattr(psutil, "swap_memory", lambda: type("Swap", (), {"used": 0})())
    linux_topo = topology.Topology(
        system="Linux",
        sockets=1,
        physical_cores=4,
        logical_cores=4,
        total_memory_mb=1024,
        numa_nodes=[topology.NumaNode(node_id=0, cpus=[0, 1, 2, 3], memory_mb=1024)],
        numa_available=True,
    )
    monkeypatch.setattr(bench.topology, "discover", lambda: linux_topo)
    monkeypatch.setattr(bench.engine, "run", lambda *args, **kwargs: _valid_run(simulated=False))

    def fake_try(cmd, label):
        return bench.ControlResult(ok=False, label=label, detail="permission denied")

    monkeypatch.setattr(bench, "_try_privileged", fake_try)
    try:
        bench.run_bench(
            workloads=["short"],
            configs=["naive"],
            reps=30,
            simulate=False,
            collect_telemetry=False,
            strict=True,
        )
    except bench.BenchValidationError as exc:
        assert any("disable NUMA balancing failed" in failure for failure in exc.failures)
        assert any("drop OS caches failed" in failure for failure in exc.failures)
    else:
        raise AssertionError("expected BenchValidationError")


def test_allow_invalid_overrides_strict_failures(monkeypatch):
    monkeypatch.setattr(psutil, "swap_memory", lambda: type("Swap", (), {"used": 64 * 1024 * 1024})())
    monkeypatch.setattr(bench.engine, "run", lambda *args, **kwargs: _valid_run(simulated=False))
    report = bench.run_bench(
        workloads=["short"],
        configs=["naive"],
        reps=30,
        simulate=False,
        collect_telemetry=False,
        strict=True,
        allow_invalid=True,
    )
    assert report.validity["passed_gates"] is False
    assert report.validity["allow_invalid"] is True


def test_cli_bench_returns_nonzero_on_gate_failure(monkeypatch):
    report = bench.BenchReport(
        topology_summary="fake topology",
        validity={
            "total_runs": 0,
            "valid_runs": 0,
            "invalid_runs": 0,
            "failed_runs": 0,
            "all_measurements_valid": True,
            "strict_mode": True,
            "allow_invalid": False,
            "gate_failures": ["swap in use at bench start: 64.0 MB"],
            "passed_gates": False,
        },
    )

    def fake_run_bench(**kwargs):
        raise bench.BenchValidationError(report=report, failures=report.validity["gate_failures"])

    monkeypatch.setattr(cli.bench_mod, "run_bench", fake_run_bench)
    monkeypatch.setattr(cli.bench_mod, "save_report", lambda report, out: Path(__file__))
    args = argparse.Namespace(
        workloads="short",
        configs="naive",
        reps=30,
        model=None,
        binary=None,
        simulate=False,
        no_telemetry=True,
        strict=False,
        allow_invalid=False,
        seed=42,
        out=".",
    )
    exit_code = cli._cmd_bench(args)
    assert exit_code == 1


def test_bench_persists_telemetry_summary_and_samples(monkeypatch):
    class FakeCollector:
        def __init__(self, interval_s=0.5, pid=None):
            self.interval_s = interval_s
            self.pid = pid
            self.started_with = None
            self.samples = [
                telemetry.TelemetrySample(
                    t=1.0,
                    pid=77,
                    cpu_percent=[10.0],
                    numa_local_mb={0: 10.0},
                    numa_remote_mb={0: 1.0},
                    numa_scope="process",
                    perf_cache_misses=5,
                    perf_cache_refs=50,
                    perf_llc_miss_rate=0.1,
                    mpstat_iowait_pct=0.0,
                )
            ]

        def start(self, pid=None):
            self.started_with = pid

        def stop(self):
            return telemetry.TelemetrySummary(
                duration_s=1.0,
                samples=1,
                mean_cpu_percent=10.0,
                remote_numa_fraction=0.09,
                numa_available=True,
                numa_scope="process",
                mean_llc_miss_rate=0.1,
                peak_iowait_pct=0.0,
                perf_available=True,
                mpstat_available=True,
            )

    def fake_run(*args, **kwargs):
        kwargs["on_start"](77)
        return engine.EngineRun(
            simulated=False,
            command=["fake"],
            stdout="ok",
            stderr="",
            exit_code=0,
            wall_seconds=1.0,
            tokens_generated=256,
            tokens_per_second=5.0,
            message="real execution completed",
            measurement_valid=True,
            pid=77,
        )

    monkeypatch.setattr(bench, "_try_privileged", lambda cmd, label: bench.ControlResult(ok=True, label=label))
    monkeypatch.setattr(bench.engine, "run", fake_run)
    monkeypatch.setattr(bench.telemetry, "TelemetryCollector", FakeCollector)
    monkeypatch.setattr(psutil, "swap_memory", lambda: type("Swap", (), {"used": 0})())
    report = bench.run_bench(
        workloads=["short"],
        configs=["naive"],
        reps=1,
        simulate=True,
        collect_telemetry=True,
    )
    assert report.results[0].telemetry_summary["numa_scope"] == "process"
    assert report.results[0].telemetry_summary["perf_available"] is True
    assert len(report.results[0].telemetry_samples) == 1
    assert report.results[0].telemetry_samples[0]["pid"] == 77


def test_bench_rotates_full_routing_across_numa_nodes(monkeypatch):
    routed_nodes = []
    fake_topology = topology.Topology(
        system="Windows",
        sockets=2,
        physical_cores=8,
        logical_cores=8,
        total_memory_mb=2048,
        numa_nodes=[
            topology.NumaNode(node_id=0, cpus=[0, 1, 2, 3], memory_mb=1024),
            topology.NumaNode(node_id=1, cpus=[4, 5, 6, 7], memory_mb=1024),
        ],
        numa_available=False,
    )

    def fake_run(decision, binding, *args, **kwargs):
        routed_nodes.append(binding.numa_node)
        return _valid_run()

    monkeypatch.setattr(bench.topology, "discover", lambda: fake_topology)
    monkeypatch.setattr(bench.engine, "run", fake_run)
    monkeypatch.setattr(bench, "_try_privileged", lambda cmd, label: bench.ControlResult(ok=True, label=label))
    monkeypatch.setattr(psutil, "swap_memory", lambda: type("Swap", (), {"used": 0})())
    report = bench.run_bench(
        workloads=["short"],
        configs=["full"],
        reps=4,
        simulate=True,
        collect_telemetry=False,
    )
    assert routed_nodes == [0, 1, 0, 1]
    assert report.results[0].binding_validation == []
