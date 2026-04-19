from cmlis import bench


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
    # Full should beat naive in the simulator
    by = {(s.config, s.workload): s for s in report.stats}
    assert by[("full", "short")].mean_tps > by[("naive", "short")].mean_tps
    assert "short" in report.significance


def test_welch_t_handles_tiny_samples():
    t, p, df = bench._welch_t([1.0], [2.0])
    assert p == 1.0
    assert df == 0
