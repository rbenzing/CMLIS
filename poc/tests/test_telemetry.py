from cmlis import telemetry


def test_read_numastat_scopes_to_pid(monkeypatch):
    monkeypatch.setattr(telemetry, "_have", lambda cmd: True)
    monkeypatch.setattr(telemetry.sys, "platform", "linux")

    captured = {}

    def fake_run(cmd, capture_output, text, timeout):
        captured["cmd"] = cmd

        class Result:
            stdout = "numa_hit 10 20\nnuma_miss 1 2\n"

        return Result()

    monkeypatch.setattr(telemetry.subprocess, "run", fake_run)
    local, remote, scope = telemetry._read_numastat(pid=123)
    assert captured["cmd"] == ["numastat", "-p", "123"]
    assert scope == "process"
    assert local == {0: 10.0, 1: 20.0}
    assert remote == {0: 1.0, 1: 2.0}


def test_collector_stop_without_start_reports_unavailable():
    collector = telemetry.TelemetryCollector()
    summary = collector.stop()
    assert summary.samples == 0
    assert summary.numa_available is False
    assert summary.remote_numa_fraction is None
    assert summary.perf_available is False
    assert summary.mpstat_available is False


def test_summary_distinguishes_unavailable_from_zero():
    collector = telemetry.TelemetryCollector()
    collector.samples = [
        telemetry.TelemetrySample(t=1.0, pid=7, cpu_percent=[0.0], numa_scope="unavailable"),
        telemetry.TelemetrySample(t=2.0, pid=7, cpu_percent=[0.0], numa_scope="unavailable"),
    ]
    collector._thread = object()
    collector._t_start = 1.0

    class DummyThread:
        def join(self, timeout):
            return None

    collector._thread = DummyThread()
    summary = collector.stop()
    assert summary.mean_cpu_percent == 0.0
    assert summary.numa_available is False
    assert summary.numa_scope == "unavailable"
    assert summary.remote_numa_fraction is None
