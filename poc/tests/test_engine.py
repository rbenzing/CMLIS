"""Tests for engine.py."""

import pytest

from cmlis import engine
from cmlis.engine import _parse_tps, _simulate
from cmlis.memctl import BindingPlan
from cmlis.router import RoutingDecision, WorkloadClass

# Real llama.cpp llama_print_timings output samples (from different versions).
LLAMA_OUTPUT_SAMPLES = [
    (
        "llama_print_timings:        eval time =   1234.56 ms /   256 runs   (    4.82 ms per token,   207.47 tokens per second)",
        256,
        207.47,
    ),
    (
        "llama_print_timings:        eval time =    512.00 ms /    64 runs   (    8.00 ms per token,   125.00 tokens per second)",
        64,
        125.00,
    ),
    (
        "llama_print_timings:        eval time =    100.00 ms /     3 runs   (   33.33 ms per token,    30.00 tokens per second)",
        3,
        30.00,
    ),
    (
        "some noise\nllama_print_timings:        eval time =   8192.00 ms /   512 runs   (   16.00 ms per token,    62.50 tokens per second)\nmore noise",
        512,
        62.50,
    ),
]


def _decision() -> RoutingDecision:
    return RoutingDecision(
        workload=WorkloadClass.MEDIUM,
        active_experts=0,
        kv_cache_chunks=2,
        prefer_numa_node=0,
        threads=7,
        rationale="test",
    )


@pytest.mark.parametrize("text,expected_tokens,expected_tps", LLAMA_OUTPUT_SAMPLES)
def test_parse_tps_real_output(text, expected_tokens, expected_tps):
    tokens, tps = _parse_tps(text)
    assert tokens == expected_tokens
    assert abs(tps - expected_tps) < 0.01


def test_parse_tps_no_match_returns_zeros():
    tokens, tps = _parse_tps("no timing info here")
    assert tokens == 0
    assert tps == 0.0


def test_simulate_naive_vs_full_differ():
    """Full config should outperform naive in simulation."""
    decision = RoutingDecision(
        workload=WorkloadClass.SHORT,
        active_experts=2,
        kv_cache_chunks=1,
        prefer_numa_node=0,
        threads=7,
        rationale="test",
    )
    binding_noop = BindingPlan(numa_node=0, cpus=[], enforced=False, prefix=[], notes=[])
    naive = _simulate(decision, binding_noop, output_tokens=256, seed=42, config="naive")
    full = _simulate(decision, binding_noop, output_tokens=256, seed=42, config="full")
    assert full.tokens_per_second > naive.tokens_per_second


def test_simulate_medium_numa_beats_naive():
    """For medium workload (active_experts=0), numa should still beat naive via numa_bonus."""
    decision = _decision()
    binding_noop = BindingPlan(numa_node=0, cpus=[], enforced=False, prefix=[], notes=[])
    naive = _simulate(decision, binding_noop, output_tokens=512, seed=42, config="naive")
    numa = _simulate(decision, binding_noop, output_tokens=512, seed=42, config="numa")
    assert numa.tokens_per_second > naive.tokens_per_second


def test_simulate_is_deterministic():
    """Same seed and config must produce identical results."""
    decision = _decision()
    binding = BindingPlan(numa_node=0, cpus=[], enforced=False, prefix=[], notes=[])
    r1 = _simulate(decision, binding, output_tokens=512, seed=99, config="full")
    r2 = _simulate(decision, binding, output_tokens=512, seed=99, config="full")
    assert r1.tokens_per_second == r2.tokens_per_second


def test_run_fails_closed_when_binary_missing(monkeypatch):
    monkeypatch.setattr(engine, "resolve_binary", lambda binary=None: None)
    binding = BindingPlan(numa_node=0, cpus=[], enforced=False, prefix=[], notes=[])
    result = engine.run(_decision(), binding, prompt="x", output_tokens=16, simulate=False)
    assert result.simulated is False
    assert result.exit_code == 2
    assert "binary not found" in result.message


def test_run_fails_closed_when_model_missing(monkeypatch):
    monkeypatch.setattr(engine, "resolve_binary", lambda binary=None: "fake-binary")
    binding = BindingPlan(numa_node=0, cpus=[], enforced=False, prefix=[], notes=[])
    result = engine.run(_decision(), binding, prompt="x", output_tokens=16, model_path="missing.gguf")
    assert result.simulated is False
    assert result.exit_code == 2
    assert "model not found" in result.message


def test_failed_real_subprocess_does_not_report_throughput(monkeypatch):
    class FakeProc:
        pid = 123
        returncode = 1

        def communicate(self, timeout=None):
            return "", "bad run"

    monkeypatch.setattr(engine, "resolve_binary", lambda binary=None: "fake-binary")
    monkeypatch.setattr(engine.subprocess, "Popen", lambda *args, **kwargs: FakeProc())
    monkeypatch.setattr(engine.Path, "exists", lambda self: True)
    binding = BindingPlan(numa_node=0, cpus=[], enforced=False, prefix=[], notes=[])
    result = engine.run(_decision(), binding, prompt="x", output_tokens=16, model_path="model.gguf")
    assert result.exit_code == 1
    assert result.tokens_generated == 0
    assert result.tokens_per_second == 0.0
    assert result.measurement_valid is False
