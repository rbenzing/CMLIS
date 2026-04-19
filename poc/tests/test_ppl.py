"""Tests for ppl.py."""

from pathlib import Path

from cmlis import ppl


def test_simulated_run_returns_both_configs():
    results = ppl.run_ppl(configs=["naive", "full"], simulate=True, seed=42)
    assert len(results) == 2
    configs = {result.config for result in results}
    assert configs == {"naive", "full"}
    assert all(result.simulated for result in results)
    assert all(result.ppl > 0 for result in results)


def test_simulated_ppl_within_threshold():
    """Simulated full PPL must be within 0.1 of naive (SPEC coherence constraint)."""
    results = ppl.run_ppl(configs=["naive", "full"], simulate=True, seed=42)
    degradation = ppl.check_degradation(results)
    assert degradation["full"]["passes"], (
        f"Simulated full PPL delta {degradation['full']['delta']:.4f} exceeds 0.1 threshold"
    )


def test_check_degradation_flags_excessive_delta():
    """Delta > 0.1 must be flagged as failing."""
    results = [
        ppl.PplResult(config="naive", ppl=4.23, ppl_stderr=0.01, simulated=True),
        ppl.PplResult(config="full", ppl=4.45, ppl_stderr=0.01, simulated=True),
    ]
    degradation = ppl.check_degradation(results)
    assert not degradation["full"]["passes"]
    assert abs(degradation["full"]["delta"] - 0.22) < 1e-6


def test_check_degradation_passes_within_threshold():
    results = [
        ppl.PplResult(config="naive", ppl=4.23, ppl_stderr=0.01, simulated=True),
        ppl.PplResult(config="full", ppl=4.30, ppl_stderr=0.01, simulated=True),
    ]
    degradation = ppl.check_degradation(results)
    assert degradation["full"]["passes"]


def test_parse_ppl_real_output():
    """Verify regex against real llama.cpp --perplexity output."""
    sample = "Final estimate: PPL = 4.2345 +/- 0.0123"
    ppl_val, ppl_err = ppl._parse_ppl(sample)
    assert abs(ppl_val - 4.2345) < 1e-4
    assert abs(ppl_err - 0.0123) < 1e-4


def test_parse_ppl_no_match_returns_zero():
    ppl_val, ppl_err = ppl._parse_ppl("no perplexity output here")
    assert ppl_val == 0.0
    assert ppl_err is None


def test_naive_only_run():
    """Single-config run must still work and return one result."""
    results = ppl.run_ppl(configs=["naive"], simulate=True)
    assert len(results) == 1
    assert results[0].config == "naive"


def test_real_ppl_fails_closed_when_binary_missing(monkeypatch):
    monkeypatch.setattr(ppl.engine, "resolve_binary", lambda binary=None: None)
    results = ppl.run_ppl(configs=["naive"], simulate=False, model_path="model.gguf")
    assert len(results) == 1
    assert results[0].simulated is False
    assert results[0].exit_code == 2
    assert "binary not found" in results[0].message


def test_real_ppl_fails_closed_when_model_missing(monkeypatch):
    monkeypatch.setattr(ppl.engine, "resolve_binary", lambda binary=None: "fake-binary")
    results = ppl.run_ppl(configs=["naive"], simulate=False, model_path="missing.gguf")
    assert len(results) == 1
    assert results[0].simulated is False
    assert results[0].exit_code == 2
    assert "model not found" in results[0].message


def test_real_ppl_fails_closed_when_dataset_unavailable(monkeypatch):
    model_path = Path(__file__)
    monkeypatch.setattr(ppl.engine, "resolve_binary", lambda binary=None: "fake-binary")

    def fail_download():
        raise OSError("network unavailable")

    monkeypatch.setattr(ppl, "_download_wikitext2", fail_download)
    results = ppl.run_ppl(configs=["naive"], simulate=False, model_path=str(model_path))
    assert len(results) == 1
    assert results[0].simulated is False
    assert results[0].exit_code == 2
    assert "failed to prepare WikiText-2 dataset" in results[0].message
