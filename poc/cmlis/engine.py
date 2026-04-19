"""Inference engine wrapper.

Launches `llama.cpp` (llama-cli / main binary) with the composed command
prefix from memctl and flags from the router. If the binary or a model
file is unavailable, runs in simulation mode that synthesises plausible
tokens/sec numbers so the orchestration path can be exercised end to end.
"""

from __future__ import annotations

import os
import random
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .memctl import BindingPlan
from .router import RoutingDecision, WorkloadClass


@dataclass
class EngineRun:
    simulated: bool
    command: list[str]
    stdout: str
    stderr: str
    exit_code: int
    wall_seconds: float
    tokens_generated: int
    tokens_per_second: float
    pid: int | None = None  # PID of the llama.cpp process; None for simulated runs


def _default_binary() -> str | None:
    for name in ("llama-cli", "main", "llama.cpp"):
        p = shutil.which(name)
        if p:
            return p
    env = os.environ.get("LLAMA_CPP_BIN")
    if env and Path(env).exists():
        return env
    return None


def _parse_tps(output: str) -> tuple[int, float]:
    """Parse llama.cpp eval output for (tokens, tokens/sec).

    Real llama.cpp llama_print_timings format:
      llama_print_timings:        eval time =   1234.56 ms /   256 runs   (    4.82 ms per token,   207.47 tokens per second)

    The TPS value appears after 'tokens per second' which is preceded by a comma
    and optional whitespace, not by a leading '('.
    """
    # Capture: / <runs> runs ... <tps> tokens per second)
    m = re.search(r"/\s*(\d+)\s*runs[^)]*,\s*([\d.]+)\s*tokens per second\)", output)
    if m:
        return int(m.group(1)), float(m.group(2))
    return 0, 0.0


def _simulate(
    decision: RoutingDecision,
    binding: BindingPlan,
    output_tokens: int,
    seed: int,
    config: str = "full",
) -> EngineRun:
    """Generate plausible synthetic numbers for dry-run validation.

    numa_bonus is derived from the config name so that simulation correctly
    differentiates naive / numa / full even on non-Linux where binding.enforced
    is always False (no numactl available).
    """
    rng = random.Random(seed)
    base = {
        WorkloadClass.SHORT: 6.5,
        WorkloadClass.MEDIUM: 4.2,
        WorkloadClass.LONG: 2.6,
        WorkloadClass.MIXED: 4.0,
    }[decision.workload]

    # binding.enforced is False on Windows/macOS simulation — use config name instead.
    numa_bonus = 1.35 if (binding.enforced or config in ("numa", "full")) else 1.0
    router_bonus = 1.15 if decision.active_experts and decision.active_experts <= 2 else 1.05
    noise = rng.uniform(0.93, 1.07)
    tps = base * numa_bonus * router_bonus * noise
    wall = output_tokens / tps

    return EngineRun(
        simulated=True,
        command=["<simulated>"],
        stdout=f"simulated run: {output_tokens} tokens @ {tps:.2f} tok/s",
        stderr="",
        exit_code=0,
        wall_seconds=wall,
        tokens_generated=output_tokens,
        tokens_per_second=tps,
    )


def run(
    decision: RoutingDecision,
    binding: BindingPlan,
    prompt: str,
    output_tokens: int,
    model_path: str | None = None,
    binary: str | None = None,
    extra_flags: list[str] | None = None,
    simulate: bool = False,
    seed: int = 0,
    timeout: float = 600.0,
    config: str = "full",
) -> EngineRun:
    """Run one inference job with the composed configuration."""
    import sys

    binary = binary or _default_binary()
    model_ok = model_path and Path(model_path).exists()

    if not simulate and binary and not model_ok and model_path:
        print(
            f"WARNING: binary found but model not found at {model_path!r}; running in simulation mode",
            file=sys.stderr,
        )

    if simulate or not binary or not model_ok:
        return _simulate(decision, binding, output_tokens, seed, config=config)

    cmd = list(binding.prefix) + [
        binary,
        "-m",
        model_path,
        "-p",
        prompt,
        "-n",
        str(output_tokens),
        "--seed",
        str(seed),
    ]
    cmd.extend(decision.as_flags())
    if extra_flags:
        cmd.extend(extra_flags)

    t0 = time.perf_counter()
    try:
        # Use Popen so we can expose the PID to the telemetry collector.
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        pid = proc.pid
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            return EngineRun(
                simulated=False,
                command=cmd,
                stdout=stdout or "",
                stderr=f"timeout after {timeout}s",
                exit_code=124,
                wall_seconds=timeout,
                tokens_generated=0,
                tokens_per_second=0.0,
                pid=pid,
            )
        wall = time.perf_counter() - t0
        toks, tps = _parse_tps(stdout + "\n" + stderr)
        if toks == 0:
            toks = output_tokens
            tps = toks / wall if wall > 0 else 0.0
        return EngineRun(
            simulated=False,
            command=cmd,
            stdout=stdout,
            stderr=stderr,
            exit_code=proc.returncode,
            wall_seconds=wall,
            tokens_generated=toks,
            tokens_per_second=tps,
            pid=pid,
        )
    except OSError as e:
        return EngineRun(
            simulated=False,
            command=cmd,
            stdout="",
            stderr=str(e),
            exit_code=1,
            wall_seconds=0.0,
            tokens_generated=0,
            tokens_per_second=0.0,
            pid=None,
        )
