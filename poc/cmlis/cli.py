"""CMLIS PoC command-line interface.

Subcommands:
  topo      Print discovered hardware topology.
  plan      Show the routing decision and binding prefix for a single job.
  run       Execute one inference job (real or simulated).
  bench     Run the full benchmark suite across configs and workloads.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import bench as bench_mod
from . import engine, memctl, router, topology
from . import ppl as ppl_mod


def _cmd_topo(args: argparse.Namespace) -> int:
    t = topology.discover()
    if args.json:
        print(json.dumps(t.to_dict(), indent=2))
    else:
        print(t.summary())
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    t = topology.discover()
    cores = len(t.numa_nodes[0].cpus) if t.numa_nodes else (t.physical_cores or 4)
    decision = router.decide(args.input_tokens, cores_per_node=cores, numa_node=args.numa_node)
    node = t.numa_nodes[decision.prefer_numa_node] if t.numa_nodes else None
    binding = memctl.build_binding(
        decision.prefer_numa_node,
        node.cpus if node else [],
    )
    out = {
        "workload": decision.workload.value,
        "rationale": decision.rationale,
        "active_experts": decision.active_experts,
        "kv_cache_chunks": decision.kv_cache_chunks,
        "threads": decision.threads,
        "binding_enforced": binding.enforced,
        "binding_prefix": binding.prefix,
        "binding_notes": binding.notes,
        "llama_flags": decision.as_flags(),
    }
    print(json.dumps(out, indent=2))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    t = topology.discover()
    cores = len(t.numa_nodes[0].cpus) if t.numa_nodes else (t.physical_cores or 4)
    decision = router.decide(args.input_tokens, cores_per_node=cores, numa_node=0)
    node = t.numa_nodes[0] if t.numa_nodes else None
    binding = memctl.build_binding(0, node.cpus if node else [])
    result = engine.run(
        decision,
        binding,
        prompt=args.prompt,
        output_tokens=args.output_tokens,
        model_path=args.model,
        binary=args.binary,
        simulate=args.simulate,
        seed=args.seed,
    )
    out = {
        "simulated": result.simulated,
        "command": result.command,
        "exit_code": result.exit_code,
        "wall_seconds": round(result.wall_seconds, 3),
        "tokens": result.tokens_generated,
        "tokens_per_second": round(result.tokens_per_second, 3),
    }
    print(json.dumps(out, indent=2))
    return 0 if result.exit_code == 0 else result.exit_code


def _cmd_bench(args: argparse.Namespace) -> int:
    workloads = args.workloads.split(",") if args.workloads else ["short", "medium"]
    configs = args.configs.split(",") if args.configs else list(bench_mod.CONFIGS)
    report = bench_mod.run_bench(
        workloads=workloads,
        configs=configs,
        reps=args.reps,
        model_path=args.model,
        binary=args.binary,
        simulate=args.simulate,
        collect_telemetry=not args.no_telemetry,
        seed=args.seed,
    )
    path = bench_mod.save_report(report, args.out)
    print(report.topology_summary)
    if report.swap_mb_at_start > 0:
        print(f"\nWARNING: swap in use at bench start: {report.swap_mb_at_start} MB")
    print()
    print(
        f"{'config':<8} {'workload':<10} {'n':>3} {'mean':>8} {'stdev':>7} {'cv%':>6} {'min':>7} {'max':>7} {'var_ok':>7}"
    )
    for s in report.stats:
        ok = "OK" if s.variance_ok else "WARN"
        print(
            f"{s.config:<8} {s.workload:<10} {s.n:>3} {s.mean_tps:>8.2f} {s.stdev_tps:>7.2f}"
            f" {s.cv_pct:>6.1f} {s.min_tps:>7.2f} {s.max_tps:>7.2f} {ok:>7}"
        )
    print()
    print("significance (full vs naive):")
    for wl, sig in report.significance.items():
        marker = "PASS" if sig["meets_25pct"] and sig["significant_p01"] else "----"
        print(
            f"  {wl:<10} uplift {sig['uplift_pct']:>6.2f}%  t={sig['t']:+.3f}  p={sig['p']:.4f}  [{marker}]"
        )
    print()
    print(f"report: {path}")
    return 0


def _cmd_ppl(args: argparse.Namespace) -> int:
    configs = args.configs.split(",") if args.configs else list(ppl_mod.PPL_CONFIGS)
    results = ppl_mod.run_ppl(
        configs=configs,
        model_path=args.model,
        binary=args.binary,
        simulate=args.simulate,
        seed=args.seed,
    )
    degradation = ppl_mod.check_degradation(results)
    out = {
        "results": [
            {"config": r.config, "ppl": r.ppl, "ppl_stderr": r.ppl_stderr, "simulated": r.simulated}
            for r in results
        ],
        "degradation": degradation,
        "passes": all(v["passes"] for v in degradation.values()),
    }
    print(json.dumps(out, indent=2))
    return 0 if out["passes"] else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cmlis", description="CMLIS PoC orchestrator")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_topo = sub.add_parser("topo", help="print hardware topology")
    p_topo.add_argument("--json", action="store_true")
    p_topo.set_defaults(func=_cmd_topo)

    p_plan = sub.add_parser("plan", help="show routing + binding plan")
    p_plan.add_argument("--input-tokens", type=int, default=2048)
    p_plan.add_argument("--numa-node", type=int, default=0)
    p_plan.set_defaults(func=_cmd_plan)

    p_run = sub.add_parser("run", help="run a single inference job")
    p_run.add_argument("--prompt", default="The capital of France is")
    p_run.add_argument("--input-tokens", type=int, default=2048)
    p_run.add_argument("--output-tokens", type=int, default=128)
    p_run.add_argument("--model", default=None, help="path to GGUF model")
    p_run.add_argument("--binary", default=None, help="path to llama.cpp binary")
    p_run.add_argument("--simulate", action="store_true")
    p_run.add_argument("--seed", type=int, default=42)
    p_run.set_defaults(func=_cmd_run)

    p_bench = sub.add_parser("bench", help="run benchmark suite")
    p_bench.add_argument("--workloads", default="short,medium", help="comma-sep: short,medium,long")
    p_bench.add_argument("--configs", default="naive,numa,full")
    p_bench.add_argument("--reps", type=int, default=None)
    p_bench.add_argument("--model", default=None)
    p_bench.add_argument("--binary", default=None)
    p_bench.add_argument("--simulate", action="store_true")
    p_bench.add_argument("--no-telemetry", action="store_true")
    p_bench.add_argument("--seed", type=int, default=42)
    p_bench.add_argument("--out", default="./reports")
    p_bench.set_defaults(func=_cmd_bench)

    p_ppl = sub.add_parser("ppl", help="measure perplexity on WikiText-2")
    p_ppl.add_argument("--model", default=None, help="path to GGUF model")
    p_ppl.add_argument("--binary", default=None, help="path to llama.cpp binary")
    p_ppl.add_argument("--simulate", action="store_true")
    p_ppl.add_argument("--seed", type=int, default=42)
    p_ppl.add_argument("--configs", default="naive,full", help="comma-sep configs to test")
    p_ppl.set_defaults(func=_cmd_ppl)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
