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
from pathlib import Path

from . import bench as bench_mod
from . import engine, memctl, router, topology
from . import ppl as ppl_mod
from .gguf import GGUFError, read_gguf_metadata


def _print_bench_report(report: bench_mod.BenchReport, path: str) -> None:
    print(report.topology_summary)
    if report.swap_mb_at_start > 0:
        print(f"\nWARNING: swap in use at bench start: {report.swap_mb_at_start} MB")
    if report.validity:
        print(
            "\nvalidity:"
            f" total={report.validity['total_runs']}"
            f" valid={report.validity['valid_runs']}"
            f" invalid={report.validity['invalid_runs']}"
            f" failed={report.validity['failed_runs']}"
        )
        for failure in report.validity.get("gate_failures", []):
            print(f"gate failure: {failure}")
    print()
    print(
        f"{'config':<8} {'workload':<10} {'n':>3} {'mean':>8} {'stdev':>7} {'cv%':>6} {'min':>7} {'max':>7} {'var_ok':>7}"
    )
    for stat in report.stats:
        ok = "OK" if stat.variance_ok else "WARN"
        print(
            f"{stat.config:<8} {stat.workload:<10} {stat.n:>3} {stat.mean_tps:>8.2f} {stat.stdev_tps:>7.2f}"
            f" {stat.cv_pct:>6.1f} {stat.min_tps:>7.2f} {stat.max_tps:>7.2f} {ok:>7}"
        )
    print()
    print("significance (full vs naive):")
    for workload, sig in report.significance.items():
        marker = "PASS" if sig["meets_25pct"] and sig["significant_p01"] else "----"
        print(
            f"  {workload:<10} uplift {sig['uplift_pct']:>6.2f}%  t={sig['t']:+.3f}  p={sig['p']:.4f}  [{marker}]"
        )
    print()
    print(f"report: {path}")


def _find_gguf_files(model_dir: str) -> list[Path]:
    """Return all .gguf files under *model_dir*, sorted by name."""
    return sorted(Path(model_dir).rglob("*.gguf"))


def _cmd_topo(args: argparse.Namespace) -> int:
    if getattr(args, "topo_cmd", None) == "list-models":
        return _cmd_list_models(args)
    t = topology.discover()
    if args.json:
        print(json.dumps(t.to_dict(), indent=2))
    else:
        print(t.summary())
    return 0


def _cmd_list_models(args: argparse.Namespace) -> int:
    model_dir = args.model_dir
    files = _find_gguf_files(model_dir)
    if not files:
        print(f"no .gguf files found in {model_dir!r}")
        return 0
    results = []
    for f in files:
        entry: dict = {"path": str(f), "size_mb": f.stat().st_size // (1024 * 1024)}
        try:
            meta = read_gguf_metadata(str(f))
            entry["version"] = meta.version
            entry["tensor_count"] = meta.tensor_count
            entry["quant_type"] = meta.quant_type
            entry["model_arch"] = meta.model_arch
            entry["param_count"] = meta.param_count
            entry["estimated_ram_mb"] = meta.estimated_ram_mb
            entry["valid"] = True
        except GGUFError as exc:
            entry["valid"] = False
            entry["error"] = str(exc)
        results.append(entry)
    if getattr(args, "json", False):
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            status = "OK" if r["valid"] else "INVALID"
            quant = r.get("quant_type") or "unknown"
            ram = r.get("estimated_ram_mb")
            ram_str = f"{ram} MB" if ram else "unknown"
            print(f"[{status}] {r['path']}  quant={quant}  ~{ram_str}")
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    t = topology.discover()
    cores = len(t.numa_nodes[0].cpus) if t.numa_nodes else (t.physical_cores or 4)
    decision = router.decide(
        args.input_tokens,
        cores_per_node=cores,
        numa_nodes=max(1, len(t.numa_nodes)),
        route_index=args.numa_node,
    )
    node = t.numa_nodes[decision.prefer_numa_node] if t.numa_nodes else None
    binding = memctl.build_binding(
        decision.prefer_numa_node,
        node.cpus if node else [],
    )
    binding_validation = router.validate_binding(decision, binding, t)
    out = {
        "workload": decision.workload.value,
        "rationale": decision.rationale,
        "routing_notes": decision.notes,
        "active_experts": decision.active_experts,
        "kv_cache_chunks": decision.kv_cache_chunks,
        "prefer_numa_node": decision.prefer_numa_node,
        "threads": decision.threads,
        "binding_enforced": binding.enforced,
        "binding_prefix": binding.prefix,
        "binding_notes": binding.notes,
        "binding_validation": binding_validation,
        "llama_flags": decision.as_flags(),
    }
    print(json.dumps(out, indent=2))
    return 0


def _resolve_model(args: argparse.Namespace) -> str | None:
    """Return model path from --model, or first GGUF found in --model-dir."""
    if getattr(args, "model", None):
        return args.model
    model_dir = getattr(args, "model_dir", None)
    if model_dir:
        files = _find_gguf_files(model_dir)
        if files:
            return str(files[0])
    return None


def _cmd_run(args: argparse.Namespace) -> int:
    model_path = _resolve_model(args)
    t = topology.discover()
    cores = len(t.numa_nodes[0].cpus) if t.numa_nodes else (t.physical_cores or 4)

    quant_type: str | None = None
    if model_path and not args.simulate:
        try:
            quant_type = read_gguf_metadata(model_path).quant_type
        except (GGUFError, Exception):
            pass

    decision = router.decide(
        args.input_tokens,
        cores_per_node=cores,
        numa_nodes=max(1, len(t.numa_nodes)),
        route_index=args.route_index,
        quant_type=quant_type,
    )
    node = t.numa_nodes[decision.prefer_numa_node] if t.numa_nodes else None
    binding = memctl.build_binding(decision.prefer_numa_node, node.cpus if node else [])
    binding_validation = router.validate_binding(decision, binding, t)
    result = engine.run(
        decision,
        binding,
        prompt=args.prompt,
        output_tokens=args.output_tokens,
        model_path=model_path,
        binary=args.binary,
        simulate=args.simulate,
        seed=args.seed,
    )
    out = {
        "mode": "simulated" if result.simulated else "real",
        "message": result.message,
        "routing_notes": decision.notes,
        "prefer_numa_node": decision.prefer_numa_node,
        "binding_validation": binding_validation,
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
    strict = args.strict or not args.simulate
    model_path = _resolve_model(args)
    try:
        report = bench_mod.run_bench(
            workloads=workloads,
            configs=configs,
            reps=args.reps,
            model_path=model_path,
            binary=args.binary,
            simulate=args.simulate,
            collect_telemetry=not args.no_telemetry,
            seed=args.seed,
            strict=strict,
            allow_invalid=args.allow_invalid,
        )
    except bench_mod.BenchValidationError as exc:
        path = bench_mod.save_report(exc.report, args.out)
        _print_bench_report(exc.report, str(path))
        return 1

    path = bench_mod.save_report(report, args.out)
    _print_bench_report(report, str(path))
    if strict and report.validity and not report.validity.get("passed_gates", True) and not args.allow_invalid:
        return 1
    return 0


def _cmd_ppl(args: argparse.Namespace) -> int:
    configs = args.configs.split(",") if args.configs else list(ppl_mod.PPL_CONFIGS)
    model_path = _resolve_model(args)
    results = ppl_mod.run_ppl(
        configs=configs,
        model_path=model_path,
        binary=args.binary,
        simulate=args.simulate,
        seed=args.seed,
    )
    degradation = ppl_mod.check_degradation(results)
    out = {
        "results": [
            {
                "config": r.config,
                "mode": "simulated" if r.simulated else "real",
                "message": r.message,
                "ppl": r.ppl,
                "ppl_stderr": r.ppl_stderr,
                "simulated": r.simulated,
                "exit_code": r.exit_code,
            }
            for r in results
        ],
        "degradation": degradation,
        "passes": all(v["passes"] for v in degradation.values()) and all(r.exit_code == 0 for r in results),
    }
    print(json.dumps(out, indent=2))
    return 0 if out["passes"] else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cmlis", description="CMLIS PoC orchestrator")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_topo = sub.add_parser("topo", help="print hardware topology")
    p_topo.add_argument("--json", action="store_true")
    topo_sub = p_topo.add_subparsers(dest="topo_cmd")
    p_list = topo_sub.add_parser("list-models", help="list GGUF models in a directory")
    p_list.add_argument("model_dir", help="directory to search for .gguf files")
    p_list.add_argument("--json", action="store_true")
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
    p_run.add_argument("--model-dir", default=None, dest="model_dir", help="directory to scan for GGUF models; first found is used")
    p_run.add_argument("--binary", default=None, help="path to llama.cpp binary")
    p_run.add_argument("--simulate", action="store_true")
    p_run.add_argument("--route-index", type=int, default=0, help="routing slot used to pick a NUMA node")
    p_run.add_argument("--seed", type=int, default=42)
    p_run.set_defaults(func=_cmd_run)

    p_bench = sub.add_parser("bench", help="run benchmark suite")
    p_bench.add_argument("--workloads", default="short,medium", help="comma-sep: short,medium,long")
    p_bench.add_argument("--configs", default="naive,numa,full")
    p_bench.add_argument("--reps", type=int, default=None)
    p_bench.add_argument("--model", default=None)
    p_bench.add_argument("--model-dir", default=None, dest="model_dir", help="directory to scan for GGUF models")
    p_bench.add_argument("--binary", default=None)
    p_bench.add_argument("--simulate", action="store_true")
    p_bench.add_argument("--no-telemetry", action="store_true")
    p_bench.add_argument("--strict", action="store_true", help="enforce methodology validity gates")
    p_bench.add_argument("--allow-invalid", action="store_true", help="override methodology validity gate failures")
    p_bench.add_argument("--seed", type=int, default=42)
    p_bench.add_argument("--out", default="./reports")
    p_bench.set_defaults(func=_cmd_bench)

    p_ppl = sub.add_parser("ppl", help="measure perplexity on WikiText-2")
    p_ppl.add_argument("--model", default=None, help="path to GGUF model")
    p_ppl.add_argument("--model-dir", default=None, dest="model_dir", help="directory to scan for GGUF models")
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
