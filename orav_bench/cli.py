"""Command line entry for evaluation, ranking, and pointwise diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import load_manifest, read_json, write_json


def rank(args):
    from . import ranking

    comparisons = read_json(args.results)
    run = read_json(Path(args.results).with_name("run.json"))
    systems = comparisons["systems"]
    tasks = {t["identity"]: t for t in run["tasks"]}
    selected = {
        key
        for key, task in tasks.items()
        if (args.scope == "all" or all(c["state"] == "delivered" for c in task["candidates"].values()))
        and (not args.role or args.role in task["combination"])
        and (not args.no_audio_reference or not set(task["combination"]) & {"speech", "music"})
    }
    rows = []
    for comparison in comparisons["comparisons"]:
        if comparison["task_identity"] not in selected:
            continue
        fused = comparison["fused"]
        kind = (
            ("tie" if fused["winner"] == "tie" else "decided")
            if fused["status"] == "ok"
            else ("conflict" if fused["status"] == "order_conflict" else "judge_unavailable")
        )
        rows.append(
            {
                "task_identity": comparison["task_identity"],
                "systems": comparison["systems"],
                "kind": kind,
                "winner": fused["winner"],
            }
        )
    if not rows:
        raise ValueError("No comparisons in the requested subset")
    policy = "conflict_dropped" if args.exclude_conflicts else "conflict_as_tie"
    result = ranking.fit(rows, systems, policy, args.model)
    result["subset"] = {
        "scope": args.scope,
        "role": args.role,
        "no_audio_reference": args.no_audio_reference,
        "instances": len(selected),
    }
    result["coverage"] = {
        s: {
            "delivered": sum(t["candidates"][s]["state"] == "delivered" for t in tasks.values()),
            "instances": len(tasks),
        }
        for s in systems
    }
    if args.bootstrap:
        result["bootstrap"] = ranking.cluster_bootstrap(
            rows, systems, policy, args.model, args.bootstrap, args.seed
        )
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="orav", description="ORAV evaluation and diagnostic measurements")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "evaluate"):
        p = commands.add_parser(
            name,
            help="Prepare media offline" if name == "prepare" else "Run the listener and ordered comparisons",
        )
        p.add_argument("manifest", type=Path)
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--workers", type=int, default=8)
    p = commands.add_parser("rank", help="Fit system strengths and bootstrap confidence intervals")
    p.add_argument("results", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scope", choices=("shared", "all"), default="shared")
    p.add_argument("--role")
    p.add_argument("--no-audio-reference", action="store_true")
    p.add_argument("--exclude-conflicts", action="store_true")
    p.add_argument("--model", choices=("davidson", "rao_kupper", "bt_half_credit"), default="davidson")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p = commands.add_parser("diagnostics", help="Compute the paper's pointwise diagnostics")
    sub = p.add_subparsers(dest="diagnostic_command", required=True)
    q = sub.add_parser("run")
    q.add_argument("manifest", type=Path)
    q.add_argument("--config", type=Path, required=True)
    q.add_argument("--metrics", nargs="+", required=True)
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--device", default="cuda:0")
    q.add_argument("--shard", type=int, default=0)
    q.add_argument("--shards", type=int, default=1)
    q = sub.add_parser("summarize")
    q.add_argument("manifest", type=Path)
    q.add_argument("results", type=Path)
    q.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command in ("evaluate", "prepare"):
        from .runner import evaluate

        if args.workers < 1:
            parser.error("--workers must be positive")
        evaluate(
            load_manifest(args.manifest),
            read_json(args.config),
            args.output,
            args.workers,
            args.command == "prepare",
        )
    elif args.command == "rank":
        if args.bootstrap < 0:
            parser.error("--bootstrap must be nonnegative")
        rank(args)
    else:
        from .diagnostics import run, summarize

        if args.diagnostic_command == "run":
            run(args)
        else:
            summarize(args)


if __name__ == "__main__":
    main()
