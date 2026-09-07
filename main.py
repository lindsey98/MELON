#!/usr/bin/env python
"""Unified command-line entry point for the MELON / AgentDojo benchmark.

Attack run:
    python main.py MODEL --run-attack \
        --attack important_instructions \
        --suites banking slack travel workspace \
        --defense melon

No-attack (utility) run:
    python main.py MODEL \
        --suites banking slack travel workspace \
        --defense melon

MODEL is the model name (e.g. gpt-4o-2024-05-13, or a local served name like
Qwen3.6-35B-A3B). `--defense` is None (omit) or `melon`. Output always goes to
`logs/`. `--force_rerun` and `--html` are optional. Omit `--suites` to run all
suites in the benchmark version.
"""

import argparse
import logging
from pathlib import Path

from rich.logging import RichHandler

from agentdojo.agent_pipeline.agent_pipeline import DEFENSES
from agentdojo.models import ModelsEnum
from agentdojo.scripts.benchmark import run_benchmark

# Output directory is fixed (not a CLI option).
LOGDIR = Path("logs")


def _defense(value: str) -> str | None:
    """`None`/`none` -> no defense; otherwise must be a registered defense (e.g. melon)."""
    if value is None or value.lower() == "none":
        return None
    if value not in DEFENSES:
        raise argparse.ArgumentTypeError(f"--defense must be None or one of {DEFENSES}, got {value!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Run the MELON / AgentDojo benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "model",
        metavar="MODEL",
        choices=[m.value for m in ModelsEnum],
        help="Model to benchmark (e.g. gpt-4o-2024-05-13, or a local served name like Qwen3.6-35B-A3B).",
    )
    p.add_argument(
        "--run-attack",
        action="store_true",
        help="Run with an injection attack. Without this flag the run is attack-free (utility only).",
    )
    p.add_argument(
        "--attack",
        default="important_instructions",
        help="Attack to use when --run-attack is set (e.g. important_instructions).",
    )
    p.add_argument(
        "--suites",
        nargs="+",
        default=[],
        metavar="SUITE",
        help="Space-separated suites, e.g. --suites banking slack travel workspace. Omit to run all suites.",
    )
    p.add_argument(
        "--defense",
        type=_defense,
        default=None,
        help="Defense to apply: None (omit) or melon.",
    )
    # Optional, unchanged in behaviour.
    p.add_argument("--force_rerun", "--force-rerun", "-f", dest="force_rerun", action="store_true",
                   help="Re-run tasks even if a cached result JSON already exists.")
    p.add_argument("--html", action="store_true",
                   help="Also write a rendered HTML trace next to each per-task JSON.")
    p.add_argument("--user-task", "-ut", dest="user_tasks", action="append", default=[], metavar="ID",
                   help="Restrict to these user tasks (repeatable). Only when a single suite is run.")
    p.add_argument("--injection-task", "-it", dest="injection_tasks", action="append", default=[], metavar="ID",
                   help="Restrict to these injection tasks (repeatable).")
    # Advanced / pass-through options.
    p.add_argument("--model-id", default=None, help="Model id for local models (OpenAI-compatible endpoints).")
    p.add_argument("--benchmark-version", default="v1.2", help="Benchmark suite version.")
    p.add_argument("--tool-delimiter", default="tool", help="Tool delimiter (local models only).")
    p.add_argument("--system-message", default=None, help="Override system message (string).")
    p.add_argument("--system-message-name", default=None, help="Named system message from system_messages.yaml.")
    p.add_argument("--max-workers", type=int, default=1, help="Parallel suites via multiprocessing.")
    p.add_argument("--module-to-load", "-ml", dest="modules_to_load", action="append", default=[], metavar="MODULE",
                   help="Import a module before running (to register custom attacks/defenses/suites).")
    return p


def main() -> None:
    args = build_parser().parse_args()

    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
        handlers=[RichHandler(show_path=False, markup=True)],
    )

    attack = args.attack if args.run_attack else None

    run_benchmark(
        suites=tuple(args.suites),
        model=args.model,
        benchmark_version=args.benchmark_version,
        logdir=LOGDIR,
        user_tasks=tuple(args.user_tasks),
        injection_tasks=tuple(args.injection_tasks),
        model_id=args.model_id,
        attack=attack,
        defense=args.defense,
        tool_delimiter=args.tool_delimiter,
        system_message_name=args.system_message_name,
        system_message=args.system_message,
        max_workers=args.max_workers,
        force_rerun=args.force_rerun,
        modules_to_load=tuple(args.modules_to_load),
        emit_html=args.html,
    )


if __name__ == "__main__":
    main()
