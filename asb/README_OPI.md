# ASB Observation Prompt Injection (OPI) — experiment guide

One-file reference for running the ASB OPI benchmark in this repo, defended by
**MELON** (and optionally CaMeL), on a locally-served or hosted model. This is the
OPI slice of [ASB](https://github.com/agiresearch/ASB) reworked into a
run-at-scale tool; the original ASB README is [`ASB_README.md`](ASB_README.md),
and the MELON integration is described in [`README.md`](README.md).

## What this setup adds over stock ASB

| Capability | How | Where |
|---|---|---|
| Local / OpenAI-compatible agent LLM (vLLM, SGLang, …) | `--use_backend local`, `LOCAL_BASE_URL`/`LOCAL_API_KEY` | `aios/llm_core/llm_classes/local_llm.py` |
| ReAct dynamic tool-calling loop (full OPI attack surface) | `--workflow_mode react` | `pyopenagi/agents/react_agent_attack.py` (`run_react`) |
| AgentDojo-format output: 4-role messages, `TaskResults` JSON, `tool_trace`, real `duration` | per task | `main_attacker.py`, `react_agent_attack.py` |
| **Single-point injection** (realistic: one compromised source) | `--opi_inject_limit 1` (`0` = every observation, ASB default) | `react_agent_attack.py:call_tools` |
| **MELON defense** (contrastive detector) | `--defense_type melon` | `pyopenagi/agents/melon.py` |
| CaMeL defense (needs camel-prompt-injection kernel; kept for parity) | `--defense_type camel` | `camel_adapter/` |
| Resume / caching | tasks with an existing JSON trace are skipped and their metrics reused; `--force_rerun` recomputes | `main_attacker.py` |
| Progress bar + clean logs | `tqdm` with live ASR/util; one line per task; `ASB_VERBOSE=1` restores full logs | `main_attacker.py` |
| Configurable refuse judge (independent of the agent LLM) | `ASB_JUDGE_MODEL` / `ASB_JUDGE_BASE_URL` / `ASB_JUDGE_API_KEY` | `main_attacker.py` |
| Aggregation + baseline-vs-defense diff | `scripts/analyze.py` | — |

## Install

```bash
cd asb
pip install -r requirements-opi.txt
# MELON detection embeddings — pick ONE backend:
pip install sentence-transformers   # fully local, no API
#   or rely on OpenAI embeddings (needs OPENAI_API_KEY)
```

## Serve the agent model (example: vLLM, OpenAI-compatible)

```bash
vllm serve Qwen/Qwen3.6-35B-A3B --served-model-name Qwen3.6-35B-A3B --port 8000 \
    --enable-auto-tool-choice --tool-call-parser hermes
export LOCAL_BASE_URL=http://localhost:8000/v1   # defaults to this
export LOCAL_API_KEY=EMPTY
```

## Configure MELON's embedding backend (independent of the agent LLM)

```bash
export MELON_EMBED_PROVIDER=sentence-transformers      # fully local
export MELON_EMBED_MODEL=BAAI/bge-large-en-v1.5        # or a local path
# — or hosted:            MELON_EMBED_PROVIDER=openai + OPENAI_API_KEY
# — or a local server:    MELON_EMBED_PROVIDER=openai-compatible + MELON_EMBED_BASE_URL
```

## Run

The one-stop script (defaults: ReAct loop, single-point injection, resume on):

```bash
cd asb
MODEL=Qwen3.6-35B-A3B bash scripts/run_opi.sh            # undefended baseline
MODEL=Qwen3.6-35B-A3B bash scripts/run_opi.sh melon      # MELON
MODEL=Qwen3.6-35B-A3B bash scripts/run_opi.sh camel      # CaMeL (needs kernel)

# clean-utility ceiling (no attack) — dedicated script:
MODEL=Qwen3.6-35B-A3B bash scripts/run_clean.sh          # no defense
MODEL=Qwen3.6-35B-A3B bash scripts/run_clean.sh camel    # defense on benign tasks (over-defense)
```

Knobs (env vars): `MODEL`, `ATTACK_TYPE`, `ATTACKER_TOOLS`, `TASK_NUM`,
`WORKFLOW_MODE` (`react`/`automatic`/`manual`), `REACT_MAX_TURNS`,
`OPI_INJECT_LIMIT` (default `1`; `0` = every observation), `FORCE_RERUN=1`,
`MAX_WORKERS` (concurrent tasks, default `16` — high values exhaust file
descriptors), `MAX_NEW_TOKENS`, `LOCAL_DISABLE_THINKING`, `LOG_DIR`, and the
`ASB_JUDGE_*` overrides. Per-task `conda list` reqs checks are skipped by default
(`ASB_SKIP_REQS=1`); set `ASB_SKIP_REQS=""` to restore them.

A full 1020-case sweep is just the baseline + defense runs on the full attacker
tools (`ATTACKER_TOOLS=data/all_attack_tools.jsonl`, larger `TASK_NUM`). Resume is
on by default, so an interrupted run (OOM / network / Ctrl-C) is continued by
re-running the same command; `FORCE_RERUN=1` recomputes from scratch.

## Metrics & analysis

Each task writes a JSON trace in AgentDojo `TaskResults` shape, nested like AgentDojo under a
per-run top directory (`run_opi.sh`/`run_clean.sh` bake it into `--log_dir`):
`logs/<model>_nodefense | <model>+<defense>/<agent>/<user_task>/<attack|none>/<injection|none>.json`
— e.g. `logs/Qwen3.6-35B-A3B+melon/financial_analyst_agent/<task>/context_ignoring/<attacker_tool>.json`
for a MELON run; a clean (no-attack) run lands at `.../<user_task>/none/none.json`. The record carries
`security` = attack success (**ASR**), `utility` = benign-task success (**BP**, measured under attack),
`clean`, plus `tool_trace` and `duration`.

Aggregate and diff:

```bash
python scripts/analyze.py                                  # per-label ASR/util/refuse (+ per attack_type)
python scripts/analyze.py --defense melon                  # baseline vs MELON, per-case
python scripts/analyze.py --defense camel --dump-diff d.csv
```

The diff reports how many attacks the defense **neutralised** (baseline hit →
defense miss), how many still got through, whether the defense **introduced** any
new attack, and how many benign tasks it broke (**over-defense**).

## Notes / caveats

- **BP is utility _under attack_**, not clean utility. For the clean-utility
  ceiling, run with `CLEAN=1` (no injection, no attacker tool).
- **MELON's threshold** defaults to 0.8 (as in the paper); ASB's tool surface
  differs from AgentDojo's, so sweep it if needed (`MELON.__init__(threshold=…)`).
- **CaMeL** imports `src.camel` from the camel-prompt-injection repo via
  `PYTHONPATH` (set by `run_opi.sh`); it is not functional standalone in this repo.
- One pre-existing ASB tool (`pyopenagi/tools/meteosource_weather/find_place.py`)
  uses a Python-3.12 f-string; needs Python ≥3.12 to import.
