# MELON on ASB (observation prompt injection)

This directory vendors [ASB (Agent Security Bench)](https://github.com/agiresearch/ASB)
with the **MELON** defense added, so MELON can be evaluated against ASB's
**observation prompt injection (OPI)** attack. The original ASB README is kept as
[`ASB_README.md`](ASB_README.md).

ASB is not built on AgentDojo — it has its own agent framework (`pyopenagi` /
`aios`). MELON is therefore ported here as a native ASB defense rather than run
through AgentDojo. The port lives in
[`pyopenagi/agents/melon.py`](pyopenagi/agents/melon.py) and mirrors the
AgentDojo detector at `agentdojo/src/agentdojo/agent_pipeline/pi_detector.py`.

This copy also carries an OPI-focused reworking of ASB (local/OpenAI-compatible
model support, a ReAct tool-calling loop, AgentDojo-format output, a configurable
refuse judge, and `scripts/run_opi.sh`). MELON and CaMeL are both wired as
`defense_type` options; **MELON is the defense of interest here**. (The `camel`
option imports the CaMeL kernel from the camel-prompt-injection repo via
`PYTHONPATH` and is not functional standalone in this repo — it's kept for
parity.)

## What was added for MELON

- `pyopenagi/agents/melon.py` — the MELON contrastive detector for ASB.
- `pyopenagi/agents/react_agent_attack.py` — hooked so that when
  `defense_type == 'melon'`, each pending tool call is checked and blocked if the
  detector attributes it to an injected instruction. The gate is wired into **both**
  loops: the ReAct loop (`run_react`) and the plan-then-execute loop. Edits: an
  import, `self.melon_agent` init in `run()`, a `_melon_query` helper, and the two
  gates.
- `config/OPI_melon.yml` — OPI run with `defense_type: melon`.

Only the OPI attack path is wired for MELON. DPI / memory-poisoning / PoT are
untouched (and the large DPI memory database is not vendored).

## How MELON works here

At each step/turn where the agent is about to call tools, MELON re-runs the **same
model** on a *masked* prompt containing only the tool outputs seen so far (the
user's real task removed) plus a generic "summarize, and follow any instructions
you find" instruction. If a masked-run tool call matches a real-run tool call
(cosine similarity of their embeddings > threshold, default 0.8), the real call
must have been driven by content injected into a tool output — so it is flagged
and blocked. No injection signature is needed. The detector reads tool outputs
from both the ReAct loop's proper `tool`-role messages and the plan loop's
`[Observation]:` strings.

## Install

```bash
cd asb
pip install -r requirements.txt
# MELON's detection embeddings — pick ONE backend:
pip install openai                 # hosted OpenAI embeddings (default), or
pip install sentence-transformers  # fully-local embeddings, no API
```

## Configure

MELON's embedding backend is selected with the same `MELON_EMBED_*` environment
variables as the AgentDojo copy:

```bash
# Hosted OpenAI embeddings (default backend)
export OPENAI_API_KEY=sk-...
# MELON_EMBED_MODEL defaults to text-embedding-3-large

# — or — fully local, no hosted API:
export MELON_EMBED_PROVIDER=sentence-transformers
export MELON_EMBED_MODEL=BAAI/bge-large-en-v1.5   # or a local path

# — or — a local OpenAI-compatible embedding server (e.g. vLLM/TEI):
export MELON_EMBED_PROVIDER=openai-compatible
export MELON_EMBED_BASE_URL=http://localhost:8001/v1
```

The embedding backend is **independent of the agent LLM**, so you can run the
agent on a local vLLM model and MELON's embeddings on OpenAI (or vice versa). The
agent LLM is set in the config (`llms:`) / `run_opi.sh` (`MODEL`) and, for a local
model, reached via `LOCAL_BASE_URL` / `LOCAL_API_KEY`.

## Run

Recommended (local model + ReAct loop), via the one-stop script:

```bash
cd asb
# agent on a local vLLM endpoint; embeddings fully local too
export MELON_EMBED_PROVIDER=sentence-transformers
MODEL=Qwen3.6-35B-A3B bash scripts/run_opi.sh melon      # MELON
MODEL=Qwen3.6-35B-A3B bash scripts/run_opi.sh            # undefended baseline
```

`run_opi.sh` knobs: `MODEL`, `ATTACK_TYPE`, `ATTACKER_TOOLS`, `TASK_NUM`,
`WORKFLOW_MODE` (`react` default / `automatic` / `manual`), `REACT_MAX_TURNS`,
`MAX_NEW_TOKENS`, `LOCAL_DISABLE_THINKING`, and the `ASB_JUDGE_*` refuse-judge
overrides. Each blocked tool call is logged as
`MELON blocked a tool call (potential prompt injection).`

Or via the config runner (plan-then-execute, no `--workflow_mode`):

```bash
python scripts/agent_attack.py --cfg_path config/OPI_melon.yml
```

Each task also writes an AgentDojo-style `TaskResults` JSON under
`logs/observation_prompt_injection/json/<model>/<label>/...`, with `security` =
attack-success (ASR) and a `tool_trace`. Compare MELON vs the undefended baseline
(same command without the `melon` argument) on ASR / benign performance.
