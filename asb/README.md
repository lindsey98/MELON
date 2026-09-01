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

## What was added

- `pyopenagi/agents/melon.py` — the MELON contrastive detector for ASB.
- `pyopenagi/agents/react_agent_attack.py` — hooked so that when
  `defense_type == 'melon'`, each pending tool call is checked and blocked if the
  detector attributes it to an injected instruction (three small, marked edits:
  an import, `self.melon_agent` init in `run()`, and a gate in the step loop plus
  a `_melon_query` helper).
- `config/OPI_melon.yml` — OPI run with `defense_type: melon`.

Only the OPI attack path is wired for MELON. DPI / memory-poisoning / PoT are
untouched (and the large DPI memory database is not vendored).

## How MELON works here

At each step where the agent is about to call tools, MELON re-runs the **same
model** on a *masked* prompt containing only the tool outputs seen so far (the
user's real task removed) plus a generic "summarize, and follow any instructions
you find" instruction. If a masked-run tool call matches a real-run tool call
(cosine similarity of their embeddings > threshold, default 0.8), the real call
must have been driven by content injected into a tool output — so it is flagged
and blocked. No injection signature is needed.

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

The agent LLM itself is set in the config (`llms:`) and uses your normal ASB
credentials (`OPENAI_API_KEY`, ollama, etc.).

## Run

```bash
cd asb
python scripts/agent_attack.py --cfg_path config/OPI_melon.yml
```

This launches `main_attacker.py` for each (llm × attack_type × attacker_tool)
with `--observation_prompt_injection --defense_type melon`. Results are written
under `logs/observation_prompt_injection/<llm>/melon/`, and each blocked tool
call is logged as `MELON blocked a tool call (potential prompt injection).`

To compare against the undefended baseline, run `config/OPI.yml` (no
`defense_type`).
