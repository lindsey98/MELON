# [ICML'25] MELON: Provable Defense Against Indirect Prompt Injection Attacks in AI Agents

This is the official implementation of [MELON: Provable Defense Against Indirect Prompt Injection Attacks in AI Agents](https://arxiv.org/abs/2502.05174).

## Abstract

Recent research has explored that LLM agents are vulnerable to indirect prompt injection (IPI) attacks, where malicious tasks embedded in tool-retrieved information can redirect the agent to take unauthorized actions. Existing defenses against IPI have significant limitations: either require essential model training resources, lack effectiveness against sophisticated attacks, or harm the normal utilities. We present MELON (Masked re-Execution and TooL comparisON), a novel IPI defense. Our approach builds on the observation that under a successful attack, the agent's next action becomes less dependent on user tasks and more on malicious tasks. Following this, we design MELON to detect attacks by re-executing the agent's trajectory with a masked user prompt modified through a masking function. We identify an attack if the actions generated in the original and masked executions are similar. We also include three key designs to reduce the potential false positives and false negatives. Extensive evaluation on the IPI benchmark AgentDojo demonstrates that MELON outperforms SOTA defenses in both attack prevention and utility preservation. Moreover, we show that combining MELON with a SOTA prompt augmentation defense (denoted as MELON-Aug) further improves its performance. We also conduct a detailed ablation study to validate our key designs.

---

## Table of Contents

- [Repository contents](#repository-contents)
- [1. Environment setup](#1-environment-setup)
- [2. Environment variables](#2-environment-variables)
- [3. Running with hosted models (OpenAI / Anthropic / …)](#3-running-with-hosted-models)
- [4. Running with locally-served models (Qwen3 & Llama-3.3)](#4-running-with-locally-served-models)
- [5. Using MELON in your own project](#5-using-melon-in-your-own-project)
- [How MELON is integrated (reference)](#how-melon-is-integrated-reference)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)

---

## Repository contents

| Path | Description |
| --- | --- |
| `agentdojo/` | A checkout of [AgentDojo](https://github.com/ethz-spylab/agentdojo) **with MELON already integrated** — the `melon` defense is registered and `pi_detector.py` is in place. Install it with `pip install -e .`. |
| `pi_detector.py` | Stand-alone copy of the MELON detector (`class MELON`) plus the base `PromptInjectionDetector`, for reuse in your own projects. Identical to `agentdojo/src/agentdojo/agent_pipeline/pi_detector.py`. |
| `README.md` | This file. |

MELON is implemented as an AgentDojo *pipeline element*. **No manual patching is
required** — the bundled `agentdojo/` folder already registers the `melon`
defense and ships the detector, so you only need to install it.

---

## 1. Environment setup

MELON runs on top of [AgentDojo](https://github.com/ethz-spylab/agentdojo), which
is bundled (pre-integrated) in this repository under `agentdojo/`. We recommend
Python **3.11+** and an isolated environment.

```bash
# 1) Create and activate an environment (conda or venv both work)
conda create -n melon python=3.11 -y
conda activate melon
# --- or ---
# python3.11 -m venv .venv && source .venv/bin/activate

# 2) Clone this repository (it already contains the MELON-patched AgentDojo)
git clone https://github.com/lindsey98/melon
cd melon

# 3) Install the bundled, pre-integrated AgentDojo
#    (the transformers extra is used by some detectors)
cd agentdojo
pip install -e ".[transformers]"
cd ..
```

That's it — the `melon` defense is now available to
`python -m agentdojo.scripts.benchmark` (see [Section 4](#4-running-with-hosted-models)
and [Section 5](#5-running-with-locally-served-models)).

### Optional extras (depending on how you run MELON)

```bash
# Required for MELON's embedding-based detection backend (OpenAI client is also
# used to talk to local OpenAI-compatible servers such as vLLM):
pip install openai numpy

# Only if you want a FULLY-LOCAL detection backend (no external embedding API):
pip install sentence-transformers

# Only if you want to SERVE the agent LLMs locally (Qwen3 / Llama-3.3):
pip install "vllm>=0.6.3"
```

---

## 2. Environment variables

MELON and AgentDojo are configured entirely through environment variables — no
keys are hard-coded. Set only the variables relevant to the providers you use.

### Agent-LLM provider keys (used by AgentDojo to drive the agent)

| Variable | Used by | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | `--model gpt-4o-*` etc. | OpenAI API key for hosted GPT models. |
| `ANTHROPIC_API_KEY` | `--model claude-*` | Anthropic API key. |
| `TOGETHER_API_KEY` | `--model meta-llama/*` (Together) | Together AI key for hosted open models. |
| `GCP_PROJECT`, `GCP_LOCATION` | `--model gemini-*` | Vertex AI project/location for Gemini. |
| `LOCAL_LLM_PORT` | `--model local` / `--model vllm_parsed` | Port of your local vLLM server (default `8000`). |
| `OPENAI_COMPATIBLE_BASE_URL` | `--model openai-compatible` | Base URL of any OpenAI-compatible server. |
| `OPENAI_COMPATIBLE_API_KEY` | `--model openai-compatible` | API key for that server (use `EMPTY` for vLLM). |

### MELON detection-embedding backend (used by `class MELON`)

MELON compares the original and masked tool calls in embedding space. The
embedding backend is selectable so the whole pipeline can run locally.

| Variable | Default | Description |
| --- | --- | --- |
| `MELON_EMBED_PROVIDER` | `openai` | One of `openai`, `openai-compatible` (aliases `local`/`vllm`), or `sentence-transformers`. |
| `MELON_EMBED_MODEL` | `text-embedding-3-large` (OpenAI) / `BAAI/bge-large-en-v1.5` (sentence-transformers) | Embedding model name. |
| `OPENAI_API_KEY` | – | API key when `MELON_EMBED_PROVIDER=openai`. |
| `OPENAI_BASE_URL` | – | Optional override of the OpenAI endpoint. |
| `MELON_EMBED_BASE_URL` | `http://localhost:8001/v1` | Endpoint when `MELON_EMBED_PROVIDER=openai-compatible`. |
| `MELON_EMBED_API_KEY` | `EMPTY` | API key for the OpenAI-compatible embedding endpoint. |

These can also be passed directly to the `MELON(...)` constructor
(`embed_provider`, `embed_model`, `embed_base_url`, `embed_api_key`), which take
precedence over the environment variables.

Example for the default (hosted OpenAI embeddings):

```bash
export OPENAI_API_KEY="sk-..."
```

Example for a **fully local** run (no external API at all):

```bash
export MELON_EMBED_PROVIDER="sentence-transformers"
export MELON_EMBED_MODEL="BAAI/bge-large-en-v1.5"
```

---

## 3. Running with hosted models

Set the relevant provider key (e.g. `OPENAI_API_KEY`) and run AgentDojo's
benchmark, selecting `--defense melon`:

```bash
python -m agentdojo.scripts.benchmark \
    --model gpt-4o-2024-05-13 \
    --attack tool_knowledge \
    --defense melon \
    -s slack \
    > gpt-4o-2024-05-13_tool_knowledge_melon_slack.log
```

---

## 4. Running with locally-served models

MELON supports agent LLMs that are **served locally**, including
**Qwen3-30B-A3B-Instruct-2507** and **Llama-3.3-70B-Instruct**. AgentDojo talks
to them through an OpenAI-compatible server; we use [vLLM](https://github.com/vllm-project/vllm).
Tool calling must be enabled with the model's tool-call parser so the agent can
emit structured tool calls.

### Step 1 — serve the model with vLLM

**Qwen3-30B-A3B-Instruct-2507** (uses the Hermes-style tool parser):

```bash
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

**Llama-3.3-70B-Instruct** (uses the Llama-3 JSON tool parser):

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --port 8000 \
    --tensor-parallel-size 4 \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json
```

> Adjust `--tensor-parallel-size` to the number of GPUs you have. For gated
> models on Hugging Face, run `huggingface-cli login` first.

### Step 2 — point AgentDojo at the local server

```bash
export LOCAL_LLM_PORT=8000
```

`--model vllm_parsed` auto-detects the served model id from the vLLM server and
uses native (parsed) tool calling, so no `--model-id` is needed:

```bash
# Qwen3-30B-A3B-Instruct-2507  (vLLM serving it on $LOCAL_LLM_PORT)
python -m agentdojo.scripts.benchmark \
    --model vllm_parsed \
    --attack tool_knowledge \
    --defense melon \
    -s slack \
    > qwen3-30b-a3b-instruct-2507_tool_knowledge_melon_slack.log
```

```bash
# Llama-3.3-70B-Instruct  (vLLM serving it on $LOCAL_LLM_PORT)
python -m agentdojo.scripts.benchmark \
    --model vllm_parsed \
    --attack tool_knowledge \
    --defense melon \
    -s slack \
    > llama-3.3-70b-instruct_tool_knowledge_melon_slack.log
```

Alternatively, use the explicit `openai-compatible` provider (any host/port, with
`--model-id` naming the served model):

```bash
export OPENAI_COMPATIBLE_BASE_URL="http://localhost:8000/v1"
export OPENAI_COMPATIBLE_API_KEY="EMPTY"

python -m agentdojo.scripts.benchmark \
    --model openai-compatible \
    --model-id Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --attack tool_knowledge \
    --defense melon \
    -s slack
```

### Step 3 (optional) — keep MELON's detection embeddings local too

So that *no* component calls a hosted API, run MELON's detection embeddings with
`sentence-transformers`:

```bash
export MELON_EMBED_PROVIDER="sentence-transformers"
export MELON_EMBED_MODEL="BAAI/bge-large-en-v1.5"
```

Or serve an embedding model with another vLLM/TEI instance and point MELON at it:

```bash
export MELON_EMBED_PROVIDER="openai-compatible"
export MELON_EMBED_BASE_URL="http://localhost:8001/v1"
export MELON_EMBED_API_KEY="EMPTY"
export MELON_EMBED_MODEL="BAAI/bge-large-en-v1.5"
```

With this configured, the full attack/defense loop (agent LLM **and** detection)
runs entirely on your own hardware.

---

## 5. Using MELON in your own project

`pi_detector.py` is self-contained apart from its AgentDojo imports. The `MELON`
class can be embedded in any AgentDojo-style pipeline:

```python
from agentdojo.agent_pipeline.pi_detector import MELON

detector = MELON(
    llm,                       # your AgentDojo LLM pipeline element
    threshold=0.8,             # cosine-similarity threshold for flagging an injection
    embed_provider="sentence-transformers",   # or "openai" / "openai-compatible"
    embed_model="BAAI/bge-large-en-v1.5",
)
```

---

## How MELON is integrated (reference)

You do **not** need to do any of this — the bundled `agentdojo/` already contains
these changes. This section documents *what* was changed, for transparency and in
case you want to port MELON onto a different AgentDojo checkout.

In `agentdojo/src/agentdojo/agent_pipeline/agent_pipeline.py`:

1. `MELON` is imported alongside the other detectors:
   ```python
   from agentdojo.agent_pipeline.pi_detector import MELON, TransformersBasedPIDetector
   ```
2. `"melon"` is added to the `DEFENSES` list.
3. A `melon` branch is added to `from_config` (mirroring `tool_filter`):
   ```python
   if config.defense == "melon":
       tools_loop = ToolsExecutionLoop(
           [
               ToolsExecutor(tool_output_formatter),
               MELON(llm),
           ]
       )
       pipeline = cls([system_message_component, init_query_component, llm, tools_loop])
       pipeline.name = f"{llm_name}+melon"
       return pipeline
   ```

In `agentdojo/src/agentdojo/agent_pipeline/pi_detector.py`:

1. `class MELON` is added.
2. `class PromptInjectionDetector.query` sets `extra_args["is_injection"]`, used to
   stop the agent once an attack is detected.
3. The detection embedding backend is configurable (OpenAI / local
   OpenAI-compatible / sentence-transformers) via the environment variables in
   [Section 2](#2-environment-variables) — **no API key is hard-coded**.

### Logging layout

Logs follow AgentDojo's convention and are written under `logs/` by default
(override with `--logdir`). The pipeline name is `<model>+melon` when the MELON
defense is enabled and just `<model>` when it is off:

```
logs/
└── Llama-3.3-70B-Instruct+melon/        # <model>+melon  (no suffix when defense is off)
    └── banking/                          # suite
        └── user_task_0/                  # user task
            ├── none/none.json                                # benign run
            └── important_instructions/injection_task_0.json  # under-attack run
```

---

## Contact

If you have any questions, please contact [Kaijie Zhu](https://kaijiezhu11.github.io/).

## Acknowledgments

We thank the authors of [agentdojo](https://github.com/ethz-spylab/agentdojo) for providing the environment for the experiments.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{zhu2025melon,
    title={MELON: Provable Defense Against Indirect Prompt Injection Attacks in AI Agents}, 
    author={Zhu, Kaijie and Yang, Xianjun and Wang, Jindong and Guo, Wenbo and Wang, William Yang},
    year={2025},
    booktitle={International Conference on Machine Learning},
}
```
