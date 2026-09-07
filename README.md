# [ICML'25] MELON: Provable Defense Against Indirect Prompt Injection Attacks in AI Agents

This repo bundles [AgentDojo](https://github.com/ethz-spylab/agentdojo) under `agentdojo/` **with MELON already integrated** (the `melon` defense is registered and `pi_detector.py` is in place) — no manual patching needed.

It also bundles the three dynamic task suites from [AgentDyn](https://github.com/SaFo-Lab/AgentDyn) (`shopping`, `github`, `dailylife` — 60 open-ended user tasks and their injection test cases), so the benchmark runs on all seven suites with the same arguments.

## Install

Python 3.11+ recommended.

```bash
git clone https://github.com/lindsey98/melon && cd melon
conda create -n melon python=3.11 -y && conda activate melon
# Installs AgentDojo (incl. the AgentDyn suites) + MELON detection deps
# (numpy, sentence-transformers) + transformers extra
cd agentdojo 
pip install -e ".[transformers,melon]"
cd ../
pip install "vllm>=0.6.3"                # optional: serve local agent LLMs
```

The AgentDyn suites are part of the bundled `agentdojo` package — no separate AgentDyn install is needed (do **not** `pip install` AgentDyn's own fork on top; it would replace the MELON-integrated `agentdojo`).

The `melon` extra pulls in `sentence-transformers`, so the fully-local embedding backend works out of the box.

## Configure API keys

Copy `.env.example` to `.env`, fill in the keys you need (e.g. `OPENAI_API_KEY`), then load it:

```bash
cp .env.example .env      # then edit .env
set -a && source .env && set +a
```

All supported variables are documented inline in [`.env.example`](.env.example). `.env` is git-ignored.

## Run

Use `python main.py MODEL ...`. Output always goes to `logs/`; add `--defense melon` for the MELON defense. Available suites (`--suites`): AgentDojo's `workspace`, `slack`, `banking`, `travel` and AgentDyn's `shopping`, `github`, `dailylife`. Omit `--suites` to run all seven.

**Attack run:**

```bash
python main.py gpt-4o-2024-05-13 --run-attack \
    --attack important_instructions \
    --suites banking slack travel workspace \
    --defense melon
```

**No-attack (utility) run** — just drop `--run-attack`:

```bash
python main.py gpt-4o-2024-05-13 --suites banking slack travel workspace --defense melon
```

Optional flags: `--force_rerun` (ignore cached results), `--html` (write a rendered HTML trace next to each per-task JSON), `-ut`/`-it` (restrict to specific user/injection tasks, repeatable). Run `python main.py --help` for the full list. Omit `--defense` (or pass `--defense None`) for the undefended baseline. Per-task JSON traces are written under `logs/<model[+defense]>/<suite>/<user_task>/<attack|none>/<injection|none>.json`.

The underlying click CLI (`python -m agentdojo.scripts.benchmark`) still works and shares the same engine.

**ChatInject attack** ([ChatInject](https://github.com/hwanchang00/ChatInject)): reconstructs the target model's chat template inside an injected tool output so an attacker-authored system/user/assistant exchange is parsed as real conversation history. Pick the variant matching your served model's template:

- `chat_inject_qwen3`, `chat_inject_glm` — single fake turn; works on any suite.
- `chat_inject_{qwen3,glm}_with_utility_system_multiturn_7` and the `..._authority_endorsement_...` variants — prepend a pre-generated multi-turn dialogue. Data lives in `agentdojo/src/agentdojo/attacks/chatinject_data/` and **only covers the `banking` / `slack` / `travel` injection GOALs** (matched by exact GOAL string); other suites raise a `ValueError`.

```bash
python main.py Qwen3-30B-A3B-Instruct-2507 --run-attack \
    --attack chat_inject_qwen3_with_utility_system_multiturn_7 --defense melon --suites banking
```

**Locally-served model (Qwen3-30B-A3B-Instruct-2507 / Llama-3.3-70B-Instruct):**

`Qwen3-30B-A3B-Instruct-2507` and `Llama-3.3-70B-Instruct` are registered in `models.py` and use **native tool calling** (the `vllm_parsed` provider → `OpenAILLM`), so vLLM **must** be served with `--enable-auto-tool-choice --tool-call-parser` — otherwise tool calls arrive as plain text and the agent can't act. Serve the model under its registered name, set `LOCAL_LLM_PORT` in `.env`, then pass the name as the `MODEL` argument:

```bash
# Qwen3 → --tool-call-parser hermes  |  Llama-3.3 → --tool-call-parser llama3_json
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 --port 8000 \
    --served-model-name Qwen3-30B-A3B-Instruct-2507 \
    --enable-auto-tool-choice --tool-call-parser hermes

python main.py Qwen3-30B-A3B-Instruct-2507 --run-attack --attack tool_knowledge --defense melon
```

For a fully-local run (no hosted API at all), set `MELON_EMBED_PROVIDER=sentence-transformers` in `.env` so MELON's detection embeddings run locally too. Optionally set `HF_HOME` and pre-download the embedding model so the first run doesn't block:

```bash
python scripts/prefetch_embed_model.py   # downloads MELON_EMBED_MODEL into $HF_HOME
```

## Use MELON in your own project

```python
from agentdojo.agent_pipeline.pi_detector import MELON

detector = MELON(
    llm,                                       # your AgentDojo LLM pipeline element
    threshold=0.8,                             # cosine-similarity threshold
    embed_provider="sentence-transformers",    # or "openai" / "openai-compatible"
)
```

## Contact

Questions? Contact [Kaijie Zhu](https://kaijiezhu11.github.io/).

## Citation

```bibtex
@inproceedings{zhu2025melon,
    title={MELON: Provable Defense Against Indirect Prompt Injection Attacks in AI Agents},
    author={Zhu, Kaijie and Yang, Xianjun and Wang, Jindong and Guo, Wenbo and Wang, William Yang},
    year={2025},
    booktitle={International Conference on Machine Learning},
}
```
