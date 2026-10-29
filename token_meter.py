"""Per-task token accounting, written into every record -- so in/out tokens are counted the AgentPI way:
ALL calls, including every internal defense node, not just the agent's turns.

How: a single `install()` monkeypatches
  1. the SYNC OpenAI client (`openai.resources.chat.completions.Completions.create`) to accumulate
     `response.usage` -- this catches the agent AND every defense-side call that hits an OpenAI-compatible
     endpoint (local vLLM qualifies), regardless of whether the repo's own code remembered to tally it; and
  2. AgentDojo's `TraceLogger`: reset the counter when a task's logger is entered, and write the totals
     (`prompt_tokens`/`completion_tokens`/`total_tokens`) via `set_contextarg` on exit, so they land in the
     record JSON next to `duration`.

Scope/limits: OpenAI-compatible SYNC clients only (what these agentdojo pipelines use for local models); a
node using a non-OpenAI or async client would need its own hook. Global counters -> runs must be SEQUENTIAL
(the benchmark and the /metrics cross-check already require this); not thread-safe. Idempotent.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_prompt = 0
_completion = 0
_installed = False


def reset() -> None:
    global _prompt, _completion
    _prompt = 0
    _completion = 0


def add(prompt, completion) -> None:
    global _prompt, _completion
    try:
        _prompt += int(prompt or 0)
        _completion += int(completion or 0)
    except (TypeError, ValueError):
        pass


def totals() -> tuple[int, int]:
    return _prompt, _completion


def install() -> None:
    global _installed
    if _installed:
        return
    _installed = True

    # 1) every sync openai chat completion (agent + defense nodes)
    try:
        from openai.resources.chat import completions as _c

        _orig_create = _c.Completions.create

        def _create(self, *args, **kwargs):
            resp = _orig_create(self, *args, **kwargs)
            try:
                u = getattr(resp, "usage", None)
                if u is not None:
                    add(getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0))
            except Exception:  # noqa: BLE001 -- accounting must never break a generation
                pass
            return resp

        _c.Completions.create = _create
    except Exception as e:  # noqa: BLE001
        log.warning("token_meter: could not patch openai client (%s)", e)

    # 2) per-task boundary + write totals into the record
    try:
        from agentdojo.logging import TraceLogger

        _orig_enter = TraceLogger.__enter__
        _orig_exit = TraceLogger.__exit__

        def _enter(self):
            reset()
            return _orig_enter(self)

        def _exit(self, exc_type, exc_value, traceback):
            p, c = totals()
            self.set_contextarg("prompt_tokens", p)
            self.set_contextarg("completion_tokens", c)
            self.set_contextarg("total_tokens", p + c)
            return _orig_exit(self, exc_type, exc_value, traceback)

        TraceLogger.__enter__ = _enter
        TraceLogger.__exit__ = _exit
    except Exception as e:  # noqa: BLE001
        log.warning("token_meter: could not patch agentdojo TraceLogger (%s)", e)

    log.info("token_meter installed (sync-openai + TraceLogger)")
