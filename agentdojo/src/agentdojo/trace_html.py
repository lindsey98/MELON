"""Render a per-task benchmark trace (the same dict written to ``<task>.json``) as a
standalone HTML page, so every JSON trace gets a human-readable ``<task>.html`` next
to it.

The layout mirrors the case-study trace figures: role-colored message cards
(system / user / assistant / tool), assistant tool calls and tool results, and — when
the MELON defense recorded detections — purple MELON cards with the original-vs-masked
comparison and per-candidate cosine-similarity bars.

Kept dependency-free and defensive: it renders whatever fields are present and never
raises on missing/oddly-shaped data, so it can be called right after the JSON dump
without risking the benchmark run.
"""

from __future__ import annotations

import html
import json
from typing import Any

_CSS = """
:root { --sys:#64748b; --usr:#2563eb; --ast:#16a34a; --tool:#d97706; --mel:#9333ea; }
* { box-sizing:border-box; margin:0; padding:0; }
body { font-family:"Helvetica Neue", Arial, sans-serif; background:#fff; display:flex; justify-content:center; padding:24px; }
.figure { width:940px; max-width:100%; }
.title { font-size:15px; font-weight:700; color:#111; margin-bottom:4px; }
.title span { font-weight:400; color:#666; font-size:13px; }
.subnote { font-size:11.5px; color:#94a3b8; margin-bottom:12px; }
.msg { border:1.5px solid; border-radius:8px; padding:10px 14px; margin-bottom:10px; font-size:12.5px; line-height:1.45; color:#1f2937; position:relative; }
.role { display:inline-block; font-size:10.5px; font-weight:700; letter-spacing:0.06em; text-transform:uppercase; color:#fff; border-radius:4px; padding:2px 8px; margin-bottom:6px; }
.sys { border-color:var(--sys); background:#f8fafc; } .sys .role { background:var(--sys); }
.usr { border-color:var(--usr); background:#eff6ff; } .usr .role { background:var(--usr); }
.ast { border-color:var(--ast); background:#f0fdf4; } .ast .role { background:var(--ast); }
.tool { border-color:var(--tool); background:#fffbeb; margin-left:32px; } .tool .role { background:var(--tool); }
.melcard { border-color:var(--mel); background:#faf5ff; margin-left:32px; } .melcard .role { background:var(--mel); }
code, .mono { font-family:"SF Mono", Menlo, Consolas, monospace; font-size:11.5px; }
.call { background:#dcfce7; border:1px solid #86efac; border-radius:6px; padding:6px 10px; margin-top:6px; display:block; white-space:pre-wrap; word-break:break-word; }
.result { background:#fef3c7; border:1px solid #fcd34d; border-radius:6px; padding:6px 10px; margin-top:4px; display:block; white-space:pre-wrap; word-break:break-word; }
.text { white-space:pre-wrap; word-break:break-word; }
.nocontent { color:#6b7280; font-style:italic; font-size:11.5px; }
.melbox { background:#f3e8ff; border:1px solid #d8b4fe; border-radius:6px; padding:6px 10px; margin-top:4px; display:block; white-space:pre-wrap; color:#581c87; }
.hl { background:#fde68a; font-weight:700; }
.box { margin-top:8px; border-top:1px dashed #d8b4fe; padding-top:8px; }
.box-label { font-size:10.5px; font-weight:700; color:var(--mel); letter-spacing:0.06em; margin-bottom:4px; }
.cmp { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:6px; }
.cmp .pane { border:1px solid #d8b4fe; border-radius:6px; padding:6px 9px; background:#faf5ff; font-size:11px; color:#4c1d95; }
.cmp .pane .lbl { font-weight:700; font-size:10px; letter-spacing:0.05em; text-transform:uppercase; color:var(--mel); margin-bottom:3px; }
.cmp .pane code { display:block; margin-top:4px; color:#6b21a8; }
.simrow { display:flex; align-items:center; gap:8px; margin:2px 0; font-size:11px; }
.bar { height:10px; background:#d8b4fe; border-radius:2px; } .bar.hit { background:#9333ea; }
.simrow code { color:#4c1d95; min-width:280px; } .simrow .val { font-weight:700; color:#581c87; min-width:52px; text-align:right; }
.under { color:#15803d; font-weight:700; } .over { background:#fecaca; font-weight:700; }
.stop-badge { position:absolute; top:10px; right:12px; background:var(--mel); color:#fff; font-size:10.5px; font-weight:700; border-radius:4px; padding:2px 8px; letter-spacing:0.06em; }
.ok { color:#15803d; font-weight:700; } .no { color:#b91c1c; font-weight:700; }
"""


def _esc(x: Any) -> str:
    return html.escape("" if x is None else str(x))


def _text_of(content: Any) -> str:
    """Extract plain text from a message ``content`` (str, None, or list of blocks)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("content", block.get("text", ""))))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _fmt_tool_call(tc: Any) -> str:
    """Render a tool call (FunctionCall dict) as ``name(k=v, ...)``."""
    if not isinstance(tc, dict):
        return str(tc)
    name = tc.get("function", tc.get("name", "?"))
    args = tc.get("args", tc.get("parameters")) or {}
    if isinstance(args, dict):
        inner = ", ".join(f"{k}={json.dumps(v, default=str)}" for k, v in args.items())
    else:
        inner = str(args)
    return f"{name}({inner})"


def _flag(value: Any) -> str:
    if value is True:
        return '<span class="ok">&#10003;</span>'
    if value is False:
        return '<span class="no">&#10007;</span>'
    return _esc(value)


def _render_message(msg: dict) -> str:
    role = msg.get("role", "?")
    text = _text_of(msg.get("content"))
    if role == "system":
        return (
            '<div class="msg sys"><span class="role">System</span>'
            f'<div class="text">{_esc(text)}</div></div>'
        )
    if role == "user":
        return (
            '<div class="msg usr"><span class="role">User</span>'
            f'<div class="text">{_esc(text)}</div></div>'
        )
    if role == "assistant":
        body = f'<div class="text">{_esc(text)}</div>' if text else '<div class="nocontent">(no text)</div>'
        for tc in msg.get("tool_calls") or []:
            body += f'<code class="call">&rarr; {_esc(_fmt_tool_call(tc))}</code>'
        return f'<div class="msg ast"><span class="role">Assistant</span>{body}</div>'
    if role == "tool":
        tc = msg.get("tool_call")
        fname = _fmt_tool_call(tc) if tc else (msg.get("tool_call_id") or "tool")
        err = msg.get("error")
        result = f"error: {err}" if err else text
        return (
            '<div class="msg tool"><span class="role">Tool</span>'
            f'<code class="result">{_esc(fname)} &rarr; {_esc(result) or "&lt;empty&gt;"}</code></div>'
        )
    return f'<div class="msg sys"><span class="role">{_esc(role)}</span><div class="text">{_esc(text)}</div></div>'


def _render_melon(detections: list) -> str:
    if not detections:
        return ""
    cards = ['<div class="box-label" style="margin-top:16px;font-size:12px">MELON DETECTION</div>']
    for i, d in enumerate(detections):
        if not isinstance(d, dict):
            continue
        is_inj = d.get("is_injection")
        cos = d.get("max_cosine_similarity")
        thr = d.get("threshold")
        cos_s = f"{cos:.3f}" if isinstance(cos, (int, float)) else _esc(cos)
        thr_s = f"{thr:.2f}" if isinstance(thr, (int, float)) else _esc(thr)
        verdict = (
            f'<span class="hl">true</span>   max cosine <span class="over">{cos_s}</span> &ge; {thr_s}'
            if is_inj
            else f'<span class="under">false</span>   max cosine {cos_s} &lt; {thr_s}'
        )
        badge = '<span class="stop-badge">FLAGGED</span>' if is_inj else ""
        orig = d.get("original_response") or {}
        masked = d.get("masked_response") or {}
        orig_calls = ", ".join(orig.get("tool_calls") or []) or "&mdash;"
        masked_calls = ", ".join(masked.get("tool_calls") or []) or "&mdash;"
        panes = (
            '<div class="cmp">'
            f'<div class="pane"><div class="lbl">Original run</div><code>{_esc(orig_calls)}</code></div>'
            f'<div class="pane"><div class="lbl">Masked run (task hidden)</div><code>{_esc(masked_calls)}</code></div>'
            "</div>"
        )
        sims = d.get("similarities") or []
        sim_rows = ""
        if sims:
            top = sorted(sims, key=lambda s: s.get("cosine_similarity", 0), reverse=True)[:6]
            max_sim = max((s.get("cosine_similarity", 0) for s in top), default=1) or 1
            rows = ""
            for s in top:
                cs = s.get("cosine_similarity", 0)
                width = int(40 + 160 * (cs / max_sim)) if max_sim else 40
                hit = " hit" if isinstance(thr, (int, float)) and cs > thr else ""
                val_cls = "over" if isinstance(thr, (int, float)) and cs > thr else ""
                pair = f'{s.get("original_tool_call", "?")}  vs  {s.get("masked_tool_call", "?")}'
                rows += (
                    f'<div class="simrow"><code>{_esc(pair)}</code>'
                    f'<div class="bar{hit}" style="width:{width}px"></div>'
                    f'<span class="val {val_cls}">{cs:.3f}</span></div>'
                )
            sim_rows = f'<div class="box"><div class="box-label">ORIGINAL vs MASKED CANDIDATES</div>{rows}</div>'
        cards.append(
            f'<div class="msg melcard">{badge}<span class="role">MELON &middot; step {i + 1}</span>'
            f'<code class="melbox">is_injection: {verdict}</code>{panes}{sim_rows}</div>'
        )
    return "".join(cards)


def render_trace_html(record: dict) -> str:
    """Render a task-result ``record`` (the JSON trace dict) as a standalone HTML page."""
    suite = record.get("suite_name", "?")
    user_task = record.get("user_task_id", "?")
    injection_task = record.get("injection_task_id") or "none"
    attack = record.get("attack_type") or "none"
    pipeline = record.get("pipeline_name", "?")
    utility = record.get("utility")
    security = record.get("security")
    duration = record.get("duration")
    error = record.get("error")

    meta_bits = [
        f"suite: {_esc(suite)}",
        _esc(user_task),
        f"injection: {_esc(injection_task)}",
        f"attack: {_esc(attack)}",
        _esc(pipeline),
    ]
    status = []
    if utility is not None:
        status.append(f"utility {_flag(utility)}")
    if security is not None:
        status.append(f"security {_flag(security)}")
    subnote = []
    if isinstance(duration, (int, float)):
        subnote.append(f"{duration:.2f}s")
    if error:
        subnote.append(f"error: {_esc(error)}")

    messages = record.get("messages") or []
    body = "".join(_render_message(m) for m in messages if isinstance(m, dict))
    melon = _render_melon(record.get("melon_detection") or [])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trace &mdash; {_esc(suite)} / {_esc(user_task)} / {_esc(injection_task)}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="figure">
  <div class="title">Execution Trace <span>&mdash; {" &middot; ".join(meta_bits)}{("  &middot;  " + " &middot; ".join(status)) if status else ""}</span></div>
  <div class="subnote">{" &middot; ".join(subnote) if subnote else "&nbsp;"}</div>
  {body}
  {melon}
</div>
</body>
</html>
"""
