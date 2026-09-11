"""Render a chat template under controlled conditions instead of pattern-matching it.

Guessing a template's behaviour from regexes breaks the moment a vendor uses a
different reasoning convention (Gemma-4 opens a `<|channel>thought`, ByteDance
uses `<seed:think>`). Rendering is exact: build the same conversation three
times, with the reasoning flag unset, true and false, and compare what each one
appends after the assistant header.
"""
import json, re
from jinja2 import Environment, BaseLoader, nodes
from jinja2.exceptions import TemplateError
from jinja2.ext import Extension
from jinja2.sandbox import ImmutableSandboxedEnvironment


class GenerationExtension(Extension):
    """transformers templates use `{% generation %}...{% endgeneration %}` to mark
    assistant spans for token masking. Jinja does not know the tag, so templates
    using it fail to parse. Here it just emits its body unchanged."""
    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return nodes.CallBlock(self.call_method("_noop", []), [], [], body).set_lineno(lineno)

    def _noop(self, caller):
        return caller()

MSGS = [{"role": "user", "content": "What is 84 * 3 / 2?"}]

# Reasoning-delimiter conventions seen in the wild. Each entry is
# (open_pattern, close_pattern, label).
CONVENTIONS = [
    (r"<think>",            r"</think>",            "think"),
    (r"<seed:think>",       r"</seed:think>",       "seed:think"),
    (r"<reasoning>",        r"</reasoning>",        "reasoning"),
    (r"<\|channel\|>analysis", r"<\|end\|>",        "harmony-channel"),
    (r"<\|channel>thought", r"<\|/channel>|<\|turn>", "gemma-channel"),
    (r"<thinking>",         r"</thinking>",         "thinking"),
]


def _env():
    env = ImmutableSandboxedEnvironment(loader=BaseLoader(), trim_blocks=True,
                                        lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols", GenerationExtension])
    def raise_exception(msg):
        raise TemplateError(msg)
    def tojson(x, ensure_ascii=True, **kw):
        return json.dumps(x, ensure_ascii=ensure_ascii)
    def strftime_now(fmt):
        import datetime
        return datetime.datetime.now().strftime(fmt)
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now
    env.filters["tojson"] = tojson
    return env


def render(tpl_src, **extra):
    """Render with add_generation_prompt. Returns text or raises."""
    tpl = _env().from_string(tpl_src)
    return tpl.render(messages=MSGS, add_generation_prompt=True,
                      bos_token="", eos_token="", **extra)


def tail_after_last_turn(text):
    """The bytes a model would be asked to continue from: everything after the
    final role header the template emitted."""
    marks = [m.end() for m in re.finditer(
        r"<\|im_start\|>assistant|<\|start\|>assistant|<\|assistant\|>|"
        r"<\|turn>model|\[/INST\]|<start_of_turn>model|### Response:|"
        r"<\|start_header_id\|>assistant<\|end_header_id\|>", text)]
    return text[marks[-1]:] if marks else text[-260:]


def prefill_state(tail):
    """OPEN  - an unclosed reasoning delimiter is sitting in the prefill
       CLOSED - a complete empty reasoning block was emitted
       NONE  - no reasoning delimiter at all; the model decides"""
    for op, cl, label in CONVENTIONS:
        opens = list(re.finditer(op, tail))
        if not opens:
            continue
        closes = list(re.finditer(cl, tail))
        if closes and closes[-1].start() > opens[-1].start():
            return "CLOSED", label
        return "OPEN", label
    return "NONE", None


def audit(tpl_src):
    """Render under unset / true / false and report the default plus whether the
    flag actually does anything."""
    out = {}
    for name, kw in (("unset", {}), ("true", {"enable_thinking": True}),
                     ("false", {"enable_thinking": False})):
        try:
            txt = render(tpl_src, **kw)
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {e}"[:160]}
            continue
        tail = tail_after_last_turn(txt)
        state, conv = prefill_state(tail)
        out[name] = {"state": state, "convention": conv, "tail": tail[-90:]}

    states = {k: v.get("state") for k, v in out.items()}
    default = states.get("unset")
    honors = states.get("true") != states.get("false")
    # The eval-fairness question: does an unset flag match what the flag=true
    # path would have given? If not, a default harness silently handicaps it.
    matches_on = default == states.get("true")
    return {"default": default, "honors_flag": honors,
            "unset_matches_thinking_on": matches_on,
            "states": states, "detail": out}
