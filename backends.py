"""One generation interface over MLX (Apple Silicon) and vLLM (CUDA).

The eval scripts must not know which machine they are on. They hand over a list
of prompts and get back one record per prompt. MLX walks the list; vLLM runs it
as a batch, which is the whole reason for moving long jobs to a GPU box.

Prompt construction deliberately goes through transformers' AutoTokenizer on
both backends, so the two machines build byte-identical prompts. Letting each
backend template its own way would reintroduce exactly the class of bug this
project exists to find.
"""
import os, platform, time

# Each model needs a different artifact per backend: vLLM wants the original
# repo, MLX wants a converted one.
MODELS = {
    "minicpm5-2b": {"hf": "openbmb/MiniCPM5-2B",
                    "mlx": "openbmb/MiniCPM5-2B",
                    "local": "models/minicpm5-2b"},
    "qwen3.5-2b":  {"hf": "Qwen/Qwen3.5-2B",
                    "mlx": "mlx-community/Qwen3.5-2B-MLX-bf16",
                    "local": "models/qwen3.5-2b"},
}


def pick_backend(name="auto"):
    if name != "auto":
        return name
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mlx"
    return "vllm"


def resolve_path(key, backend):
    """Prefer a local directory if one was fetched, else the hub id."""
    spec = MODELS[key]
    local = spec.get("local")
    if local and os.path.isdir(local) and backend == "mlx":
        return local
    return spec["hf"] if backend == "vllm" else spec["mlx"]


def load_tokenizer(key, backend):
    from transformers import AutoTokenizer
    path = resolve_path(key, backend)
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def build_prompt(tokenizer, user, system=None, thinking=True):
    """Apply the chat template. `thinking` maps to the enable_thinking kwarg that
    both MiniCPM5 and Qwen3.5 templates honour, so both models get the same knob."""
    msgs = ([{"role": "system", "content": system}] if system else [])
    msgs.append({"role": "user", "content": user})
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
        )
    except TypeError:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


class MlxBackend:
    name = "mlx"

    def __init__(self, key, **kw):
        from mlx_lm import load
        self.path = resolve_path(key, "mlx")
        t = time.time()
        self.model, self._tok = load(self.path)
        self.tokenizer = load_tokenizer(key, "mlx")
        print(f"[mlx] loaded {self.path} in {time.time()-t:.1f}s", flush=True)

    def generate(self, prompts, max_tokens, temp=0.6, top_p=0.95, seeds=None):
        import mlx.core as mx
        from mlx_lm.generate import stream_generate
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=temp, top_p=top_p)
        out = []
        for i, p in enumerate(prompts):
            if seeds:
                mx.random.seed(seeds[i])
            chunks, n, t0 = [], 0, time.time()
            for r in stream_generate(self.model, self._tok, p,
                                     max_tokens=max_tokens, sampler=sampler):
                chunks.append(r.text)
                n += 1
            dt = time.time() - t0
            out.append({"text": "".join(chunks), "gen_tokens": n,
                        "seconds": round(dt, 2),
                        "gen_tps": round(n / dt, 1) if dt else 0.0,
                        "truncated": n >= max_tokens})
        return out


class VllmBackend:
    name = "vllm"

    def __init__(self, key, max_model_len=None, gpu_memory_utilization=0.92, **kw):
        from vllm import LLM
        self.path = resolve_path(key, "vllm")
        t = time.time()
        self.llm = LLM(model=self.path, dtype="bfloat16",
                       max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_memory_utilization,
                       trust_remote_code=True)
        self.tokenizer = load_tokenizer(key, "vllm")
        print(f"[vllm] loaded {self.path} in {time.time()-t:.1f}s", flush=True)

    def generate(self, prompts, max_tokens, temp=0.6, top_p=0.95, seeds=None):
        from vllm import SamplingParams
        params = [SamplingParams(temperature=temp, top_p=top_p, max_tokens=max_tokens,
                                 seed=(seeds[i] if seeds else None))
                  for i in range(len(prompts))]
        t0 = time.time()
        outs = self.llm.generate(prompts, params)
        dt = time.time() - t0
        res = []
        for o in outs:
            c = o.outputs[0]
            n = len(c.token_ids)
            res.append({"text": c.text, "gen_tokens": n,
                        # wall time is shared across the batch, so report the
                        # per-item share rather than pretending each ran alone
                        "seconds": round(dt / len(outs), 2),
                        "gen_tps": round(n / (dt / len(outs)), 1) if dt else 0.0,
                        "truncated": c.finish_reason == "length"})
        return res


def get_backend(key, backend="auto", **kw):
    b = pick_backend(backend)
    if b == "mlx":
        return MlxBackend(key, **kw)
    if b == "vllm":
        return VllmBackend(key, **kw)
    raise ValueError(f"unknown backend {b!r}")
