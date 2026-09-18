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
    if local and os.path.isdir(local):
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


def prepare_cuda_env():
    """Make a pip-only CUDA install usable without a system CUDA toolkit.

    `pip install vllm` pulls its own nvcc and ninja but leaves neither on PATH,
    so vLLM's kernel compilation fails with "Could not find nvcc". Point CUDA_HOME
    at the wheel's toolkit and put both on PATH.

    flashinfer's vendored CCCL headers do not compile against that nvcc, so its
    JIT sampling kernel is disabled and vLLM falls back to the PyTorch sampler.
    That changes which sampling code path runs, so it is printed rather than
    applied silently: a suite must not straddle this setting any more than it
    may straddle backends.

    Anything already set in the environment wins, so this can be overridden.
    """
    import glob, sys
    changed = []
    root = os.path.dirname(os.path.dirname(os.path.abspath(sys.executable)))
    if not os.environ.get("CUDA_HOME"):
        hits = sorted(glob.glob(os.path.join(
            root, "lib", "python*", "site-packages", "nvidia", "cu*", "bin", "nvcc")))
        if hits:
            home = os.path.dirname(os.path.dirname(hits[-1]))
            os.environ["CUDA_HOME"] = home
            changed.append(f"CUDA_HOME={home}")
    binpath = os.path.join(root, "bin")
    extra = [p for p in (binpath, os.path.join(os.environ.get("CUDA_HOME", ""), "bin"))
             if p and os.path.isdir(p) and p not in os.environ.get("PATH", "").split(os.pathsep)]
    if extra:
        os.environ["PATH"] = os.pathsep.join(extra + [os.environ.get("PATH", "")])
        changed.append("PATH+=" + os.pathsep.join(extra))
    if "VLLM_USE_FLASHINFER_SAMPLER" not in os.environ:
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        changed.append("VLLM_USE_FLASHINFER_SAMPLER=0 (torch sampler, not flashinfer)")
    for c in changed:
        print(f"[vllm-env] {c}", flush=True)


class VllmBackend:
    name = "vllm"

    def __init__(self, key, max_model_len=None, gpu_memory_utilization=0.92,
                 enforce_eager=False, **kw):
        prepare_cuda_env()
        from vllm import LLM
        self.path = resolve_path(key, "vllm")
        t = time.time()
        # CUDA graphs cost roughly 0.9 GiB of the KV budget. On an 8 GB card that
        # is most of the cache: MiniCPM5-2B measures 12,928 KV tokens with graphs
        # and 40,432 without, so long-context runs need enforce_eager even though
        # it gives up some decode speed.
        self.llm = LLM(model=self.path, dtype="bfloat16",
                       max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_memory_utilization,
                       enforce_eager=enforce_eager,
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

    def generate_stream(self, prompts, max_tokens, temp=0.6, top_p=0.95,
                        seeds=None, window=None):
        """Yield (index, record) for each prompt the moment it finishes.

        At most `window` prompts are in the engine at once; as each finishes the
        next is added, so one long answer never holds a finished batch hostage
        and every answer can be saved before the next power cut.

        `seconds` is wall time from submission to finish. With several prompts
        in flight that overlaps other work, so it is latency, not a per-answer
        cost; `gen_tps` is derived from it and means the same.
        """
        from vllm import SamplingParams
        engine = self.llm.llm_engine
        window = max(1, window or len(prompts))
        pending = list(range(len(prompts)))
        started = {}

        def submit(i):
            engine.add_request(str(i), prompts[i],
                               SamplingParams(temperature=temp, top_p=top_p,
                                              max_tokens=max_tokens,
                                              seed=(seeds[i] if seeds else None)))
            started[i] = time.time()

        while pending and len(started) < window:
            submit(pending.pop(0))
        while engine.has_unfinished_requests():
            for o in engine.step():
                if not o.finished:
                    continue
                i = int(o.request_id)
                c = o.outputs[0]
                n = len(c.token_ids)
                dt = time.time() - started.pop(i)
                if pending:
                    submit(pending.pop(0))
                yield i, {"text": c.text, "gen_tokens": n,
                          "seconds": round(dt, 2),
                          "gen_tps": round(n / dt, 1) if dt else 0.0,
                          "truncated": c.finish_reason == "length"}


def get_backend(key, backend="auto", **kw):
    b = pick_backend(backend)
    if b == "mlx":
        return MlxBackend(key, **kw)
    if b == "vllm":
        return VllmBackend(key, **kw)
    raise ValueError(f"unknown backend {b!r}")
