"""Confirm a model loads, the chat template honours the reasoning flag, and
generation works. Run this before any long sweep on a new machine."""
import argparse
from common import get_backend, build_prompt, strip_think

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--backend", default="auto", choices=["auto", "mlx", "vllm"])
ap.add_argument("--max-model-len", type=int, default=4096)
ap.add_argument("--gpu-mem", type=float, default=0.92)
a = ap.parse_args()

backend = get_backend(a.model, a.backend, max_model_len=a.max_model_len,
                      gpu_memory_utilization=a.gpu_mem)

for thinking in (True, False):
    p = build_prompt(backend.tokenizer,
                     "What is 84 * 3 / 2? Answer with the number only.",
                     thinking=thinking)
    print(f"\n--- enable_thinking={thinking} ---")
    print("prompt tail:", repr(p[-110:]))
    g = backend.generate([p], 2048, temp=0.0)[0]
    r, ans = strip_think(g["text"])
    print(f"{g['gen_tokens']} tok  {g['gen_tps']} tok/s  truncated={g['truncated']}")
    print(f"reasoning chars: {len(r)}")
    print("answer:", repr((ans or g["text"])[:160]))
