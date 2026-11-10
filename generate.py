"""Sample a continuation from a saved Tiny Transformer checkpoint."""

import argparse
import math
from pathlib import Path

import torch

from model import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt")
    prompts.add_argument("--prompt-file", type=Path)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    prompt = args.prompt if args.prompt is not None else args.prompt_file.read_text(encoding="utf-8")
    if not prompt or args.count < 1 or not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("Provide a nonempty prompt, a positive count and a finite positive temperature.")

    torch.set_num_threads(1)
    model, tokenizer = load_checkpoint(args.checkpoint)
    tokens = torch.tensor([tokenizer.encode(prompt)])
    rng = torch.Generator().manual_seed(args.seed)
    output = model.generate(tokens, args.count, args.temperature, rng)
    print(prompt + tokenizer.decode(output[0, tokens.size(1):].tolist()))


if __name__ == "__main__":
    main()
