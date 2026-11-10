"""Train the model and save checkpoints, metrics and comparable text samples."""

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from model import CharacterTokenizer, ModelConfig, TinyTransformer


DATA = Path(__file__).resolve().parent / "data"
PROMPTS = {
    "seen_venue": "Venue: Alder House\nShift: Early\nTopic: Booking request\nDetail: ",
    "unseen_venue": "Venue: Spruce House\nShift: Late\nTopic: Lost property\nDetail: ",
}
SAMPLE_SEED = 2026
TEMPERATURE = 0.8
SAMPLE_LENGTH = 200


def make_batch(stream, starts, offsets):
    positions = starts[:, None] + offsets
    return stream[positions], stream[positions + 1]


@torch.inference_mode()
def evaluate(model, stream, offsets, batch_size):
    # Fixed, non-overlapping windows. Exclude the trailing incomplete window.
    starts = torch.arange(0, len(stream) - len(offsets), len(offsets), device=stream.device)
    total_loss = 0.0
    for batch_starts in starts.split(batch_size):
        _, loss = model(*make_batch(stream, batch_starts, offsets))
        total_loss += loss.item() * len(batch_starts)
    return total_loss / len(starts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("run"))
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if min(args.steps, args.eval_every, args.batch_size, args.threads) < 1:
        parser.error("Steps, evaluation interval, batch size and threads must be positive.")
    args.out.mkdir(parents=True, exist_ok=False)

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    raw = {split: (DATA / f"{split}.txt").read_text(encoding="utf-8")
           for split in ("train", "validation", "test")}
    tokenizer = CharacterTokenizer("".join(sorted(set(raw["train"]))))
    streams = {split: torch.tensor(tokenizer.encode(data), device=args.device)
               for split, data in raw.items()}
    config = ModelConfig(tokenizer.vocab_size)
    model = TinyTransformer(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    offsets = torch.arange(config.context, device=args.device)
    batch_rng = torch.Generator(device=args.device).manual_seed(args.seed + 1)
    sample_steps = {0, min(100, args.steps), args.steps // 2, args.steps}
    started = time.perf_counter()
    print(f"Training {sum(p.numel() for p in model.parameters()):,} parameters on {args.device}",
          flush=True)

    with (args.out / "loss.csv").open("w", encoding="utf-8", newline="") as loss_file, (
        args.out / "samples.jsonl"
    ).open("w", encoding="utf-8") as sample_file:
        metrics = csv.writer(loss_file)
        metrics.writerow(["step", "train_loss", "validation_loss"])
        for step in range(args.steps + 1):
            if step:
                starts = torch.randint(
                    len(streams["train"]) - config.context, (args.batch_size,),
                    generator=batch_rng, device=args.device,
                )
                optimizer.zero_grad(set_to_none=True)
                _, loss = model(*make_batch(streams["train"], starts, offsets))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            if step % args.eval_every == 0 or step in sample_steps:
                train_loss = evaluate(model, streams["train"], offsets, args.batch_size)
                val_loss = evaluate(model, streams["validation"], offsets, args.batch_size)
                metrics.writerow([step, f"{train_loss:.8f}", f"{val_loss:.8f}"])
                loss_file.flush()
                print(f"step {step:4d} | train {train_loss:.4f} | validation {val_loss:.4f}",
                      flush=True)
            if step in sample_steps:
                torch.save(
                    {"model": model.state_dict(), "config": asdict(config),
                     "alphabet": tokenizer.alphabet, "step": step},
                    args.out / f"step-{step:04d}.pt",
                )
                for label, prompt in PROMPTS.items():
                    rng = torch.Generator(device=args.device).manual_seed(SAMPLE_SEED)
                    tokens = torch.tensor([tokenizer.encode(prompt)], device=args.device)
                    output = model.generate(tokens, SAMPLE_LENGTH, TEMPERATURE, rng)
                    sample = {
                        "step": step, "prompt_id": label, "prompt": prompt,
                        "continuation": tokenizer.decode(output[0, tokens.size(1):].tolist()),
                    }
                    sample_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
                sample_file.flush()
            model.train()

    model.eval()
    test_loss = evaluate(model, streams["test"], offsets, args.batch_size)
    summary = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "config": asdict(config),
        "training_seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "threads": args.threads,
        "device": args.device,
        "torch_version": str(torch.__version__),
        "characters_by_split": {split: len(data) for split, data in raw.items()},
        "unknown_characters_by_split": {
            split: int((stream == 0).sum().item()) for split, stream in streams.items()
        },
        "test_loss": test_loss,
        "seconds": round(time.perf_counter() - started, 1),
        "sample_seed": SAMPLE_SEED,
        "sample_temperature": TEMPERATURE,
        "sample_length": SAMPLE_LENGTH,
    }
    (args.out / "run.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Final test loss: {test_loss:.4f}. Run took {summary['seconds']} seconds.", flush=True)


if __name__ == "__main__":
    main()
