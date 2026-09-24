import argparse
import json


def main():
    p = argparse.ArgumentParser(
        prog="agimac", description="Train and run a small language model on your Mac"
    )
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("prepare", help="Fit byte BPE and pack local JSONL code documents")
    d.add_argument("--source", required=True)
    d.add_argument("--output", required=True)
    d.add_argument("--vocab-size", type=int, default=16384)
    d.add_argument("--valid-fraction", type=float, default=0.05)
    d.add_argument("--fim-rate", type=float, default=0.5)
    d.add_argument("--seed", type=int, default=42)
    t = sub.add_parser("train")
    t.add_argument("--config", default="configs/small.json")
    t.add_argument("--data", required=True)
    t.add_argument("--output", required=True)
    t.add_argument("--steps", type=int, default=1000)
    t.add_argument("--sequence", type=int, default=512)
    t.add_argument("--batch-size", type=int, default=1)
    t.add_argument("--accumulate", type=int, default=4)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--warmup", type=int, default=100)
    t.add_argument("--eval-every", type=int, default=100)
    t.add_argument("--eval-batches", type=int, default=10)
    t.add_argument("--log-every", type=int, default=10)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    t.add_argument(
        "--master-weights", action="store_true", help="Keep FP32 master weights for BF16 training"
    )
    t.add_argument(
        "--compile-step",
        action="store_true",
        help="Compile a whole optimizer update; requires --accumulate 1",
    )
    t.add_argument("--compile", action="store_true", help="Compile loss and gradients; opt-in")
    t.add_argument(
        "--keep-last",
        type=int,
        default=0,
        help="Retain this many checkpoints in this run; 0 keeps all",
    )
    restart = t.add_mutually_exclusive_group()
    restart.add_argument("--resume", help="Checkpoint directory; restores optimizer and data RNG")
    restart.add_argument(
        "--init-from", help="Continue from checkpoint weights with a new optimizer and schedule"
    )
    g = sub.add_parser("generate")
    g.add_argument("--model", required=True)
    g.add_argument("--prompt", required=True)
    g.add_argument("--suffix", help="Use fill-in-the-middle; print only generated middle")
    g.add_argument("--tokens", type=int, default=128)
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--seed", type=int, default=42)
    c = sub.add_parser("chat", help="Interactive, streaming prompts to a local checkpoint")
    c.add_argument(
        "--model",
        default="latest",
        help="Checkpoint directory, or latest (by metadata modification time)",
    )
    c.add_argument("--runs", default="runs", help="Search root for --model latest")
    c.add_argument("--tokens", type=int, default=128)
    c.add_argument("--temperature", type=float, default=0.7)
    c.add_argument("--seed", type=int, default=42)
    b = sub.add_parser("bench")
    group = b.add_mutually_exclusive_group(required=True)
    group.add_argument("--model")
    group.add_argument("--config", help="Benchmark random weights, not quality")
    b.add_argument("--bits", type=int, choices=[4, 8])
    b.add_argument("--prompt-tokens", type=int, default=128)
    b.add_argument("--tokens", type=int, default=128)
    b.add_argument("--repeats", type=int, default=3)
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--output")
    q = sub.add_parser("quantize")
    q.add_argument("--model", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--bits", type=int, choices=[4, 8], default=4)
    e = sub.add_parser(
        "sample-tasks", help="Export coding benchmark completions without executing them"
    )
    e.add_argument("--model", required=True)
    e.add_argument("--tasks", required=True, help="JSONL: task_id and prompt, e.g. HumanEval")
    e.add_argument("--output", required=True)
    e.add_argument("--tokens", type=int, default=256)
    e.add_argument("--temperature", type=float, default=0.0)
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--limit", type=int)
    args = p.parse_args()
    if args.command == "prepare":
        from .data import prepare

        result = prepare(
            args.source, args.output, args.vocab_size, args.valid_fraction, args.seed, args.fim_rate
        )
    elif args.command == "train":
        from .train import train

        if args.log_every < 1:
            p.error("--log-every must be positive")
        result = train(args)
    elif args.command == "generate":
        import mlx.core as mx
        from .checkpoint import load_model
        from .inference import completion

        mx.random.seed(args.seed)
        model, tok, _ = load_model(args.model)
        print(completion(model, tok, args.prompt, args.tokens, args.temperature, args.suffix))
        return
    elif args.command == "bench":
        from .inference import benchmark

        result = benchmark(args)
    elif args.command == "chat":
        from .chat import chat

        try:
            chat(args)
        except (ValueError, FileNotFoundError) as error:
            p.error(str(error))
        return
    elif args.command == "quantize":
        from .checkpoint import quantize

        quantize(args.model, args.output, args.bits)
        result = {"output": args.output, "bits": args.bits}
    else:
        from .inference import sample_tasks

        result = sample_tasks(args)
    print(json.dumps(result, indent=2))
