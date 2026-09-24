import json
import platform
from pathlib import Path
import statistics
import time

import mlx.core as mx
import mlx.nn as nn

from .checkpoint import load_model
from .model import Config, Model


def generate_ids(model, prompt, max_tokens=128, temperature=0.0, eos=None):
    """Prefill the prompt once, then decode one token at a time using cached K/V.

    Temperature zero is greedy; a positive value samples scaled logits. The
    caller decides which control tokens end a turn or trigger a tool.
    """
    if not prompt or max_tokens < 1 or temperature < 0:
        raise ValueError("Need a nonempty prompt, positive max_tokens, nonnegative temperature")
    if len(prompt) + max_tokens > model.config.context:
        raise ValueError("Prompt + requested output exceeds context; shorten either explicitly")
    # Every generation request starts with empty per-layer attention caches.
    cache = model.make_cache()
    # Prefill: process the full prompt once and keep only its final next-token scores.
    logits = model(mx.array([prompt]), cache, last_only=True)[:, -1, :]
    for i in range(max_tokens):
        token = (
            mx.argmax(logits, axis=-1)
            if temperature == 0
            else mx.random.categorical(logits / temperature)
        )
        # Reading the sampled token on the CPU forces its GPU computation to finish.
        value = int(token.item())
        if value == eos:
            break
        yield value
        if i + 1 < max_tokens:
            # Decode: pass only the new token; earlier tokens are represented by cached keys/values.
            logits = model(token.reshape(1, 1), cache, last_only=True)[:, -1, :]


def completion(model, tokenizer, prompt, max_tokens=128, temperature=0.0, suffix=None):
    ids = tokenizer.encode(prompt).ids
    if suffix is not None:
        ids = (
            [tokenizer.token_to_id("<|fim_prefix|>")]
            + ids
            + [tokenizer.token_to_id("<|fim_suffix|>")]
            + tokenizer.encode(suffix).ids
            + [tokenizer.token_to_id("<|fim_middle|>")]
        )
    if not ids:
        ids = [tokenizer.token_to_id("<|endoftext|>")]
    output = list(
        generate_ids(model, ids, max_tokens, temperature, tokenizer.token_to_id("<|endoftext|>"))
    )
    return tokenizer.decode(output, skip_special_tokens=True)


def benchmark(args):
    if args.prompt_tokens < 1 or args.tokens < 2 or args.repeats < 1:
        raise ValueError("Need prompt_tokens>=1, tokens>=2, repeats>=1")
    mx.random.seed(args.seed)
    if args.model:
        model, _, meta = load_model(args.model)
        source = args.model
        parameters = meta.get("parameters", model.parameter_count())
    else:
        model = Model(Config.read(args.config))
        parameters = model.parameter_count()
        model.set_dtype(mx.float16)
        meta = {"dtype": "float16", "trained_from_scratch": False}
        source = "random weights: speed only, no quality claim"
    if args.bits:
        if "quantization" in meta:
            raise ValueError("Already quantized; omit --bits")
        model.set_dtype(mx.float16)
        nn.quantize(model, group_size=64, bits=args.bits)
        meta["quantization"] = {"group_size": 64, "bits": args.bits}
        meta["dtype"] = "float16"
    model.eval()
    mx.eval(model.parameters())
    if args.prompt_tokens + args.tokens > model.config.context:
        raise ValueError("Benchmark exceeds model context")
    prompt = mx.random.randint(0, model.config.vocab_size, (1, args.prompt_tokens))
    mx.eval(prompt)
    rows = []
    for repeat in range(args.repeats + 1):
        cache = model.make_cache()
        mx.synchronize()
        start = time.perf_counter()
        logits = model(prompt, cache, last_only=True)[:, -1, :]
        token = mx.argmax(logits, axis=-1)
        mx.eval(token)
        prefill_seconds = time.perf_counter() - start
        start = time.perf_counter()
        # Fixed output length, ignoring EOS: exactly tokens-1 single-token forwards.
        for _ in range(args.tokens - 1):
            logits = model(token.reshape(1, 1), cache, last_only=True)[:, -1, :]
            token = mx.argmax(logits, axis=-1)
            mx.eval(token)
        decode_seconds = time.perf_counter() - start
        row = {
            "prefill_tokens_per_second": args.prompt_tokens / prefill_seconds,
            "time_to_first_token_ms": prefill_seconds * 1000,
            "decode_tokens_per_second": (args.tokens - 1) / decode_seconds,
        }
        if repeat:
            rows.append(row)
    result = {
        "source": source,
        "parameters": parameters,
        "dtype": meta["dtype"],
        "quantization": meta.get("quantization"),
        "prompt_tokens": args.prompt_tokens,
        "output_tokens": args.tokens,
        "batch_size": 1,
        "warmup_runs": 1,
        "mlx_version": mx.__version__,
        "os": platform.platform(),
        "device": mx.device_info(),
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "runs": rows,
        "median": {k: statistics.median(r[k] for r in rows) for k in rows[0]},
    }
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def sample_tasks(args):
    """Export EvalPlus-style samples; never execute generated code on the host."""
    model, tok, _ = load_model(args.model)
    mx.random.seed(args.seed)
    tasks, seen = [], set()
    with Path(args.tasks).open() as f:
        for line in f:
            task = json.loads(line)
            if task["task_id"] in seen:
                raise ValueError("Duplicate task_id")
            seen.add(task["task_id"])
            tasks.append(task)
    if args.limit:
        tasks = tasks[: args.limit]
    with Path(args.output).open("x") as f:
        for task in tasks:
            text = completion(model, tok, task["prompt"], args.tokens, args.temperature)
            f.write(json.dumps({"task_id": task["task_id"], "completion": text}) + "\n")
            f.flush()
    return {"samples": len(tasks), "output": args.output, "executed": False}
