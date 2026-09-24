"""Interactive base-model completion. No invented chat template or tool execution."""

import json
import math
from pathlib import Path
import time

import mlx.core as mx
from tokenizers.decoders import DecodeStream

from .checkpoint import load_model
from .inference import generate_ids


def latest_checkpoint(root):
    """Select the newest complete-looking trained checkpoint, including quantized copies.

    Metadata is written after weights/tokenizer. This filters interrupted saves,
    but actual weight integrity is still checked by load_model.
    """
    candidates = []
    for metadata in Path(root).glob("**/metadata.json"):
        path = metadata.parent
        # Ignore directories the trainer has not yet finished publishing.
        if path.name.startswith(".pending-"):
            continue
        if not all(
            (path / name).is_file()
            for name in ("config.json", "tokenizer.json", "model.safetensors")
        ):
            continue
        try:
            meta = json.loads(metadata.read_text())
            # Do not automatically select an untrained speed-test checkpoint.
            trained = int(meta.get("tokens_seen", 0)) > 0
        except (OSError, ValueError, TypeError, AttributeError):
            continue
        if trained:
            candidates.append((metadata.stat().st_mtime_ns, str(path.resolve()), path))
    if not candidates:
        raise FileNotFoundError(f"No trained checkpoint found under {root}; pass --model PATH")
    # Latest means newest metadata timestamp; it does not mean best validation score.
    return max(candidates)[2]


def prompt_ids(tokenizer, text, context, requested):
    ids = tokenizer.encode(text).ids or [tokenizer.token_to_id("<|endoftext|>")]
    # The prompt and generated reply share one fixed context budget.
    available = context - len(ids)
    if available < 1:
        raise ValueError(
            f"Prompt uses {len(ids)} tokens; context is {context}. Shorten it or /reset."
        )
    return ids, min(requested, available)


def chat(args):
    if args.tokens < 1 or not math.isfinite(args.temperature) or args.temperature < 0:
        raise ValueError("Need positive --tokens and finite nonnegative --temperature")
    path = latest_checkpoint(args.runs) if args.model == "latest" else Path(args.model)
    model, tokenizer, meta = load_model(path)
    mx.random.seed(args.seed)
    from .terminal import banner

    banner()
    print(f"Checkpoint: {path.resolve()}")
    bits = meta.get("quantization", {}).get("bits")
    precision = f"{bits}-bit weights" if bits else meta.get("dtype", "float32")
    print(
        f"{meta.get('parameters', model.parameter_count()):,} parameters | "
        f"{model.config.context} token context | {precision}"
    )
    # Only instruction checkpoints use roles, persistent conversation history and tool handling.
    if meta.get("chat_format") == "macoder-chat-v1":
        from .conversation import conversation_cli

        conversation_cli(model, tokenizer, args)
        return
    print("Base model: continues text; it has not been trained for chat or tool calls.")
    print("Each prompt starts fresh. /continue extends the previous text. /help lists commands.")
    print("Try: Once upon a time, a little robot found a key")
    previous = ""
    while True:
        try:
            text = input("\nYou> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return
        if not text.strip():
            continue
        if text.strip() in ("/exit", "/quit"):
            print("Bye.")
            return
        if text.strip() == "/help":
            print(
                "/continue  Continue previous prompt + output (new cache)\n"
                "/reset     Discard previous text\n"
                "/tokens N  Set maximum generated tokens\n"
                "/temp N    Set temperature; 0 = greedy\n"
                "/exit      Quit; Ctrl-C stops generation or exits at the prompt"
            )
            continue
        if text.strip() == "/reset":
            previous = ""
            print("Previous text cleared.")
            continue
        if text.startswith(("/tokens", "/temp")):
            try:
                command, value = text.split()
                if command == "/tokens":
                    value = int(value)
                    if value < 1:
                        raise ValueError
                    args.tokens = value
                elif command == "/temp":
                    value = float(value)
                    if not math.isfinite(value) or value < 0:
                        raise ValueError
                    args.temperature = value
                else:
                    raise ValueError
                print(f"tokens={args.tokens}, temperature={args.temperature:g}")
            except ValueError:
                print(
                    "Use /tokens with a positive integer or /temp with a finite nonnegative number."
                )
            continue
        # For a base model, continuation explicitly feeds the previous prompt and output back in.
        if text.strip() == "/continue":
            if not previous:
                print("Nothing to continue yet.")
                continue
            text = previous
        elif text.startswith("/"):
            print("Unknown command. Type /help.")
            continue
        try:
            ids, limit = prompt_ids(tokenizer, text, model.config.context, args.tokens)
        except ValueError as error:
            print(error)
            continue
        if limit < args.tokens:
            print(f"Output capped at {limit} tokens to fit the context; prompt was not truncated.")
        output, first, last = [], None, None
        # Buffer partial byte sequences so streamed tokens render as valid text.
        decoder = DecodeStream(skip_special_tokens=True)
        print("Model> ", end="", flush=True)
        # Clear pending GPU work before measuring this response.
        mx.synchronize()
        start = time.perf_counter()
        interrupted = False
        try:
            for token in generate_ids(
                model, ids, limit, args.temperature, tokenizer.token_to_id("<|endoftext|>")
            ):
                last = time.perf_counter()
                if first is None:
                    first = last
                output.append(token)
                chunk = decoder.step(tokenizer, token)
                if chunk:
                    print(chunk, end="", flush=True)
        except KeyboardInterrupt:
            interrupted = True
        elapsed = time.perf_counter() - start
        # Retain text for /continue, but do not keep the GPU cache between separate requests.
        previous = text + tokenizer.decode(output, skip_special_tokens=True)
        if interrupted:
            reason = "stopped"
        elif len(output) == limit:
            reason = "token limit"
        else:
            reason = "EOS"
        # Exclude the first token from decode speed because it also pays the prompt-processing cost.
        rate = (len(output) - 1) / (last - first) if len(output) > 1 and last > first else None
        rate_text = f"{rate:.1f} tok/s decode" if rate is not None else "decode rate n/a"
        ttft = (
            f"{(first - start) * 1000:.1f} ms first token"
            if first is not None
            else "no text tokens"
        )
        print(f"\n[{len(output)} tokens | {rate_text} | {ttft} | {elapsed:.2f}s | {reason}]")
        print("[Live timing includes streaming overhead; model loading excluded.]")
