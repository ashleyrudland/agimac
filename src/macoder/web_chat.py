"""One short-lived inference worker per web request; GPU memory is freed on exit."""

import json
import sys
import time
from pathlib import Path


def checkpoints(run):
    # Only expose published checkpoints belonging to this run, never arbitrary paths.
    names = []
    for path in sorted(run.glob("step-*"), reverse=True):
        if (
            path.is_dir()
            and not path.is_symlink()
            and all(
                (path / f).is_file()
                for f in ("metadata.json", "config.json", "model.safetensors", "tokenizer.json")
            )
        ):
            names.append(path.name)
    return names


def validate_request(data, run):
    if not isinstance(data, dict):
        raise ValueError("Expected a chat request")
    name = data.get("checkpoint")
    if name not in checkpoints(run):
        raise ValueError("Checkpoint no longer available; select a newer checkpoint and reset")
    messages = data.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 60:
        raise ValueError("Need 1–60 messages; reset long conversations")
    for m in messages:
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant", "tool"):
            raise ValueError("Invalid message role")
        if "tool_call" in m:
            if m["role"] != "assistant":
                raise ValueError("Only assistants request tools")
            from .conversation import execute_calculator

            execute_calculator(m["tool_call"])
        elif not isinstance(m.get("content"), str):
            raise ValueError("Message content must be text")
    if messages[-1]["role"] != "user" or not messages[-1].get("content", "").strip():
        raise ValueError("Enter a message")
    tokens = data.get("tokens", 256)
    if type(tokens) is not int or not 1 <= tokens <= 512:
        raise ValueError("Output limit must be 1–512 tokens")
    return {"model": str(run / name), "messages": messages, "tokens": tokens}


def worker(request):
    from .checkpoint import load_model
    from .conversation import TOOL_SYSTEM, reply

    model, tok, meta = load_model(request["model"])
    model.eval()
    if meta.get("chat_format") != "macoder-chat-v1":
        raise ValueError("This is a base completion checkpoint; select an instruction checkpoint")
    history = [{"role": "system", "content": TOOL_SYSTEM}, *request["messages"]]
    added = []
    events = []
    count = 0
    start = time.perf_counter()
    for _ in range(3):
        r = reply(model, tok, history, request["tokens"], 0.0)
        count += r["tokens"]
        if "tool_error" in r:
            raise ValueError("Tool call rejected: " + r["tool_error"])
        if "tool_call" in r:
            turns = [
                {"role": "assistant", "tool_call": r["tool_call"]},
                {"role": "tool", "content": json.dumps(r["tool_result"], separators=(",", ":"))},
            ]
            history.extend(turns)
            added.extend(turns)
            events.append({"call": r["tool_call"], "result": r["tool_result"]})
            continue
        added.append({"role": "assistant", "content": r["text"]})
        elapsed = time.perf_counter() - start
        return {
            "messages": added,
            "text": r["text"],
            "tools": events,
            "ended": r["ended"],
            "tokens": count,
            "seconds": elapsed,
            "tokens_per_second": count / max(elapsed, 1e-9),
            "checkpoint": Path(request["model"]).name,
        }
    raise ValueError("Three consecutive tool calls; reset and try a simpler request")


if __name__ == "__main__":
    try:
        print(json.dumps(worker(json.load(sys.stdin))))
    except Exception as error:
        print(json.dumps({"error": str(error)}))
        sys.exit(1)
