"""Shared SFT/inference message format and a small, non-code-executing tool runtime."""

import json
import math

# Keep this checkpoint-format identifier stable even though the public name is agimac.
FORMAT = "macoder-chat-v1"
TOOL_SYSTEM = (
    "You are a helpful assistant. Calculator tools: add, subtract, multiply, divide. "
    "Each accepts numeric arguments a and b. Use a tool when requested, then answer from its result."
)


def execute_calculator(call):
    # Treat generated JSON as untrusted input and require exactly the expected fields.
    if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
        raise ValueError("Expected name and arguments")
    name, args = call["name"], call["arguments"]
    if name not in ("add", "subtract", "multiply", "divide"):
        raise ValueError("Unknown calculator tool")
    if not isinstance(args, dict) or set(args) != {"a", "b"}:
        raise ValueError("Expected exactly a and b")
    a, b = args["a"], args["b"]
    # Exact type checks exclude booleans, which Python otherwise treats as integers.
    if any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1e6 for v in (a, b)):
        raise ValueError("Arguments must be finite numbers within +/-1,000,000")
    if name == "divide" and b == 0:
        raise ValueError("Division by zero")
    # Select from four known operations; never evaluate a string as Python code.
    return {
        "result": {
            "add": lambda: a + b,
            "subtract": lambda: a - b,
            "multiply": lambda: a * b,
            "divide": lambda: a / b,
        }[name]()
    }


def render_messages(tokenizer, messages, generation=False):
    # Keep one learning flag per token: 1 means predict this token, 0 means context only.
    ids, mask = [], []

    def special(name, learn=False):
        value = tokenizer.token_to_id(name)
        if value is None:
            raise ValueError(f"Missing conversation token: {name}")
        ids.append(value)
        mask.append(int(learn))

    def text(value, learn=False):
        # Keep literal role delimiters in source text from becoming control tokens.
        encoded = tokenizer.encode(value.replace("<|", "< |")).ids
        ids.extend(encoded)
        mask.extend([int(learn)] * len(encoded))

    # Use the document boundary token as a fixed beginning for this conversation.
    special("<|endoftext|>")
    for message in messages:
        role = message["role"]
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError("Unknown message role")
        # Role markers are supplied by the formatter; only assistant content is a training target.
        special(f"<|{role}|>")
        learn = role == "assistant"
        if "tool_call" in message:
            if not learn:
                raise ValueError("Only assistants request tools")
            # Train the opening marker, JSON arguments and closing marker as one assistant response.
            special("<|tool_call|>", True)
            text(json.dumps(message["tool_call"], separators=(",", ":")), True)
            special("<|end_call|>", True)
        else:
            text(message["content"], learn)
        # Learning an end-turn token teaches the model to stop instead of continuing forever.
        special("<|end_turn|>", learn)
    # End the prompt with the assistant role so generation starts at its response content.
    if generation:
        special("<|assistant|>")
    return ids, mask


def reply(model, tokenizer, messages, max_tokens=256, temperature=0.0):
    """Generate one assistant turn; validate a tagged call before executing it.

    The model supplies the JSON arguments. The runtime performs arithmetic;
    it never evaluates generated Python, shells out, or accesses the network.
    The CLI appends a tool result before requesting the next assistant turn.
    """
    from .inference import generate_ids

    ids, _ = render_messages(tokenizer, messages, generation=True)
    # Reserve room for the prompt; shorten the requested output rather than drop history.
    limit = min(max_tokens, model.config.context - len(ids))
    if limit < 1:
        raise ValueError("Conversation exceeds context; reset or shorten it")
    output = []
    ended = False
    # Include stop tokens here to distinguish a real end-turn from a length cap.
    for token in generate_ids(model, ids, limit, temperature):
        if token in (tokenizer.token_to_id("<|end_turn|>"), tokenizer.token_to_id("<|endoftext|>")):
            ended = True
            break
        output.append(token)
    call_start = tokenizer.token_to_id("<|tool_call|>")
    call_end = tokenizer.token_to_id("<|end_call|>")
    result = {
        "text": tokenizer.decode(output, skip_special_tokens=True),
        "tokens": len(output),
        "ended": ended,
    }
    # Execute only responses beginning with the tool marker, not JSON mentioned in ordinary text.
    if output and output[0] == call_start:
        try:
            # A truncated or unfinished call must never execute, even if some JSON looks parseable.
            if output[-1] != call_end or not ended:
                raise ValueError("Incomplete tool call")
            # Remove the surrounding tags before parsing the model-generated JSON object.
            call = json.loads(tokenizer.decode(output[1:-1], skip_special_tokens=False))
            value = execute_calculator(call)
            result.update(tool_call=call, tool_result=value)
        except (ValueError, TypeError, KeyError, IndexError) as error:
            result["tool_error"] = str(error)
    return result


def conversation_cli(model, tokenizer, args):
    import time
    from .terminal import answer, tool

    # Tell the model which tools exist; it still has to generate the correct call itself.
    messages = [{"role": "system", "content": TOOL_SYSTEM}]
    print(
        "Instruction checkpoint: conversation history enabled. /reset, /exit. Calculator tools only."
    )
    while True:
        try:
            text = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return
        if text in ("/exit", "/quit"):
            return
        if text == "/reset":
            messages = messages[:1]
            print("Conversation cleared.")
            continue
        if not text:
            continue
        # Remember the previous history so errors can undo this entire unfinished user turn.
        prior_length = len(messages)
        messages.append({"role": "user", "content": text})
        try:
            # Bound consecutive tool requests so a confused model cannot loop indefinitely.
            for turn in range(3):
                start = time.perf_counter()
                result = reply(model, tokenizer, messages, args.tokens, args.temperature)
                # This user-facing timing includes prompt processing as well as generated tokens.
                elapsed = time.perf_counter() - start
                if "tool_call" in result:
                    tool(result["tool_call"], result["tool_result"])
                    # Add the actual call and computed result to history before generating the next reply.
                    messages.extend(
                        [
                            {"role": "assistant", "tool_call": result["tool_call"]},
                            {
                                "role": "tool",
                                "content": json.dumps(result["tool_result"], separators=(",", ":")),
                            },
                        ]
                    )
                    # A tool call is not the final answer: give the model another turn with the result.
                    continue
                answer(result["text"])
                print(
                    f"[{result['tokens']} tokens | {result['tokens'] / max(elapsed, 1e-9):.1f} tok/s including prefill | {'end turn' if result['ended'] else 'token cap'}]"
                )
                if "tool_error" in result:
                    print("[Tool rejected:", result["tool_error"] + "]")
                messages.append({"role": "assistant", "content": result["text"]})
                break
            else:
                print("Tool-call limit reached. /reset to start again.")
        except ValueError as error:
            del messages[prior_length:]
            print(error)
        except KeyboardInterrupt:
            del messages[prior_length:]
            print("\nGeneration stopped.")
