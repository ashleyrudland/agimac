"""Verify protocol compatibility without loading a model or running generated code."""

import ast
import importlib.util
import re
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "chat_eval", Path(__file__).parents[1] / "experiments/chat_eval.py"
)
evalmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evalmod)


def test_prompts_and_answers():
    prompt, letters, answer = evalmod.example(
        "MMLU", {"question": "2 + 2?", "choices": ["1", "2", "3", "4"], "answer": 3}
    )
    assert (
        prompt
        == "Multiple Choice question: 2 + 2?\n- 1=A\n- 2=B\n- 3=C\n- 4=D\n\nRespond only with the letter of the correct answer."
    )
    assert letters == list("ABCD") and answer == "D"
    _, numeric, answer = evalmod.example(
        "ARC-Easy",
        {"question": "q", "choices": {"label": ["1", "2"], "text": ["x", "y"]}, "answerKey": "2"},
    )
    assert numeric == ["1", "2"] and answer == "2"


def test_exact_gsm_extraction():
    assert evalmod.extract_answer("work\n#### -1,200.50") == "-1200.50"
    assert evalmod.extract_answer("Answer: 42") is None
    assert evalmod.extract_answer("#### 42 then #### 99") == "42"
    assert evalmod.extract_answer("#### 42.0") != evalmod.extract_answer("#### 42")


def test_context_and_resume(tmp_path):
    assert evalmod.output_limit(2048, 2000) == 48
    assert evalmod.output_limit(2048, 2048) == 0
    assert evalmod.output_limit(2048, 100) == 512
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b'{"index":0}\n{"ind')
    assert evalmod.resume_rows(path) == [{"index": 0}]
    assert path.read_bytes() == b'{"index":0}\n'


def test_pinned_upstream_functions():
    upstream = Path("/tmp/macoder-nanochat-review-20260922")
    if not upstream.exists():
        import pytest

        pytest.skip("Pinned upstream checkout unavailable")
    for filename, function, cases in [
        ("tasks/common.py", "render_mc", [("Question?", ["A", "B"], ["one", "two"])]),
        ("tasks/gsm8k.py", "extract_answer", [("#### -1,200.50",), ("answer 42",), ("#### 42.0",)]),
    ]:
        tree = ast.parse((upstream / filename).read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
        namespace = {"re": re, "GSM_RE": re.compile(r"#### (\-?[0-9\.\,]+)")}
        exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), namespace)
        for args in cases:
            assert namespace[function](*args) == getattr(evalmod, function)(*args)
