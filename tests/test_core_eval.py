import sys
from pathlib import Path
import numpy as np
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))
from core_eval import score_sequence, crop
from core_protocol import (
    find_common_length,
    render_prompts_mc,
    render_prompts_schema,
    render_prompts_lm,
)


def test_scoring_target_shift_and_answer_only():
    logits = np.array(
        [[9.0, 0.0, 0.0], [0.0, 1.0, 3.0], [0.0, 4.0, 1.0], [9.0, 0.0, 0.0]], dtype=np.float32
    )

    class Fake:
        def __call__(self, ids):
            return mx.array(logits[None, : ids.shape[1]])

    loss, exact = score_sequence(Fake(), [0, 0, 2, 1, 0], 2, 4)
    selected = logits[1:3].astype(np.float64)
    expected = np.mean(np.log(np.exp(selected).sum(-1)) - selected[np.arange(2), [2, 1]])
    assert abs(loss - expected) < 1e-5 and exact
    assert not score_sequence(Fake(), [0, 0, 1, 1, 0], 2, 4)[1]


def test_crop_and_shared_boundaries():
    assert crop([[0, 1, 2, 3, 4]], [4], [5], 3) == [([2, 3, 4], 2, 3, 2)]
    import pytest

    with pytest.raises(ValueError):
        crop([[0, 1, 2, 3, 4]], [1], [5], 3)
    assert find_common_length([[0, 1, 2], [0, 1, 3]]) == 2
    assert find_common_length([[0, 1, 2], [3, 1, 2]], "right") == 2


def test_prompt_bytes():
    assert render_prompts_mc(
        {"query": "Q", "choices": ["a", "b"]},
        " ",
        [{"query": "X", "choices": ["yes", "no"], "gold": 0}],
    ) == ["X yes\n\nQ a", "X yes\n\nQ b"]
    assert render_prompts_schema({"context_options": ["a", "b"], "continuation": "c"}, " ") == [
        "a c",
        "b c",
    ]
    assert render_prompts_lm({"context": " x  ", "continuation": "y"}, " ") == ["x", "x y"]
