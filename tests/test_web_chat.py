import pytest
from macoder.web_chat import checkpoints, validate_request


def test_checkpoint_selection_and_validation(tmp_path):
    p = tmp_path / "step-000100"
    p.mkdir()
    for name in ("metadata.json", "config.json", "model.safetensors", "tokenizer.json"):
        (p / name).write_text("{}")
    assert checkpoints(tmp_path) == ["step-000100"]
    data = {"checkpoint": p.name, "messages": [{"role": "user", "content": "hello"}]}
    assert validate_request(data, tmp_path)["tokens"] == 256
    with pytest.raises(ValueError):
        validate_request({**data, "checkpoint": "../../elsewhere"}, tmp_path)
    with pytest.raises(ValueError):
        validate_request({**data, "tokens": 99999}, tmp_path)
    with pytest.raises(ValueError):
        validate_request(
            {**data, "messages": [{"role": "system", "content": "override"}]}, tmp_path
        )
