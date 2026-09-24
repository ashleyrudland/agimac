import json
from macoder import dashboard


def test_snapshot_handles_partial_write_and_completion(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "gpu_stats", lambda: {})
    rows = [
        {
            "step": i,
            "loss": 2.0,
            "elapsed_seconds": float(i),
            "actual_input_tokens": i * 100,
            "input_tokens_per_second": 100,
            "total_steps": 100,
        }
        for i in range(1, 31)
    ]
    rows[-1]["valid_loss"] = 1.9
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + '\n{"step":')
    info = {"parameters": 42}
    s = dashboard.snapshot(tmp_path, info)
    assert s["step"] == 30 and s["tokens"] == 3000 and s["eta"] == 70
    assert s["validation"] == [[30, 1.9]] and not s["complete"]
    (tmp_path / "summary.json").write_text("{}")
    assert dashboard.snapshot(tmp_path, info)["state"] == "Training complete"


def test_snapshot_empty_run(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "gpu_stats", lambda: {})
    s = dashboard.snapshot(tmp_path, {})
    assert s["step"] == 0 and s["loss"] == [] and s["speed"] is None


def test_pipeline_preparation_never_shows_old_completion(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "gpu_stats", lambda: {})
    runs = tmp_path / "runs"
    runs.mkdir()
    old = runs / "sft-v1"
    old.mkdir()
    (old / "summary.json").write_text("{}")
    state = runs / "v2-status.json"
    state.write_text(
        json.dumps({"stage": "preparing_data", "parameters": 145000000, "target_steps": 100})
    )
    result = dashboard.pipeline_snapshot(state)
    assert result["state"] == "Preparing training data" and not result["complete"]
    assert result["loss"] == [] and result["step"] == 0
    assert dashboard.pipeline_run(state) == runs / "general-v2"
    assert not (runs / "general-v2").exists()  # Viewing must not pre-create the trainer's output.


def test_pipeline_follows_training_then_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "gpu_stats", lambda: {})
    runs = tmp_path / "runs"
    runs.mkdir()
    run = runs / "general-v2-resume-001"
    run.mkdir()
    (run / "metrics.jsonl").write_text(
        json.dumps({"step": 20, "loss": 3.0, "train_tokens_per_second": 4000}) + "\n"
    )
    state = runs / "v2-status.json"
    state.write_text(
        json.dumps(
            {
                "stage": "pretraining",
                "parameters": 145000000,
                "target_steps": 100,
                "command": ["train", "--output", "runs/general-v2-resume-001"],
            }
        )
    )
    result = dashboard.pipeline_snapshot(state)
    assert result["step"] == 20 and result["tokens"] == 81920 and result["loss"] == [[20, 3.0]]
    assert not result["complete"]
    (run / "summary.json").write_text("{}")
    state.write_text(json.dumps({"stage": "chat_diagnostics", "target_steps": 100}))
    assert dashboard.pipeline_snapshot(state)["complete"] is False
