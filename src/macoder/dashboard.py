"""Localhost training dashboard with isolated checkpoint inference. Training only launches this separate process."""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import webbrowser
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def snapshot(run, info):
    info = dict(info)
    meta = {}
    # Standalone viewing can recover model/schedule details from a saved checkpoint.
    try:
        checkpoint = Path((run / "latest.txt").read_text().strip())
        meta = read_json(checkpoint / "metadata.json", {})
        settings = read_json(checkpoint / "trainer.json", {}).get("args", {})
        info.setdefault("parameters", meta.get("parameters"))
        info.setdefault("total_steps", settings.get("steps"))
        if all(k in settings for k in ("batch_size", "sequence", "accumulate")):
            info.setdefault(
                "tokens_per_step",
                settings["batch_size"] * settings["sequence"] * settings["accumulate"],
            )
    except OSError:
        pass
    # A writer may be halfway through its last line; ignore only incomplete records.
    rows = []
    metrics = run / "metrics.jsonl"
    try:
        for line in metrics.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    except FileNotFoundError:
        pass
    last = rows[-1] if rows else {}
    total = last.get("total_steps", info.get("total_steps"))
    samples = rows[-200:]
    key = (
        "input_tokens_per_second"
        if "input_tokens_per_second" in last
        else "train_tokens_per_second"
    )
    speed = (
        statistics.median(r[key] for r in samples if key in r)
        if any(key in r for r in samples)
        else None
    )
    tokens = last.get("actual_input_tokens", last.get("input_tokens"))
    if tokens is None and info.get("tokens_per_step"):
        tokens = last.get("step", 0) * info["tokens_per_step"]
    eta = None
    if total and len(rows) > 20 and "elapsed_seconds" in last:
        first = rows[max(0, len(rows) - 501)]
        seconds = (last["elapsed_seconds"] - first["elapsed_seconds"]) / max(
            1, last["step"] - first["step"]
        )
        eta = max(0, total - last["step"]) * seconds
    elif total and speed and info.get("tokens_per_step"):
        eta = max(0, total - last.get("step", 0)) * info["tokens_per_step"] / speed
    alive = None
    if info.get("parent"):
        try:
            os.kill(info["parent"], 0)
            alive = True
        except ProcessLookupError:
            alive = False
        except PermissionError:
            alive = None
    age = time.time() - metrics.stat().st_mtime if metrics.exists() else None
    summary = read_json(run / "summary.json", {})
    stopped = summary.get("status") == "stopped"
    complete = (run / "summary.json").exists() and not stopped
    state = (
        "Training complete"
        if complete
        else (
            "Process exited"
            if alive is False
            else (
                "No recent metrics"
                if age and age > 120
                else ("Receiving metrics" if rows else "Waiting for first step")
            )
        )
    )
    if stopped:
        state = "Training stopped"
        eta = None
        speed = None
    validation = [[r["step"], r["valid_loss"]] for r in rows if "valid_loss" in r]
    initial = read_json(run / "initial-evaluation.json", {})
    if "valid_loss" in initial:
        validation.insert(0, [0, initial["valid_loss"]])
    # Send at most about 1,200 points; smooth before sampling, not after.
    stride = max(1, len(rows) // 1200)
    points = []
    rolling = []
    acc = 0
    for i, r in enumerate(rows):
        v = r["loss"]
        rolling.append(v)
        acc += v
        if len(rolling) > 50:
            acc -= rolling.pop(0)
        if i % stride == 0 or i == len(rows) - 1:
            points.append([r["step"], acc / len(rolling)])
    return {
        "run": run.name,
        "state": state,
        "step": last.get("step", 0),
        "total": total,
        "parameters": info.get("parameters"),
        "tokens": tokens,
        "initial_tokens": meta.get("initial_tokens_seen", 0),
        "total_tokens": (meta.get("initial_tokens_seen", 0) + tokens)
        if tokens is not None
        else meta.get("tokens_seen"),
        "training_backend": "Modal GPU · conversation/tool SFT"
        if meta.get("backend") == "torch-cuda-sft"
        else (
            "Modal GPU training" if meta.get("backend") == "torch-cuda" else "local MLX training"
        ),
        "assistant_targets": last.get("assistant_targets"),
        "speed": speed,
        "eta": eta,
        "memory": last.get("peak_memory_gb"),
        "validation": validation,
        "loss": points,
        "age": age,
        "complete": complete,
        "updated": time.strftime("%H:%M:%S"),
        "gpu": gpu_stats(),
    }


def pipeline_run(pipeline):
    """Follow only this pipeline's outputs; never substitute the previous experiment."""
    root = pipeline.parent.parent
    status = read_json(pipeline, {})
    command = status.get("command", [])
    if "--output" in command and status.get("stage") in (
        "pretraining",
        "conversation_and_calculator_training",
    ):
        return root / command[command.index("--output") + 1]
    candidates = [
        p
        for prefix in ("general-v2", "sft-v2")
        for p in (root / "runs").glob(prefix + "*")
        if p.is_dir() and (p / "metrics.jsonl").exists()
    ]
    return (
        max(candidates, key=lambda p: (p / "metrics.jsonl").stat().st_mtime)
        if candidates
        else root / "runs/general-v2"
    )


def pipeline_snapshot(pipeline):
    status = read_json(pipeline, {})
    run = pipeline_run(pipeline)
    stage = status.get("stage", "starting")
    info = {
        "parameters": status.get("parameters"),
        "parent": status.get("child_pid") or status.get("pid"),
    }
    if not run.name.startswith("sft-v2"):
        info.update(total_steps=status.get("target_steps"), tokens_per_step=4096)
    result = snapshot(run, info)
    states = {
        "preparing_data": "Preparing training data",
        "waiting_for_previous_core": "Waiting for previous CORE evaluation",
        "pretraining": "Training base model",
        "conversation_and_calculator_training": "Training conversations and tools",
        "complete": "Pipeline complete",
        "failed": "Pipeline failed",
    }
    result.update(
        run="agimac v2 · 145M · " + run.name,
        stage=stage,
        state=states.get(stage, stage.replace("_", " ").capitalize()),
        complete=stage == "complete",
    )
    details = ""
    if stage == "preparing_data":
        sources = list((pipeline.parent.parent / "data/general-v2").glob("*.jsonl"))
        size = sum(p.stat().st_size for p in sources) / 1e9
        details = f"{size:.2f} GB of source documents staged · target: 2B packed tokens. GPU training has not started."
        result.update(
            loss=[], validation=[], step=0, tokens=0, speed=None, eta=None, memory=None, age=None
        )
    elif stage == "waiting_for_previous_core":
        details = "Dataset prepared. Training starts automatically after the earlier benchmark job finishes."
    elif stage == "cancelled":
        details = "Cancelled at your request. Saved files are retained."
        result.update(speed=None, eta=None)
    elif stage == "failed":
        details = status.get("error", "See runs-v2.log for details.")
    elif stage == "pretraining":
        details = "11B processed-token budget · loss refreshes every 5 seconds."
    else:
        details = "Following the current pipeline phase automatically."
    if (
        stage
        not in ("preparing_data", "waiting_for_previous_core", "complete", "failed", "cancelled")
        and result.get("age", 0)
        and result["age"] > 180
    ):
        details += (
            " No recent training metrics; this phase may be evaluating or saving a checkpoint."
        )
    result["details"] = details
    return result


def gpu_stats():
    # macOS driver counters are system-wide, not attributable to this trainer.
    import re

    try:
        output = subprocess.check_output(
            ["ioreg", "-r", "-c", "AGXAccelerator", "-l"], text=True, timeout=2
        )
        util = re.search(r'"Device Utilization %"=(\d+)', output)
        memory = re.search(r'"In use system memory"=(\d+)', output)
        return {
            "utilization": int(util[1]) if util else None,
            "memory_gb": int(memory[1]) / 1e9 if memory else None,
        }
    except (OSError, subprocess.SubprocessError):
        return {}


def launch(run, **info):
    """Best-effort UI launch; a browser failure must never stop a training job."""
    if os.environ.get("AGIMAC_DASHBOARD", "1") == "0":
        return
    run = Path(run).resolve()
    try:
        with (run / "dashboard.log").open("a") as log:
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "macoder.dashboard",
                    "--run",
                    str(run),
                    "--parent",
                    str(os.getpid()),
                    "--info",
                    json.dumps(info),
                    "--open",
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
    except OSError as e:
        print("Dashboard unavailable:", e, flush=True)


def serve(args):
    run = args.run.resolve()
    if not run.is_dir():
        raise SystemExit("Run directory does not exist")
    pipeline = getattr(args, "pipeline", None)
    if pipeline:
        pipeline = pipeline.resolve()
    info = json.loads(args.info)
    info["parent"] = args.parent
    token = secrets.token_urlsafe(32)
    html = (
        Path(__file__)
        .with_name("dashboard.html")
        .read_text()
        .replace("__CHAT_TOKEN__", token)
        .encode()
    )
    inference_lock = threading.Lock()
    cache = {"time": 0, "data": None}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/api/status":
                if time.time() - cache["time"] > 4:
                    cache.update(
                        time=time.time(),
                        data=pipeline_snapshot(pipeline) if pipeline else snapshot(run, info),
                    )
                body = json.dumps(cache["data"], allow_nan=False).encode()
                kind = "application/json"
            elif self.path == "/api/checkpoints":
                from .web_chat import checkpoints

                chat_run = pipeline_run(pipeline) if pipeline else run
                names = checkpoints(chat_run) if chat_run.exists() else []
                if pipeline:
                    names = [
                        n
                        for n in names
                        if read_json(chat_run / n / "metadata.json", {}).get("chat_format")
                    ]
                body = json.dumps({"checkpoints": names, "run": chat_run.name}).encode()
                kind = "application/json"
            elif self.path in ("/", "/index.html"):
                body = html
                kind = "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/api/chat":
                self.send_error(404)
                return
            # A per-server secret prevents other web pages from initiating inference.
            if not secrets.compare_digest(self.headers.get("X-Chat-Token", ""), token):
                self.send_error(403)
                return
            locked = False
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 65536:
                    raise ValueError("Request too large; reset the conversation")
                from .web_chat import validate_request

                request = validate_request(
                    json.loads(self.rfile.read(size)), pipeline_run(pipeline) if pipeline else run
                )
                locked = inference_lock.acquire(blocking=False)
                if not locked:
                    raise ValueError("Another reply is running; try again shortly")
                # Separate process: training continues and model allocations disappear after the reply.
                process = subprocess.run(
                    [sys.executable, "-m", "macoder.web_chat"],
                    input=json.dumps(request),
                    text=True,
                    capture_output=True,
                    timeout=90,
                )
                result = json.loads(process.stdout)
                code = 400 if "error" in result else 200
            except subprocess.TimeoutExpired:
                result = {"error": "Reply timed out after 90 seconds; try a shorter request"}
                code = 408
            except (ValueError, OSError) as e:
                result = {"error": str(e)}
                code = 400
            finally:
                if locked:
                    inference_lock.release()
            body = json.dumps(result).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{server.server_port}"
    (run / ("v2-dashboard.json" if pipeline else "dashboard.json")).write_text(
        json.dumps({"url": url, "pid": os.getpid(), "parent": args.parent, **info}) + "\n"
    )
    print("agimac dashboard:", url, flush=True)
    if args.open:
        webbrowser.open(url)
    server.serve_forever()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--parent", type=int)
    p.add_argument(
        "--pipeline",
        type=Path,
        help="Follow pipeline status across preparation and training phases",
    )
    p.add_argument("--info", default="{}")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--open", action="store_true")
    serve(p.parse_args())
