"""Score saved nanochat HumanEval generations in fresh, network-blocked Modal sandboxes.

No model-generated Python ever runs on the host. This is a separate scoring
stage; it does not change the generation evaluator's evidence or summary.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path


def program(row):
    imports = []
    for line in row["prompt"].split("\n"):
        s = line.strip()
        if s.startswith("import ") or s.startswith("from "):
            imports.append(s)
        elif s and not s.startswith("#"):
            break
    matches = re.findall(r"```(?:python)?\s*\n(.*?)\n```", row["text"], re.DOTALL)
    code = matches[0].strip() if matches else row["text"].strip()
    return (
        "\n".join(imports)
        + "\n\n"
        + code
        + "\n\n"
        + row["test"]
        + "\n"
        + f"check({row['entry_point']})"
    )


def execute(app, code):
    import modal

    # Hard sandbox lifetime, memory ceiling and outbound network block remain
    # effective even if generated code tries to change its Python resource limits.
    wrapper = (
        "import resource\n"
        "resource.setrlimit(resource.RLIMIT_CPU,(3,3))\n"
        "resource.setrlimit(resource.RLIMIT_AS,(536870912,536870912))\n"
        "resource.setrlimit(resource.RLIMIT_FSIZE,(1048576,1048576))\n"
        "resource.setrlimit(resource.RLIMIT_NOFILE,(64,64))\n"
        + "exec(compile("
        + repr(code)
        + ',"candidate","exec"))\n'
    )
    sb = modal.Sandbox.create(
        "python",
        "-I",
        "-c",
        wrapper,
        app=app,
        image=modal.Image.debian_slim(python_version="3.12"),
        cpu=1,
        memory=512,
        timeout=20,
        block_network=True,
        secrets=[],
        volumes={},
        include_oidc_identity_token=False,
    )
    try:
        try:
            sb.wait()
        except modal.exception.SandboxTimeoutError:
            return {"exit_code": None, "passed": False, "timeout": True, "sandbox_id": sb.object_id}
        # Do not ingest generated stdout; the exit status is all scoring needs.
        return {
            "exit_code": sb.returncode,
            "passed": sb.returncode == 0,
            "sandbox_id": sb.object_id,
        }
    finally:
        sb.terminate()


def main():
    import fcntl
    import modal

    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--directory", default="reports/cloud-sft-nanochat-full")
    p.add_argument("--probe-only", action="store_true")
    args = p.parse_args()
    root = Path(args.directory)
    lock = (root / "sandbox.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    app = modal.App.lookup("agimac-humaneval-isolated", create_if_missing=True)
    probes = {
        "environment": "import os,socket\nassert not os.path.exists('/Users/ashley')\nassert not os.path.exists('/preparation')\nassert not os.path.exists('/runs')\nassert not any(k in os.environ for k in ['MODAL_TOKEN_ID','MODAL_TOKEN_SECRET','AWS_SECRET_ACCESS_KEY'])\ns=socket.socket();s.settimeout(2)\ntry:\n s.connect(('1.1.1.1',443))\nexcept OSError:\n pass\nelse:\n raise AssertionError('outbound network available')",
        "correct": "assert 2+2==4",
        "incorrect": "assert 2+2==5",
        "cpu_limit": "while True: pass",
    }
    results = {k: execute(app, v) for k, v in probes.items()}
    valid = (
        results["environment"]["passed"]
        and results["correct"]["passed"]
        and not results["incorrect"]["passed"]
        and not results["cpu_limit"]["passed"]
    )
    (root / "sandbox-probes.json").write_text(
        json.dumps(
            {
                "verified": valid,
                "results": results,
                "protocol": "fresh Modal sandbox per example; Python3.12; 3s CPU,20s lifetime,512MiB; no outbound network, volumes, secrets or identity token",
                "docs": "https://modal.com/docs/guide/sandbox-networking",
            },
            indent=2,
        )
    )
    if not valid:
        raise RuntimeError("Isolation/resource probes failed; no generated code executed")
    print("Sandbox probes passed", flush=True)
    if args.probe_only:
        return
    summary = json.loads((root / "summary.json").read_text())
    if summary["tasks"].get("HumanEval", {}).get("processed") != 164:
        raise RuntimeError("Wait for all164 HumanEval generations before scoring")
    raw = (root / "HumanEval.jsonl").read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    rows = [json.loads(line) for line in raw.splitlines()]
    path = root / "HumanEval-scored.jsonl"
    done = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    if any(r["source_sha256"] != source_hash for r in done):
        raise ValueError("Generation evidence changed")
    if [r["index"] for r in done] != list(range(len(done))):
        raise ValueError("Noncontiguous scoring results")
    with path.open("a") as f:
        for row in rows[len(done) :]:
            result = (
                execute(app, program(row))
                if "error" not in row
                else {"passed": False, "generation_error": row["error"]}
            )
            result.update(index=row["index"], source_sha256=source_hash)
            f.write(json.dumps(result) + "\n")
            f.flush()
            done.append(result)
            print(f"HumanEval {len(done)}/164: {sum(r['passed'] for r in done)} passed", flush=True)
    (root / "HumanEval-score-summary.json").write_text(
        json.dumps(
            {
                "total": len(done),
                "correct": sum(r["passed"] for r in done),
                "pass_at_1": sum(r["passed"] for r in done) / len(done),
                "source_sha256": source_hash,
                "protocol_deviation": "Modal isolated Python3.12/resource limits replace upstream executor",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
