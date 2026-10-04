"""Build and push a Kaggle GPU kernel for one job, or fetch its outputs.

    python kaggle/launch.py push final-a
    python kaggle/launch.py status final-a
    python kaggle/launch.py fetch final-a
"""
import json
import os
import shutil
import subprocess
import sys

USER = "aadigoyal42"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

RESNET = ["--epochs", "200", "--bs", "128", "--opt", "sgd", "--lr", "0.1", "--wd", "5e-4", "--warmup", "2"]
VIT = ["--epochs", "200", "--bs", "256", "--opt", "adamw", "--lr", "1e-3", "--wd", "0.05", "--warmup", "5",
       "--label-smoothing", "0.1", "--mix", "--clip", "1.0", "--randaug"]
KD = ["--distill", "0.5", "--kd-temp", "3"]


def run(name, model, args, seed):
    return {"name": f"{name}_s{seed}", "steps": [
        ["src.train", "--model", model, "--seed", str(seed), "--out", "{out}", "--data", "{data}", *args],
        ["src.evaluate", "--run", "{out}", "--data", "{data}", "--no-latency"],
    ]}


def r18(s):
    return run("resnet18", "resnet18", RESNET, s)


def ee(s):
    return run("ee_resnet18_kd", "ee_resnet18", RESNET + KD, s)


def vit(s):
    return run("ee_vit_tiny", "ee_vit_tiny", VIT + KD, s)


POOL = run("ee_resnet18_pool", "ee_resnet18_pool", RESNET + KD, 0)
CE = run("ee_resnet18_ce", "ee_resnet18", RESNET, 0)

# one list of runs per GPU; each Kaggle session has two T4s
JOBS = {
    "final-a": {"queues": [[vit(0), ee(1), r18(2)], [vit(2), POOL, r18(0)]]},
    "final-b": {"queues": [[vit(1), ee(2)], [ee(0), CE, r18(1)]]},
    "latency": {"queues": [[{"name": "latency", "steps": [
        ["src.latency", "--runs", "/kaggle/input", "--only", "_s0$", "--out", "{work}",
         "--data", "{data}"]]}]],
        "sources": ["final-a", "final-b"]},
}


def kaggle(*args):
    exe = shutil.which("kaggle") or os.path.join(os.path.dirname(sys.executable), "kaggle")
    subprocess.run([exe, *args], check=True)


def slug(job):
    return f"{USER}/adaptive-inference-{job}"


def push(job):
    d = os.path.join(ROOT, "runs", "kaggle", job)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(HERE, "job_template.py")) as f:
        script = f.read().replace("QUEUES = []", "QUEUES = " + json.dumps(JOBS[job]["queues"], indent=1))
    with open(os.path.join(d, "run.py"), "w") as f:
        f.write(script)
    meta = {"id": slug(job), "title": slug(job).split("/")[1], "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [], "competition_sources": [],
            "kernel_sources": [slug(s) for s in JOBS[job].get("sources", [])], "machine_shape": "NvidiaTeslaT4"}
    with open(os.path.join(d, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=1)
    kaggle("kernels", "push", "-p", d)


def fetch(job):
    d = os.path.join(ROOT, "runs", job)
    os.makedirs(d, exist_ok=True)
    kaggle("kernels", "output", slug(job), "-p", d, "-o")


if __name__ == "__main__":
    action, job = sys.argv[1], sys.argv[2]
    {"push": push, "fetch": fetch, "status": lambda j: kaggle("kernels", "status", slug(j))}[action](job)
