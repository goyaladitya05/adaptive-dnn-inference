"""Build and push a Kaggle GPU kernel for one job, or fetch its outputs.

    python kaggle/launch.py push resnets
    python kaggle/launch.py status resnets
    python kaggle/launch.py fetch resnets
"""
import json
import os
import shutil
import subprocess
import sys

USER = "aadigoyal42"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

RESNET = ["--epochs", "80", "--bs", "128", "--opt", "sgd", "--lr", "0.1", "--wd", "5e-4", "--warmup", "2"]
VIT = ["--epochs", "100", "--bs", "256", "--opt", "adamw", "--lr", "1e-3", "--wd", "0.05", "--warmup", "5",
       "--label-smoothing", "0.1", "--mix", "--clip", "1.0"]
JOBS = {
    "resnets": {"resnet18": ["--model", "resnet18", *RESNET], "ee_resnet18": ["--model", "ee_resnet18", *RESNET]},
    "vit": {"ee_vit_tiny": ["--model", "ee_vit_tiny", *VIT]},
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
        script = f.read().replace("JOBS = {}", "JOBS = " + json.dumps(JOBS[job]))
    with open(os.path.join(d, "run.py"), "w") as f:
        f.write(script)
    meta = {"id": slug(job), "title": slug(job).split("/")[1], "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [], "kernel_sources": [], "competition_sources": [],
            "machine_shape": "NvidiaTeslaT4"}
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
