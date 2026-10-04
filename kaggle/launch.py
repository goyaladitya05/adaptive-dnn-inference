"""Build and push a Kaggle GPU kernel for one job, or fetch its outputs.

    python kaggle/launch.py push study-c --account other_user   # credentials in ~/.kaggle/accounts/<name>.json
    python kaggle/launch.py status study-c
    python kaggle/launch.py fetch study-c
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
STRONG = ["--label-smoothing", "0.1", "--mix", "--randaug"]
VIT_PLAIN = [a for a in VIT if a not in ("--label-smoothing", "0.1", "--mix", "--randaug")]


def kd(alpha):
    return ["--distill", str(alpha), "--kd-temp", "3"]


def run(name, model, args, seed, corrupt_val=False):
    return {"name": f"{name}_s{seed}", "steps": [
        ["src.train", "--model", model, "--seed", str(seed), "--out", "{out}", "--data", "{data}", *args],
        ["src.evaluate", "--run", "{out}", "--data", "{data}", "--no-latency", *(["--corrupt-val"] if corrupt_val else [])],
    ]}


def study(name, model, args, seed):
    return run(name, model, args, seed, corrupt_val=True)


def reeval(job, name):
    """Corrupted validation logits for a run trained in an earlier job of the same account."""
    return {"name": f"reeval_{name}", "steps": [
        ["src.evaluate", "--run", f"/kaggle/input/notebooks/{USER}/adaptive-inference-{job}/{name}", "--out", "{out}",
         "--data", "{data}", "--corrupt-val-only"]]}


EXISTING = [("final-b", "ee_resnet18_kd_s0"), ("final-a", "ee_resnet18_kd_s1"), ("final-b", "ee_resnet18_kd_s2"),
            ("final-b", "ee_resnet18_ce_s0"), ("final-a", "ee_resnet18_pool_s0"), ("final-a", "ee_vit_tiny_s0"),
            ("final-b", "ee_vit_tiny_s1"), ("final-a", "ee_vit_tiny_s2")]


def strong(s):
    return study("ee_resnet18_strong", "ee_resnet18", RESNET + STRONG + KD, s)


def vit_plain(s):
    return study("ee_vit_tiny_plain", "ee_vit_tiny", VIT_PLAIN + KD, s)


def vit_ce(s):
    return study("ee_vit_tiny_ce", "ee_vit_tiny", VIT, s)


def ce(s):
    return study("ee_resnet18_ce", "ee_resnet18", RESNET, s)


def kd_alpha(alpha, s):
    return study(f"ee_resnet18_kd{int(alpha * 100)}", "ee_resnet18", RESNET + kd(alpha), s)


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
    "study-a": {"queues": [[reeval(j, n) for j, n in EXISTING] + [strong(0), vit_ce(0)],
                           [vit_plain(0), strong(1), ce(1)]],
                "sources": ["final-a", "final-b"]},
    "study-b": {"queues": [[vit_plain(1), vit_ce(1), strong(2)], [vit_plain(2), vit_ce(2), ce(2)]]},
    "study-c": {"queues": [[kd_alpha(0.25, 0), kd_alpha(0.75, 0)],
                           [kd_alpha(0.25, 1), kd_alpha(0.75, 1), study("ee_resnet18_ls", "ee_resnet18",
                                                                        RESNET + ["--label-smoothing", "0.1"] + KD, 0)]]},
    "study-d": {"queues": [[kd_alpha(0.25, 2), kd_alpha(0.75, 2)],
                           [study("ee_resnet18_mix", "ee_resnet18", RESNET + ["--mix"] + KD, 0),
                            study("ee_resnet18_ra", "ee_resnet18", RESNET + ["--randaug"] + KD, 0)]]},
    "latency": {"queues": [[{"name": "latency", "steps": [
        ["src.latency", "--runs", "/kaggle/input", "--only", "_s0$", "--out", "{work}",
         "--data", "{data}"]]}]],
        "sources": ["final-a", "final-b"]},
}


def account_env(account):
    if not account:
        return dict(os.environ)
    with open(os.path.expanduser(f"~/.kaggle/accounts/{account}.json")) as f:
        cred = json.load(f)
    return {**os.environ, "KAGGLE_USERNAME": cred["username"], "KAGGLE_KEY": cred["key"]}


def kaggle(*args, account=None):
    exe = shutil.which("kaggle") or os.path.join(os.path.dirname(sys.executable), "kaggle")
    subprocess.run([exe, *args], check=True, env=account_env(account))


def job_dir(job):
    return os.path.join(ROOT, "runs", "kaggle", job)


def owner(job, account=None):
    """Account and kernel id of a job; without an explicit account, the one used by its last push."""
    if account is None:
        f = os.path.join(job_dir(job), "account.txt")
        account = open(f).read().strip() if os.path.exists(f) else ""
    account = account or None
    user = account_env(account)["KAGGLE_USERNAME"] if account else USER
    return account, f"{user}/adaptive-inference-{job}"


def slug(job):
    return owner(job)[1]


def push(job, account=None):
    account, kid = owner(job, account or "")
    d = job_dir(job)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "account.txt"), "w") as f:
        f.write(account or "")
    with open(os.path.join(HERE, "job_template.py")) as f:
        script = f.read().replace("QUEUES = []", "QUEUES = " + json.dumps(JOBS[job]["queues"], indent=1))
    with open(os.path.join(d, "run.py"), "w") as f:
        f.write(script)
    meta = {"id": kid, "title": kid.split("/")[1], "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [], "competition_sources": [],
            "kernel_sources": [slug(s) for s in JOBS[job].get("sources", [])], "machine_shape": "NvidiaTeslaT4"}
    with open(os.path.join(d, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=1)
    kaggle("kernels", "push", "-p", d, account=account)


def fetch(job, skip_checkpoints=True):
    account, kid = owner(job)
    d = os.path.join(ROOT, "runs", job)
    os.makedirs(d, exist_ok=True)
    pattern = ["--file-pattern", r".*\.(npy|json|txt|log)$"] if skip_checkpoints else []
    kaggle("kernels", "output", kid, "-p", d, "-o", *pattern, account=account)


def status(job):
    account, kid = owner(job)
    kaggle("kernels", "status", kid, account=account)


if __name__ == "__main__":
    action, job = sys.argv[1], sys.argv[2]
    acct = sys.argv[sys.argv.index("--account") + 1] if "--account" in sys.argv else None
    if action == "push":
        push(job, acct)
    else:
        {"fetch": fetch, "status": status}[action](job)
