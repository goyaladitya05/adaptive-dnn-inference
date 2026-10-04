"""Kaggle job: clone the repo, train each model on its own GPU in parallel, then evaluate one by one."""
import os
import subprocess
import sys

REPO = "https://github.com/goyaladitya05/adaptive-dnn-inference.git"
SRC, DATA, OUT = "/tmp/repo", "/tmp/data", "/kaggle/working"
JOBS = {}


def run(cmd, **kw):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


run(["git", "clone", "--depth", "1", REPO, SRC])
os.chdir(SRC)
env = {**os.environ, "PYTHONUNBUFFERED": "1",
       "GIT_COMMIT": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
run(["nvidia-smi"])
run([sys.executable, "-m", "src.data", "--root", DATA], env=env)

procs = []
for gpu, (name, args) in enumerate(JOBS.items()):
    out = os.path.join(OUT, name)
    os.makedirs(out, exist_ok=True)
    log = open(os.path.join(out, "train.log"), "w")
    cmd = [sys.executable, "-m", "src.train", "--out", out, "--data", DATA, *args]
    print("$", " ".join(cmd), flush=True)
    procs.append((name, subprocess.Popen(cmd, env={**env, "CUDA_VISIBLE_DEVICES": str(gpu)},
                                         stdout=log, stderr=subprocess.STDOUT)))
failed = []
for name, p in procs:
    code = p.wait()
    print(name, "train exit", code, flush=True)
    with open(os.path.join(OUT, name, "train.log")) as f:
        print("".join(f.readlines()[-3:]), flush=True)
    if code:
        failed.append(name)

for name, _ in procs:
    if name not in failed:
        run([sys.executable, "-m", "src.evaluate", "--run", os.path.join(OUT, name), "--data", DATA],
            env={**env, "CUDA_VISIBLE_DEVICES": "0"})
if failed:
    sys.exit(f"training failed: {failed}")
