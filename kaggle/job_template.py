"""Kaggle job: clone the repo, train each model on its own GPU in parallel, then evaluate one by one."""
import os
import subprocess
import sys
import threading

REPO = "https://github.com/goyaladitya05/adaptive-dnn-inference.git"
SRC, DATA, OUT = "/tmp/repo", "/tmp/data", "/kaggle/working"
JOBS = {}


def run(cmd, **kw):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def pump(name, proc, path):
    """Copy a child's output to its log file and to the kernel log with a prefix."""
    with open(path, "w") as f:
        for line in proc.stdout:
            f.write(line)
            f.flush()
            print(f"[{name}] {line}", end="", flush=True)


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
    cmd = [sys.executable, "-m", "src.train", "--out", out, "--data", DATA, *args]
    print("$", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, env={**env, "CUDA_VISIBLE_DEVICES": str(gpu)}, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    t = threading.Thread(target=pump, args=(name, p, os.path.join(out, "train.log")))
    t.start()
    procs.append((name, p, t))
failed = []
for name, p, t in procs:
    code = p.wait()
    t.join()
    print(name, "train exit", code, flush=True)
    if code:
        failed.append(name)

for name, _, _ in procs:
    if name not in failed:
        run([sys.executable, "-m", "src.evaluate", "--run", os.path.join(OUT, name), "--data", DATA],
            env={**env, "CUDA_VISIBLE_DEVICES": "0"})
if failed:
    sys.exit(f"training failed: {failed}")
