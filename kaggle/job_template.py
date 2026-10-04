"""Kaggle job: clone the repo, then run one queue of steps per GPU in parallel."""
import os
import subprocess
import sys
import threading

REPO = "https://github.com/goyaladitya05/adaptive-dnn-inference.git"
SRC, DATA, OUT = "/tmp/repo", "/tmp/data", "/kaggle/working"
QUEUES = []


def run(cmd, **kw):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def step(name, module, args, gpu, log):
    """Run one module, copying its output to a log file and to the kernel log with a prefix."""
    cmd = [sys.executable, "-m", module, *args]
    print(f"[{name}] $", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, env={**env, "CUDA_VISIBLE_DEVICES": str(gpu)}, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    with open(log, "a") as f:
        for line in p.stdout:
            f.write(line)
            f.flush()
            print(f"[{name}] {line}", end="", flush=True)
    return p.wait()


def worker(gpu, queue, failed):
    for item in queue:
        out = os.path.join(OUT, item["name"])
        os.makedirs(out, exist_ok=True)
        for module, *args in item["steps"]:
            args = [a.format(out=out, data=DATA, work=OUT) for a in args]
            if step(item["name"], module, args, gpu, os.path.join(out, "log.txt")):
                failed.append(item["name"])
                break


run(["git", "clone", "--depth", "1", REPO, SRC])
os.chdir(SRC)
env = {**os.environ, "PYTHONUNBUFFERED": "1",
       "GIT_COMMIT": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
run(["nvidia-smi"])
run([sys.executable, "-m", "src.data", "--root", DATA], env=env)

failed = []
threads = [threading.Thread(target=worker, args=(gpu, q, failed)) for gpu, q in enumerate(QUEUES)]
for t in threads:
    t.start()
for t in threads:
    t.join()
if failed:
    sys.exit(f"failed: {failed}")
