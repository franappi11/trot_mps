"""Continuation of run_cmp.py: no L=48; instead L=32 at N = 10000 and 20000.

Waits for the L=32, N=5000 pair still running (PIDs on the command line), records
it from the scripts' own JSON output, then runs each N as a fixed / resampled pair
side by side with the same threads, seeds and CPMC settings as before.
Restartable: (L, N, method) already in results.jsonl are skipped.
"""
import json, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
PY = os.path.expanduser("~/.trot/bin/python")
G = "/Users/fnappi/trot_mps/trot/gmps"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = f"{HERE}/results.jsonl"
L, NS = 32, [10000, 20000]
COMMON = dict(CHI=8, N_WALKERS=200, N_EQL=100, N_BLOCKS=200, N_PROP=20,
              DT=0.01, SEED=1234, MEM_BUDGET_GB=6.0)
env = dict(os.environ, PYTHONPATH=G, OMP_NUM_THREADS="5", VECLIB_MAXIMUM_THREADS="5",
           XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=5")

def done():
    if not os.path.exists(RES):
        return set()
    return {(r["L"], r.get("N", 5000), r["method"]) for r in map(json.loads, open(RES))}

def record(L, N, method, rc, wall, raw):
    rec = dict(L=L, N=N, method=method, rc=rc, wall_total=wall,
               e_cpmc=raw.get("e_cpmc"), err_cpmc=raw.get("err_cpmc"), t_run=raw.get("t_run"),
               e_dmrg_trial=raw.get("e_dmrg_trial"), raw=raw)
    with open(RES, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"[{time.strftime('%H:%M:%S')}] L={L:2d} N={N:5d} {method:9s} rc={rc} wall "
          f"{wall if wall is None else round(wall)}s  t_run {raw.get('t_run')}  "
          f"E = {rec['e_cpmc']} +- {rec['err_cpmc']}", flush=True)

def last_raw(method, tag):
    path = f"{HERE}/_raw_{method}.jsonl"
    rows = [r for r in map(json.loads, open(path)) if r.get("tag") == tag] if os.path.exists(path) else []
    return rows[-1] if rows else {}

def fixed(N, tag):
    s = open(f"{G}/sampled_msd_cpmc.py").read()
    s = re.sub(r"(?m)^L, n_up, n_down = .*$", f"L, n_up, n_down = {L}, {L//2}, {L//2}", s, count=1)
    for k, v in dict(COMMON, N_SAMPLES=N, SAMPLE_SEED=1, COEFF="is", TABLE_MSD=True,
                     RESULT_JSON=f"{HERE}/_raw_fixed.jsonl", TAG=tag).items():
        s, n = re.subn(rf"(?m)^{k}(\s*)=.*$", f"{k} = {v!r}", s, count=1); assert n == 1, k
    p = f"{HERE}/_{tag}.py"; open(p, "w").write(s)
    return [PY, p]

def resampled(N, tag):
    kv = dict(COMMON, L=L, N_SAMPLES=N, TRIAL_SEED=1, RESAMPLE_EVERY=1, TABLE_MSD=True,
              RESULT_JSON=f"{HERE}/_raw_resampled.jsonl", TAG=tag)
    return [PY, f"{G}/resampled_msd_cpmc.py"] + [f"{k}={v!r}" for k, v in kv.items()]

def run(N, method):
    tag = f"{method}_L{L}_N{N}"
    cmd = (fixed if method == "fixed" else resampled)(N, tag)
    t0 = time.time()
    with open(f"{HERE}/{tag}.log", "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    record(L, N, method, rc, time.time() - t0, last_raw(method, tag) if rc == 0 else {})

# 1. the N=5000 pair started by run_cmp.py
wait = [int(p) for p in sys.argv[1:]]
def alive(pid):
    try:
        os.kill(pid, 0); return True
    except ProcessLookupError:
        return False
while any(alive(p) for p in wait):
    time.sleep(30)
for method in ("fixed", "resampled"):
    if (L, 5000, method) not in done():
        raw = last_raw(method, f"{method}_L{L}")
        record(L, 5000, method, 0 if raw else 1, None, raw)

# 2. the new sample counts, one fixed / resampled pair at a time
for N in NS:
    todo = [m for m in ("fixed", "resampled") if (L, N, m) not in done()]
    with ThreadPoolExecutor(2) as ex:
        list(ex.map(lambda m: run(N, m), todo))
print("ALL DONE", flush=True)
