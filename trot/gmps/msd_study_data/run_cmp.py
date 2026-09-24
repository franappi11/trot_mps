"""Fixed vs re-drawn MSD trial across system sizes.

Per L: sampled_msd_cpmc.py (fixed trial) and resampled_msd_cpmc.py (redrawn every
step) run side by side with the same threads, seeds and CPMC settings, each
appending one JSON record. Restartable: (L, method) already in results.jsonl are
skipped.
"""
import json, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
PY = os.path.expanduser("~/.trot/bin/python")
G = "/Users/fnappi/trot_mps/trot/gmps"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = f"{HERE}/results.jsonl"
LS = [int(x) for x in sys.argv[1:]] or [8, 16, 24, 32, 48]
COMMON = dict(CHI=8, N_SAMPLES=5000, N_WALKERS=200, N_EQL=100, N_BLOCKS=200, N_PROP=20,
              DT=0.01, SEED=1234, MEM_BUDGET_GB=6.0)
env = dict(os.environ, PYTHONPATH=G, OMP_NUM_THREADS="5", VECLIB_MAXIMUM_THREADS="5",
           XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=5")

def done():
    if not os.path.exists(RES):
        return set()
    return {(r["L"], r["method"]) for r in map(json.loads, open(RES)) }

def fixed(L):
    s = open(f"{G}/sampled_msd_cpmc.py").read()
    s = re.sub(r"(?m)^L, n_up, n_down = .*$", f"L, n_up, n_down = {L}, {L//2}, {L//2}", s, count=1)
    for k, v in dict(COMMON, SAMPLE_SEED=1, COEFF="is", TABLE_MSD=True,
                     RESULT_JSON=f"{HERE}/_raw_fixed.jsonl", TAG=f"fixed_L{L}").items():
        s, n = re.subn(rf"(?m)^{k}(\s*)=.*$", f"{k} = {v!r}", s, count=1); assert n == 1, k
    p = f"{HERE}/_fixed_L{L}.py"; open(p, "w").write(s)
    return [PY, p]

def resampled(L):
    kv = dict(COMMON, L=L, TRIAL_SEED=1, RESAMPLE_EVERY=1, TABLE_MSD=True,
              RESULT_JSON=f"{HERE}/_raw_resampled.jsonl", TAG=f"resampled_L{L}")
    return [PY, f"{G}/resampled_msd_cpmc.py"] + [f"{k}={v!r}" for k, v in kv.items()]

def run(L, method):
    cmd = (fixed if method == "fixed" else resampled)(L)
    t0 = time.time()
    with open(f"{HERE}/{method}_L{L}.log", "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    wall = time.time() - t0
    raw = f"{HERE}/_raw_{method}.jsonl"
    rec = [r for r in map(json.loads, open(raw))][-1] if rc == 0 and os.path.exists(raw) else {}
    rec = dict(L=L, method=method, rc=rc, wall_total=wall,
               e_cpmc=rec.get("e_cpmc"), err_cpmc=rec.get("err_cpmc"), t_run=rec.get("t_run"),
               e_dmrg_trial=rec.get("e_dmrg_trial"), raw=rec)
    with open(RES, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"[{time.strftime('%H:%M:%S')}] L={L:2d} {method:9s} rc={rc} wall {wall:7.0f}s  "
          f"E = {rec['e_cpmc']} +- {rec['err_cpmc']}", flush=True)

for L in LS:
    todo = [m for m in ("fixed", "resampled") if (L, m) not in done()]
    with ThreadPoolExecutor(2) as ex:
        list(ex.map(lambda m: run(L, m), todo))
print("ALL DONE", flush=True)
