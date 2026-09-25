"""L=8, chi=4 trial: 10 seeds for each N in (1000, 2000, 5000), fixed and resampled.

Seed index s shifts both the CPMC seed (1234 + s) and the sampling seed (1 + s); s = 0 are
the runs run_cmp3.py already made. Three jobs at a time, 2 threads each. Restartable:
(L, N, chi, method, seed) already in results.jsonl are skipped.
"""
import json, os, re, subprocess, time
from concurrent.futures import ThreadPoolExecutor
PY = os.path.expanduser("~/.trot/bin/python")
G = "/Users/fnappi/trot_mps/trot/gmps"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = f"{HERE}/results.jsonl"
L, CHI, NS, SEEDS = 8, 4, [1000, 2000, 5000], range(1, 10)
COMMON = dict(CHI=CHI, N_WALKERS=200, N_EQL=100, N_BLOCKS=200, N_PROP=20, DT=0.01, MEM_BUDGET_GB=6.0)
env = dict(os.environ, PYTHONPATH=G, OMP_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2",
           XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=2")

def done():
    return {(r["L"], r.get("N", 5000), r.get("chi", 8), r["method"], r.get("seed", 0))
            for r in map(json.loads, open(RES))} if os.path.exists(RES) else set()

def last_raw(method, tag):
    path = f"{HERE}/_raw_{method}.jsonl"
    rows = [r for r in map(json.loads, open(path)) if r.get("tag") == tag] if os.path.exists(path) else []
    return rows[-1] if rows else {}

def cmd(method, N, s, tag):
    if method == "fixed":
        src = open(f"{G}/sampled_msd_cpmc.py").read()
        src = re.sub(r"(?m)^L, n_up, n_down = .*$", f"L, n_up, n_down = {L}, {L//2}, {L//2}", src, count=1)
        for k, v in dict(COMMON, SEED=1234 + s, N_SAMPLES=N, SAMPLE_SEED=1 + s, COEFF="is", TABLE_MSD=True,
                         RESULT_JSON=f"{HERE}/_raw_fixed.jsonl", TAG=tag).items():
            src, n = re.subn(rf"(?m)^{k}(\s*)=.*$", f"{k} = {v!r}", src, count=1); assert n == 1, k
        p = f"{HERE}/_{tag}.py"; open(p, "w").write(src)
        return [PY, p]
    kv = dict(COMMON, L=L, SEED=1234 + s, N_SAMPLES=N, TRIAL_SEED=1 + s, RESAMPLE_EVERY=1, TABLE_MSD=True,
              RESULT_JSON=f"{HERE}/_raw_resampled.jsonl", TAG=tag)
    return [PY, f"{G}/resampled_msd_cpmc.py"] + [f"{k}={v!r}" for k, v in kv.items()]

def run(job):
    method, N, s = job
    tag = f"{method}_L{L}_chi{CHI}_N{N}_s{s}"
    t0 = time.time()
    with open(f"{HERE}/{tag}.log", "w") as fh:
        rc = subprocess.run(cmd(method, N, s, tag), stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    raw = last_raw(method, tag) if rc == 0 else {}
    rec = dict(L=L, N=N, chi=CHI, method=method, seed=s, rc=rc, wall_total=time.time() - t0,
               e_cpmc=raw.get("e_cpmc"), err_cpmc=raw.get("err_cpmc"), t_run=raw.get("t_run"),
               e_dmrg_trial=raw.get("e_dmrg_trial"), raw=raw)
    with open(RES, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"[{time.strftime('%H:%M:%S')}] N={N:5d} {method:9s} seed {s} rc={rc}  E = {rec['e_cpmc']} +- {rec['err_cpmc']}", flush=True)

jobs = [(m, N, s) for s in SEEDS for N in NS for m in ("fixed", "resampled")
        if (L, N, CHI, m, s) not in done()]
print(f"{len(jobs)} runs to do", flush=True)
with ThreadPoolExecutor(3) as ex:
    list(ex.map(run, jobs))
print("ALL DONE", flush=True)
