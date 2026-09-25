"""Phase A: the sampling grid.  Phase B: the MPS grid re-run at trial chi = 16.

Sequential, one subprocess per run, each appending its own JSON record the
moment it finishes. Restartable: grid points already in study_results.jsonl
(matched on kind, L, chi_trial and the swept knob) are skipped.
"""
import json, os, re, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.expanduser("~/.trot/bin/python")
RESULTS = os.path.join(HERE, "study_results.jsonl")
LOGDIR = os.path.join(HERE, "study_logs")
os.makedirs(LOGDIR, exist_ok=True)

LS = [16, 24, 32, 48]
N_SAMPLES = [5000, 10000, 20000]
CHI_WALKERS = [4, 6]
COMMON = dict(N_WALKERS=200, N_EQL=50, N_BLOCKS=200, N_PROP=20)


def done_keys():
    keys = set()
    if os.path.exists(RESULTS):
        for l in open(RESULTS):
            if l.strip():
                r = json.loads(l)
                keys.add((r.get("kind"), r.get("L"), r.get("chi_trial"),
                          r.get("chi_walker"), r.get("n_samples")))
    return keys


def launch(script, tag, L, **cfg):
    src = open(os.path.join(HERE, script)).read()
    src = re.sub(r"(?m)^L, n_up, n_down = .*$",
                 f"L, n_up, n_down = {L}, {L//2}, {L//2}", src, count=1)
    for k, v in cfg.items():
        src, n = re.subn(rf"(?m)^{k}(\s*)=.*$",
                         lambda m: f"{k}{m.group(1)}= {v!r}", src, count=1)
        assert n == 1, f"could not set {k}"
    tmp = os.path.join(LOGDIR, f"_{tag}.py")
    open(tmp, "w").write(src)
    t0 = time.time()
    with open(os.path.join(LOGDIR, f"{tag}.log"), "w") as log:
        r = subprocess.run([PY, tmp], stdout=log, stderr=subprocess.STDOUT)
    print(f"[{time.strftime('%H:%M:%S')}] {tag:24s} rc={r.returncode} "
          f"{time.time()-t0:7.0f}s", flush=True)


print("=== PHASE A: sampling grid (trial chi = 16) ===", flush=True)
for L in LS:
    for ns in N_SAMPLES:
        if ("sampling", L, 16, None, ns) in done_keys():
            print(f"[skip] sampling L={L} N={ns}", flush=True); continue
        launch("sampled_msd_cpmc.py", f"smp16_L{L}_N{ns}", L,
               N_SAMPLES=ns, CHI=16, RESULT_JSON=RESULTS,
               TAG=f"smp16_L{L}_N{ns}", **COMMON)

print("=== PHASE B: MPS grid, trial chi = 16, chi_walker in 4,6 ===", flush=True)
for L in LS:
    for cw in CHI_WALKERS:
        if ("mps", L, 16, cw, None) in done_keys():
            print(f"[skip] mps L={L} chi_w={cw} chi_trial=16", flush=True); continue
        launch("mps_cpmc.py", f"mps16_L{L}_cw{cw}", L,
               CHI=16, CHI_WALKER=cw, PLAN_B="adaptive", RESULT_JSON=RESULTS,
               TAG=f"mps16_L{L}_cw{cw}", **COMMON)

print("ALL DONE", flush=True)
