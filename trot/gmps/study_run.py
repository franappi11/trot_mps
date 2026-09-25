"""Sweep both routes over system size and their respective accuracy knob.

Every run is a fresh subprocess (clean JAX state, isolated memory) and appends
its own JSON record to RESULTS the moment it finishes, so partial results
survive an interrupt or an OOM.

    MPS route      : L x CHI_WALKER (per channel), adaptive B, trial chi = 64
    sampling route : L x N_SAMPLES
    ground truth   : DMRG at chi = 200 for each L
"""
import json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.expanduser("~/.trot/bin/python")
RESULTS = os.path.join(HERE, "study_results.jsonl")
LOGDIR = os.path.join(HERE, "study_logs")
os.makedirs(LOGDIR, exist_ok=True)

LS = [16, 24, 32, 48]
CHI_WALKERS = [2, 4, 6]
N_SAMPLES = [4000, 8000, 12000, 16000, 20000]
COMMON = dict(N_WALKERS=200, N_EQL=50, N_BLOCKS=200, N_PROP=20, CHI=64)


def edit(src, **kw):
    for k, v in kw.items():
        src, n = re.subn(rf"(?m)^{k}(\s*)=.*$", lambda m: f"{k}{m.group(1)}= {v!r}", src, count=1)
        assert n == 1, f"could not set {k}"
    return src


def launch(script, tag, **cfg):
    src = open(os.path.join(HERE, script)).read()
    if "L" in cfg:                      # L, n_up, n_down live on one line
        L = cfg.pop("L")
        src = re.sub(r"(?m)^L, n_up, n_down = .*$",
                     f"L, n_up, n_down = {L}, {L//2}, {L//2}", src, count=1)
    src = edit(src, RESULT_JSON=RESULTS, TAG=tag, **cfg)
    tmp = os.path.join(LOGDIR, f"_{tag}.py")
    open(tmp, "w").write(src)
    t0 = time.time()
    with open(os.path.join(LOGDIR, f"{tag}.log"), "w") as log:
        r = subprocess.run([PY, tmp], stdout=log, stderr=subprocess.STDOUT)
    print(f"[{time.strftime('%H:%M:%S')}] {tag:28s} rc={r.returncode} "
          f"{time.time()-t0:7.0f}s", flush=True)


# ---------------------------------------------------- 1. DMRG ground truth
REF = os.path.join(HERE, "study_dmrg_ref.jsonl")
ref_src = '''
import json, time, sys
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
sys.argv = ["x"]
exec(open("%s/_dmrg_ref_body.py").read())
''' % HERE

for L in LS:
    tag = f"dmrgref_L{L}"
    if os.path.exists(REF) and any(json.loads(l).get("L") == L
                                   for l in open(REF) if l.strip()):
        print(f"[skip] {tag} already done", flush=True)
        continue
    src = open(os.path.join(HERE, "mps_cpmc.py")).read()
    src = re.sub(r"(?m)^L, n_up, n_down = .*$",
                 f"L, n_up, n_down = {L}, {L//2}, {L//2}", src, count=1)
    # stop right after the trial is built; dump both chi=64 and chi=200 energies
    marker = "e_T_mps = float(mps_overlap(Hket_T, ket_T) / mps_overlap(ket_T, ket_T))"
    cut = src.index(marker) + len(marker)
    body = src[:cut] + f'''
import json as _json, time as _time
_t0 = _time.time()
_h200 = build_hamil(L, U)
_m200, _e200 = run_dmrg(_h200, 200, n_sweeps=DMRG_SWEEPS, seed=DMRG_SEED)
_t200 = _time.time() - _t0
_k200 = [jnp.asarray(x) for x in densify(_m200, L)]
_H200 = [jnp.asarray(x) for x in compress(apply_mpo(hubbard_mpo(L, t, U),
                                                    densify(_m200, L)))]
_e200_var = float(mps_overlap(_H200, _k200) / mps_overlap(_k200, _k200))
_rec = dict(L=L, n_up=n_up, U=U, chi_trial=CHI,
            e_trial_chi64=float(mps_overlap(Hket_T, ket_T) / mps_overlap(ket_T, ket_T)),
            bond_dims_chi64=[int(x.shape[0]) for x in ket_T],
            e_dmrg_chi200=_e200_var, e_dmrg_chi200_dav=float(_e200),
            bond_dims_chi200=[int(x.shape[0]) for x in _k200],
            t_dmrg200=_t200, e_hf=float(e_hf))
with open("{REF}", "a") as fh:
    fh.write(_json.dumps(_rec) + "\\n"); fh.flush()
print("ref saved", _rec["e_trial_chi64"], _rec["e_dmrg_chi200"])
'''
    tmp = os.path.join(LOGDIR, f"_{tag}.py")
    open(tmp, "w").write(body)
    t0 = time.time()
    with open(os.path.join(LOGDIR, f"{tag}.log"), "w") as log:
        r = subprocess.run([PY, tmp], stdout=log, stderr=subprocess.STDOUT)
    print(f"[{time.strftime('%H:%M:%S')}] {tag:28s} rc={r.returncode} "
          f"{time.time()-t0:7.0f}s", flush=True)

# ---------------------------------------------------- 2. MPS route sweep
done = set()
if os.path.exists(RESULTS):
    for l in open(RESULTS):
        if l.strip():
            r = json.loads(l)
            done.add((r.get("kind"), r.get("L"), r.get("chi_walker"), r.get("n_samples")))

for L in LS:
    for cw in CHI_WALKERS:
        if ("mps", L, cw, None) in done:
            print(f"[skip] mps L={L} chi_w={cw}", flush=True); continue
        launch("mps_cpmc.py", f"mps_L{L}_cw{cw}", L=L, CHI_WALKER=cw,
               PLAN_B="adaptive", **COMMON)

# ---------------------------------------------------- 3. sampling route sweep
for L in LS:
    for ns in N_SAMPLES:
        if ("sampling", L, None, ns) in done:
            print(f"[skip] sampling L={L} N={ns}", flush=True); continue
        launch("sampled_msd_cpmc.py", f"smp_L{L}_N{ns}", L=L, N_SAMPLES=ns, **COMMON)

print("ALL DONE", flush=True)
