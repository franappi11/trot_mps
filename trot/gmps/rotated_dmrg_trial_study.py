"""Does rotating a spin-broken trial remove the CPMC bias? Small-system study on the Hubbard chain.

Every trial (pyblock3 DMRG at the given bond dimensions; with --uhf also the pyscf UHF determinant) runs
twice with the same seed: as it is ("plain") and rotated by R_y(pi/2), i.e. projected onto the walkers'
S_z = 0 sector after the rotation ("rotated", make_mps_trial(rotate_spin(T, R))). The rotation removes
every odd-S component of an S_z = 0 trial and scales S = 2 by 1/2, S = 4 by 3/8 (Legendre P_S(0)).
The reference energy is DMRG at --chi-ref (exact for these sizes).

One JSON line per run goes to --out (trial diagnostics, CPMC mean and error, bias, node encounters), and
every block (equilibration included) to <out stem>_blocks.jsonl.

    ~/.trot/bin/python trot/gmps/rotated_dmrg_trial_study.py --L 8 --U 8 --chi 2 4 6 --uhf --out study.jsonl
    ~/.trot/bin/python trot/gmps/rotated_dmrg_trial_study.py --plot study.jsonl
    ~/.trot/bin/python trot/gmps/rotated_dmrg_trial_study.py --self-test

Keep --eql 5k and --blocks 10k with the same k (default 40 and 80), so run_qmc compiles one block size.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import json
import tempfile
import time
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss, legval

R90 = np.array([[1.0, -1.0], [1.0, 1.0]]) / np.sqrt(2.0)  # R_y(pi/2): spin moments along z go to x


def ry(beta):
    """Spin rotation by beta about y (column sigma = image of |sigma>, as trot.trial.mps.spin_rotation_unitary)."""
    c, s = np.cos(beta / 2), np.sin(beta / 2)
    return np.array([[c, -s], [s, c]])


def mps_overlap(a, b):
    env = np.ones((1, 1))
    for A, B in zip(a, b):
        env = np.einsum("ab,apc,bpd->cd", env, np.asarray(A), np.asarray(B), optimize=True)
    return float(env.reshape(()))


def spin_distribution(tensors, n_points=None):
    """Weights p_S of total spin S in an S_z = 0 MPS (normalisation irrelevant).

    f(beta) = <T|U(R_y(beta))|T> / <T|T> = sum_S p_S P_S(cos beta), because <S,0|R_y(beta)|S,0> is the
    Legendre polynomial P_S(cos beta). Gauss-Legendre quadrature in cos(beta) with n_points > S_max nodes
    is exact: p_S = (2S + 1)/2 int P_S(x) f(x) dx.
    """
    from trot.trial.mps import rotate_spin

    n = n_points or len(tensors) + 2
    x, weights = leggauss(n)
    norm = mps_overlap(tensors, tensors)
    f = np.array([mps_overlap(tensors, rotate_spin(tensors, ry(np.arccos(xk)))) for xk in x]) / norm
    return np.array(
        [(2 * s + 1) / 2 * np.sum(weights * legval(x, np.eye(n)[s]) * f) for s in range(n)]
    )


def diagnostics(tensors, h1, u, e_ref):
    """Variational energy and spin content of a trial MPS, and what the 90-degree rotation does to it."""
    from trot.meas.mps import apply_mpo, hubbard_mpo_from_h1

    T = [np.asarray(A) for A in tensors]
    e_var = mps_overlap(T, apply_mpo(hubbard_mpo_from_h1(h1, u), T)) / mps_overlap(T, T)
    p = spin_distribution(T)
    S = np.arange(len(p))
    w90 = float(np.sum(p * np.array([legval(0.0, np.eye(len(p))[s]) for s in S]) ** 2))
    return dict(
        e_var=e_var,
        e_var_error=e_var - e_ref,
        p_S=[round(float(v), 6) for v in p[:8]],
        p0=float(p[0]),
        s_s1=float(np.sum(p * S * (S + 1))),
        w90=w90,
        singlet_fraction_after_rotation=float(p[0] / w90),
    )


def domain_walls(rdm1, threshold=0.03):
    """Domain walls of the staggered magnetisation m_i = (-1)^i <S^z_i> of a trial (from its rdm1).

    Sign changes of m_i are counted between consecutive sites where |m_i| > threshold, so noise on an
    unpolarised (singlet-like) trial does not count. Returns (walls, max |m_i|, profile)."""
    g = np.asarray(rdm1)
    L = g.shape[1]
    m = (-1.0) ** np.arange(L) * 0.5 * (np.diag(g[0]) - np.diag(g[1]))
    signs = np.sign(m[np.abs(m) > threshold])
    return int(np.sum(signs[1:] != signs[:-1])), float(np.abs(m).max()), m


def hubbard_chain(L, u):
    import jax.numpy as jnp

    from trot.core.system import System
    from trot.ham.hubbard import HamHubbard, hopping_matrix

    h1 = hopping_matrix(L, 1.0)
    return h1, HamHubbard(h1=jnp.asarray(h1), u=u), System(L, (L // 2, L // 2), "unrestricted")


def dmrg(ham, sys_, chi, n_sweeps=14, seed=0):
    from trot.gmps.dmrg import make_dmrg_trial

    with contextlib.redirect_stdout(io.StringIO()):
        return make_dmrg_trial(ham, sys_, chi=chi, n_sweeps=n_sweeps, seed=seed)


def warm_dmrg(ham, sys_, chi, warm_chi, n_sweeps=12, seed=0, warm_sweeps=30):
    """DMRG at chi started from the DMRG state at warm_chi compressed to chi (an MpsTrial).

    Random-start DMRG at small chi gets stuck in states with domain walls of the staggered
    magnetisation (localised spinons), and the energy grows linearly with their number. A
    wall-free state at larger chi, compressed to chi, starts the sweeps in the uniform basin:
    at L=64, chi=16 random start has 4 walls and E - E_ref = 0.29; this gives 0 walls and 0.013.
    The source must itself be wall-free: at L=100 a chi=32 start needs warm_sweeps=30 (14 sweeps
    leave walls, and so do chi=64/128 starts with 14 sweeps).
    """
    from pyblock3.algebra.mpe import MPE

    from trot.gmps.dmrg import hubbard_pyblock3_mpo, make_pyblock3_hamiltonian
    from trot.meas.mps import hubbard_h1
    from trot.trial.mps import mps_trial_from_pyblock3

    h1, u = hubbard_h1(ham), float(ham.u)
    nelec = tuple(int(n) for n in sys_.nelec)
    hamiltonian = make_pyblock3_hamiltonian(h1, nelec)
    mpo = hubbard_pyblock3_mpo(hamiltonian, h1, u)
    source = dmrg(ham, sys_, warm_chi, n_sweeps=warm_sweeps, seed=seed)
    walls, _, _ = domain_walls(source.trial.rdm1)
    if walls:
        print(f"warning: the chi={warm_chi} warm source has {walls} domain walls", flush=True)
    mps, _ = source.mps.compress(max_bond_dim=chi, cutoff=1e-14)
    with contextlib.redirect_stdout(io.StringIO()):
        MPE(mps, mpo, mps).dmrg(
            bdims=[chi] * n_sweeps,
            noises=[1e-6] * (n_sweeps - 2) + [0.0] * 2,
            dav_thrds=[1e-10],
            iprint=-1,
            n_sweeps=n_sweeps,
        )
    return mps_trial_from_pyblock3(mps, nelec=nelec)


def uhf_orbitals(h1, u, nelec):
    """pyscf UHF of the Hubbard model from a Neel guess (the uhf_rotation_cpmc.ipynb recipe)."""
    from pyscf import ao2mo, gto, scf

    L = len(h1)
    mol = gto.M(verbose=0)
    mol.incore_anyway = True
    mol.nelec = nelec
    guess = np.zeros((2, L, L))
    for i in range(L):
        guess[i % 2, i, i] = 1.0  # up on even sites, down on odd sites
    eri = np.zeros((L,) * 4)
    eri[np.arange(L), np.arange(L), np.arange(L), np.arange(L)] = u
    mf = scf.UHF(mol)
    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: np.eye(L)
    mf._eri = ao2mo.restore(8, eri, L)
    mf.kernel(guess)
    return mf.mo_coeff[0][:, : nelec[0]], mf.mo_coeff[1][:, : nelec[1]], mf


def trials(ham, sys_, chis, with_uhf, rotated_start="plain"):
    """(name, plain MpsTrial, rotated MpsTrial) for every requested trial.

    rotated_start="plain": the rotated trial carries the plain trial's rdm1, so both runs start from the
    same walkers and differ only in the trial. "rotated": trot's GHF convention, the spin-diagonal 1-RDM
    of the rotated (unprojected) trial. That one is fully degenerate for a rotated Neel product state
    (1/2 on every site), and its arbitrary natural orbitals can give a start with zero overlap.
    """
    from trot.trial.mps import make_mps_trial, mps_trial_from_sd, rotate_spin

    nelec = tuple(int(n) for n in sys_.nelec)
    if rotated_start not in ("plain", "rotated"):
        raise ValueError(f"rotated_start must be 'plain' or 'rotated', got {rotated_start!r}")

    def start(t):
        return t.rdm1 if rotated_start == "plain" else None  # None: make_mps_trial's default

    out = []
    if with_uhf:
        Ca, Cb, _ = uhf_orbitals(np.asarray(ham.h1), float(ham.u), nelec)
        out.append(("UHF", mps_trial_from_sd(Ca, Cb)))
    for spec in chis:  # "chi", "chi:seed" or "chi@warm_chi" (see warm_dmrg)
        spec = str(spec)
        if "@" in spec:
            chi, warm = spec.split("@")
            out.append((f"DMRG chi={chi} warm={warm}", warm_dmrg(ham, sys_, int(chi), int(warm))))
            continue
        chi, _, seed = spec.partition(":")
        name = f"DMRG chi={chi}" + (f" seed={seed}" if seed else "")
        out.append((name, dmrg(ham, sys_, int(chi), seed=int(seed or 0)).trial))
    return [
        (name, t, make_mps_trial(rotate_spin(t.tensors, R90), nelec=nelec, rdm1=start(t)))
        for name, t in out
    ]


def block_logger(path, n_equilibration, tag, base_block_fn):
    """base_block_fn that appends each block's scalars to a JSONL file as soon as the block finishes
    (as trot.gmps.driver.make_block_logger: raw values, before trot's outlier rejection)."""
    import jax
    import jax.experimental
    import jax.numpy as jnp

    counter = iter(range(1 << 62))
    start = time.perf_counter()

    def write(energy, weight, e_estimate, nodes):
        block = next(counter)
        record = dict(
            tag=tag,
            block=block,
            phase="equilibration" if block < n_equilibration else "sampling",
            energy=float(energy),
            weight=float(weight),
            e_estimate=float(e_estimate),
            node_encounters=int(nodes),
            seconds=time.perf_counter() - start,
        )
        with Path(path).open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        return np.int32(0)

    def block_fn(state, **kwargs):
        state, obs = base_block_fn(state, **kwargs)
        jax.experimental.io_callback(
            write,
            jax.ShapeDtypeStruct((), jnp.int32),
            obs.scalars["energy"],
            obs.scalars["weight"],
            state.e_estimate,
            state.node_encounters,
            ordered=True,
        )
        return state, obs

    return block_fn


def make_block_fn(energy_clip=None, weight_cap=None):
    """trot's blocks.block with its estimator regularisation changed, for tests of that regularisation.

    blocks.block replaces a walker's local energy by the running estimate when it is further than
    sqrt(2/dt) from it, and the step zeroes weights above params.weight_cap. Inside the block, params.dt
    is read only for that threshold (the propagation takes dt from prop_ctx), so handing the block a
    params with dt = 2/energy_clip**2 changes the threshold to energy_clip and nothing else.
    energy_clip=None / weight_cap=None keep trot's values; float("inf") switches either off.
    """
    from trot.prop import blocks

    if energy_clip is None and weight_cap is None:
        return blocks.block

    def block(state, *, params, **kwargs):
        if energy_clip is not None:
            params = dataclasses.replace(params, dt=2.0 / energy_clip**2)
        if weight_cap is not None:
            params = dataclasses.replace(params, weight_cap=weight_cap)
        return blocks.block(state, params=params, **kwargs)

    return block


def run(ham, sys_, trial, args, tag, blocks_path):
    from trot.driver import run_qmc
    from trot.meas.mps import make_mps_meas_ops_hubbard
    from trot.prop.mps_cpmc import make_prop_ops
    from trot.trial.mps import make_mps_trial_ops, make_walker_plan
    from trot.prop.types import QmcParamsMps

    params = QmcParamsMps(
        n_walkers=args.walkers,
        n_eql_blocks=args.eql,
        n_blocks=args.blocks,
        dt=args.dt,
        n_prop_steps=args.steps,
        weight_floor=args.floor,
        seed=args.seed,
        orbital_plan="maximal" if args.walker_chi is None else "adaptive",
        walker_channel_chi=args.walker_chi,
        propagator=getattr(args, "propagator", "fast"),
    )
    # the ops of trot.gmps.driver.make_mps_cpmc_ops, built here so the study does not import the driver
    plan = make_walker_plan(ham, trial, sys_, params)
    trial_ops = make_mps_trial_ops(plan)
    meas_ops = make_mps_meas_ops_hubbard(plan, energy_kernel=params.energy_kernel)
    prop_ops = make_prop_ops(ham, sys_, plan, propagator=params.propagator)
    energy_clip, weight_cap = getattr(args, "energy_clip", None), getattr(args, "weight_cap", None)
    block_fn = make_block_fn(energy_clip, weight_cap)
    start = time.perf_counter()
    result = run_qmc(
        sys=sys_,
        params=params,
        ham_data=ham,
        trial_data=trial,
        trial_ops=trial_ops,
        meas_ops=meas_ops,
        prop_ops=prop_ops,
        prop_ctx=prop_ops.build_prop_ctx(ham, None, params),
        block_fn=block_logger(blocks_path, args.eql, tag, block_fn),
    )
    blocks = [
        b for b in map(json.loads, Path(blocks_path).read_text().splitlines()) if b["tag"] == tag
    ]
    return dict(
        cpmc=float(result.mean_energy),
        cpmc_error=float(result.stderr_energy),
        nodes=sum(b["node_encounters"] for b in blocks),
        seconds=time.perf_counter() - start,
        n_walkers=args.walkers,
        n_eql=args.eql,
        n_blocks=args.blocks,
        dt=args.dt,
        n_steps=args.steps,
        weight_floor=args.floor,
        walker_chi=args.walker_chi,
        seed=args.seed,
        propagator=getattr(args, "propagator", "fast"),
        energy_clip="default" if energy_clip is None else energy_clip,
        weight_cap="default" if weight_cap is None else weight_cap,
    )


def study(args):
    out = Path(args.out)
    blocks_path = out.with_name(out.stem + "_blocks.jsonl")
    h1, ham, sys_ = hubbard_chain(args.L, args.U)
    reference = dmrg(ham, sys_, args.chi_ref)
    e_ref = reference.variational_energy
    print(
        f"L={args.L} U={args.U:g}: reference DMRG chi={args.chi_ref} E = {e_ref:.10f}", flush=True
    )
    for name, plain, rotated in trials(ham, sys_, args.chi, args.uhf, args.rotated_start):
        walls, max_m, profile = domain_walls(plain.rdm1)
        for variant, trial in (("plain", plain), ("rotated", rotated)):
            tag = f"L{args.L}_U{args.U:g}_{name.replace(' ', '').replace('=', '')}_{variant}_s{args.seed}"
            record = dict(
                tag=tag,
                L=args.L,
                U=args.U,
                trial=name,
                variant=variant,
                e_ref=e_ref,
                rotated_start=args.rotated_start,
                trial_bond=max(trial.bond_dims),
                sector_weight=trial.sector_weight,
            )
            record.update(diagnostics(trial.tensors, h1, args.U, e_ref))
            record.update(  # of the plain trial: a rotation cannot remove or create walls
                walls=walls,
                max_staggered_m=max_m,
                staggered_m=[round(float(v), 4) for v in profile],
            )
            record.update(run(ham, sys_, trial, args, tag, blocks_path))
            record["bias"] = record["cpmc"] - e_ref
            record["bias_sigma"] = record["bias"] / record["cpmc_error"]
            with out.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(
                f"RESULT {name:20s} {variant:8s} walls {walls} p0 {record['p0']:.3f} "
                f"E_var-E_ref {record['e_var_error']:+.4f}  "
                f"CPMC-E_ref {record['bias']:+.5f} +/- {record['cpmc_error']:.5f} ({record['bias_sigma']:+.1f} sigma)  "
                f"nodes {record['nodes']}  {record['seconds']:.0f} s",
                flush=True,
            )


def plot(path, show=False):
    """One figure per (L, U, trial): CPMC block energy - E_ref against tau, plain and rotated together."""
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from trot.gmps.plot_cpmc_runs import GRID, INK, INK2, PALETTE, SURFACE

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "axes.titlesize": 14,
            "axes.labelsize": 10,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "axes.grid": True,
            "grid.color": GRID,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.6,
            "legend.frameon": False,
            "font.size": 10,
        }
    )
    path = Path(path)
    records = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    blocks = [
        json.loads(x) for x in path.with_name(path.stem + "_blocks.jsonl").read_text().splitlines()
    ]
    groups = {}
    for r in records:
        groups.setdefault((r["L"], r["U"], r["trial"]), []).append(r)
    for (L, U, name), runs in groups.items():
        fig, ax = plt.subplots(figsize=(8.0, 4.2))
        for i, r in enumerate(sorted(runs, key=lambda r: r["variant"])):
            series = [b for b in blocks if b["tag"] == r["tag"]]
            tau = (np.array([b["block"] for b in series]) + 1) * r["n_steps"] * r["dt"]
            energy = np.array([b["energy"] for b in series]) - r["e_ref"]
            color = PALETTE[0] if r["variant"] == "plain" else PALETTE[1]
            ax.plot(
                tau,
                energy,
                "-",
                marker="o",
                ms=2.0,
                color=color,
                label=f"{r['variant']} (singlet weight {r['p0']:.2f}): "
                f"{r['bias']:+.4f} $\\pm$ {r['cpmc_error']:.4f}, {r['nodes']} nodes",
            )
            ax.axhline(
                r["e_var_error"],
                color=color,
                ls=":",
                lw=1.2,
                label=f"{r['variant']} trial energy: {r['e_var_error']:+.4f}",
            )
        ax.axhline(0.0, color=INK, lw=1.5, label="reference (DMRG $\\chi$=400)")
        ax.axvline(
            runs[0]["n_eql"] * runs[0]["n_steps"] * runs[0]["dt"],
            color=INK2,
            lw=1.0,
            ls="--",
            zorder=0,
            label="end of equilibration",
        )
        ax.set_xlabel(r"imaginary time $\tau$")
        ax.set_ylabel(r"CPMC energy per block $-$ $E_{ref}$")
        ax.set_title(f"CPMC, Hubbard chain L={L}, U={U:g}, {name} trial: plain vs rotated by 90°")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
        out = path.with_name(
            f"{path.stem}_L{L}_U{U:g}_{name.replace(' ', '').replace('=', '')}.png"
        )
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"saved {out}")
    if show:
        plt.show()


# ---------------------------------------------------------------------------------------------
# Exact references (L <= 10), the no-SR weight-leak test, the chi ladder and the bridge analogue
# ---------------------------------------------------------------------------------------------


def _fock_helpers():
    import sys

    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    from tests.helpers import hubbard_fock

    return hubbard_fock


class ExactSector:
    """Dense (N_up, N_dn)-sector Hamiltonian, ground energy and trot's Trotter split for small L."""

    def __init__(self, h1, u, nelec, dt):
        hf = _fock_helpers()
        self.hf, self.nelec, self.dt = hf, tuple(nelec), dt
        self.H = np.asarray(hf.hubbard_sector_hamiltonian(h1, u, *nelec))
        self.E0 = float(np.linalg.eigvalsh(self.H)[0])
        K = np.asarray(hf.hubbard_sector_hamiltonian(h1, 0.0, *nelec))
        kv, kq = np.linalg.eigh(K)
        self.half = (kq * np.exp(-0.5 * dt * kv)) @ kq.T  # e^{-dt K/2}
        self.pot = np.exp(-dt * np.diag(self.H - K))  # e^{-dt U sum n_up n_dn}

    def amplitudes(self, tensors):
        return self.hf.mps_sector_amplitudes(tensors, *self.nelec).reshape(-1)

    def walker_amplitudes(self, walker):
        return self.hf.sd_amplitudes(np.asarray(walker[0]), np.asarray(walker[1])).reshape(-1)

    def trotter_steps(self, v, n):
        """n Trotter steps on v; returns (normalised v, log of the norm change)."""
        log_norm = 0.0
        for _ in range(n):
            v = self.half @ (self.pot * (self.half @ v))
            norm = np.linalg.norm(v)
            v, log_norm = v / norm, log_norm + np.log(norm)
        return v, log_norm


def leak_test(state, *, t_amp, exact, advance, chunk, n_chunks):
    """No-SR walk against the exact Trotterized projection of the actual start determinant.

    advance(state) -> (state, o_over_g, h_over_g): chunk steps, walkers orthonormalised, the guide
    overlaps recomputed, and per walker <T|phi>/g(phi) and <T|H|phi>/g(phi) (g = the propagation
    guide; for a plain trial g = <T|phi>, so the parts are 1 and E_L). Exact importance sampling gives
    E[mean w <T|phi>/g] = <T|e^{-tau(H - shift)}|phi0> / g(phi0), and the energy
    sum w <T|H|phi>/g / sum w <T|phi>/g estimates E_mixed(tau). Returns one row per chunk.
    """
    shift = float(state.pop_control_ene_shift)
    phi0 = exact.walker_amplitudes((state.walkers[0][0], state.walkers[1][0]))
    g0 = float(state.overlaps[0])
    v, log_norm, rows = phi0 / np.linalg.norm(phi0), np.log(np.linalg.norm(phi0)), []
    for c in range(1, n_chunks + 1):
        state, o_g, h_g = advance(state)
        v, dlog = exact.trotter_steps(v, chunk)
        log_norm += dlog
        tau = c * chunk * exact.dt
        z_exact = (t_amp @ v) * np.exp(log_norm + tau * shift) / g0
        w, o_g, h_g = (np.asarray(x) for x in (state.weights, o_g, h_g))
        tw = w * o_g
        rows.append(
            dict(
                tau=tau,
                norm_ratio=float(np.mean(tw) / z_exact),
                energy=float(np.sum(w * h_g) / np.sum(tw) - exact.E0),
                energy_exact=float((t_amp @ exact.H @ v) / (t_amp @ v) - exact.E0),
                ess=float(np.sum(w) ** 2 / np.sum(w**2)),
                max_over_mean=float(np.max(w) / np.mean(w)),
                sign=float(np.sum(tw) / np.sum(np.abs(tw))),
                nodes=int(state.node_encounters),
            )
        )
        r = rows[-1]
        print(
            f"  leak tau {tau:4.1f}: norm ratio {r['norm_ratio']:.4f}  E-E0 {r['energy']:+.4f} "
            f"(exact {r['energy_exact']:+.4f})  ESS {r['ess']:.0f}  max/mean {r['max_over_mean']:.1f}  "
            f"sign {r['sign']:+.3f}  nodes {r['nodes']}",
            flush=True,
        )
    return rows


def leak_params(args, chunk):
    from trot.prop.types import QmcParamsMps

    return QmcParamsMps(
        n_walkers=args.leak_walkers,
        n_eql_blocks=0,
        n_blocks=1,
        dt=args.dt,
        n_prop_steps=chunk,
        weight_floor=args.floor,
        weight_cap=float("inf"),
        pop_control_damping=0.0,
        seed=args.seed,
        orbital_plan="maximal",
        walker_channel_chi=None,
        auto_n_chunks=False,
        n_chunks=1,
    )


def mps_leak_test(ham, sys_, trial, exact, args, chunk=50):
    """leak_test for an MpsTrial with trot's MPS CPMC step (the trial is the guide)."""
    import jax
    from jax import lax

    from trot import walkers as wk
    from trot.core.ops import k_energy
    from trot.meas.mps import make_mps_meas_ops_hubbard
    from trot.prop.mps_cpmc import make_prop_ops
    from trot.trial.mps import make_mps_trial_ops, make_walker_plan

    params = leak_params(args, chunk)
    plan = make_walker_plan(ham, trial, sys_, params)
    trial_ops, meas_ops = make_mps_trial_ops(plan), make_mps_meas_ops_hubbard(plan)
    prop_ops = make_prop_ops(ham, sys_, plan)
    meas_ctx = meas_ops.build_meas_ctx(ham, trial)
    prop_ctx = prop_ops.build_prop_ctx(ham, None, params)
    state = prop_ops.init_prop_state(
        sys=sys_,
        ham_data=ham,
        trial_ops=trial_ops,
        trial_data=trial,
        meas_ops=meas_ops,
        params=params,
        meas_ctx=meas_ctx,
    )
    t_amp = exact.amplitudes(trial.tensors)
    start = exact.walker_amplitudes((state.walkers[0][0], state.walkers[1][0]))
    if abs(float(state.overlaps[0]) / (t_amp @ start) - 1.0) > 1e-8:
        raise RuntimeError("the start determinant's MPS overlap is not exact")
    overlap = jax.vmap(trial_ops.overlap, in_axes=(0, None))
    energy = jax.vmap(meas_ops.kernels[k_energy], in_axes=(0, None, None, None))

    @jax.jit
    def advance(s):
        def body(c, _):
            c = prop_ops.step(
                c,
                params=params,
                ham_data=ham,
                trial_data=trial,
                trial_ops=trial_ops,
                meas_ops=meas_ops,
                meas_ctx=meas_ctx,
                prop_ctx=prop_ctx,
            )
            return c, None

        s, _ = lax.scan(body, s, None, length=chunk)
        walkers = wk.orthonormalize(s.walkers, "unrestricted")
        s = s._replace(walkers=walkers, overlaps=overlap(walkers, trial))
        e = energy(walkers, ham, meas_ctx, trial)
        return s, jax.numpy.ones_like(e), e

    n_chunks = int(round(args.leak_tau / (chunk * args.dt)))
    return leak_test(
        state, t_amp=t_amp, exact=exact, advance=advance, chunk=chunk, n_chunks=n_chunks
    )


def leak_summary(rows, spike=100.0):
    by_tau = {round(r["tau"], 6): r for r in rows}
    spikes = [r["tau"] for r in rows if r["max_over_mean"] > spike]
    return dict(
        norm_ratio_tau2=by_tau.get(2.0, {}).get("norm_ratio"),
        norm_ratio_tau4=by_tau.get(4.0, {}).get("norm_ratio"),
        min_ess_fraction=min(r["ess"] for r in rows) / max(rows[0]["ess"], 1.0),
        max_weight_ratio=max(r["max_over_mean"] for r in rows),
        first_spike_tau=spikes[0] if spikes else None,
        min_sign=min(r["sign"] for r in rows),
    )


def ladder(args):
    """For each DMRG trial chi: diagnostics, standard CPMC (plain and rotated) and the leak test."""
    from trot.trial.mps import make_mps_trial, rotate_spin

    out = Path(args.out)
    blocks_path = out.with_name(out.stem + "_blocks.jsonl")
    h1, ham, sys_ = hubbard_chain(args.L, args.U)
    nelec = tuple(int(n) for n in sys_.nelec)
    exact = ExactSector(h1, args.U, nelec, args.dt)
    print(f"ladder L={args.L} U={args.U:g}: exact E0 {exact.E0:.10f}", flush=True)
    for spec in args.chi:
        chi = int(spec)
        plain = dmrg(ham, sys_, chi).trial
        rotated = make_mps_trial(rotate_spin(plain.tensors, R90), nelec=nelec, rdm1=plain.rdm1)
        walls, _, _ = domain_walls(plain.rdm1)
        p0 = diagnostics(plain.tensors, h1, args.U, exact.E0)["p0"]
        variants = [("plain", plain)] + ([] if p0 > 0.999 else [("rotated", rotated)])
        for variant, trial in variants:
            tag = f"ladder_L{args.L}_U{args.U:g}_chi{chi}_{variant}_s{args.seed}"
            record = dict(tag=tag, L=args.L, U=args.U, chi=chi, variant=variant, e_ref=exact.E0)
            record.update(diagnostics(trial.tensors, h1, args.U, exact.E0))
            record.update(walls=walls, trial_bond=max(trial.bond_dims))
            print(
                f"chi={chi} {variant}: E_var-E0 {record['e_var_error']:+.4f} p0 {record['p0']:.3f}",
                flush=True,
            )
            record.update(run(ham, sys_, trial, args, tag, blocks_path))
            record["bias"] = record["cpmc"] - exact.E0
            rows = mps_leak_test(ham, sys_, trial, exact, args)
            record.update(leak=rows, **leak_summary(rows))
            with out.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(
                f"RESULT chi={chi:3d} {variant:8s} E_var-E0 {record['e_var_error']:+.4f}  "
                f"CPMC-E0 {record['bias']:+.5f} +/- {record['cpmc_error']:.5f}  nodes {record['nodes']}  "
                f"norm ratio tau2 {record['norm_ratio_tau2']:.3f} tau4 {record['norm_ratio_tau4']:.3f}  "
                f"max w/mean {record['max_weight_ratio']:.0f}",
                flush=True,
            )


def plot_ladder(path, show=False):
    """CPMC bias and the tau=2 weight leak against the trial's variational error, plain and rotated."""
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from trot.gmps.plot_cpmc_runs import INK, PALETTE

    path = Path(path)
    records = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for i, variant in enumerate(("plain", "rotated")):
        rs = sorted((r for r in records if r["variant"] == variant), key=lambda r: r["e_var_error"])
        if not rs:
            continue
        x = [r["e_var_error"] for r in rs]
        ax1.errorbar(
            x,
            [r["bias"] for r in rs],
            yerr=[r["cpmc_error"] for r in rs],
            fmt="o-",
            color=PALETTE[i],
            capsize=3,
            label=variant,
        )
        ax2.plot(x, [1 - r["norm_ratio_tau4"] for r in rs], "o-", color=PALETTE[i], label=variant)
        labels = {}
        for r in rs:  # identical trials (e.g. chi=5 and 6 at L=8) share one label
            labels.setdefault((round(r["e_var_error"], 8), round(r["bias"], 8)), []).append(
                r["chi"]
            )
        for (xv, yv), chis in labels.items():
            ax1.annotate(
                "χ=" + ",".join(map(str, chis)),
                (xv, yv),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )
    ax1.axhline(0.0, color=INK, lw=1.2)
    ax1.set_xscale("log")
    ax1.set_xlabel("trial variational error E_var − E0")
    ax1.set_ylabel("CPMC − E0")
    ax1.set_title("CPMC bias", fontsize=14)
    ax2.axhline(0.0, color=INK, lw=1.2)
    ax2.set_xscale("log")
    ax2.set_xlabel("trial variational error E_var − E0")
    ax2.set_ylabel("weight missing at τ = 4 (1 − mean w / exact)")
    ax2.set_title("Weight leak (no population control)", fontsize=14)
    for ax in (ax1, ax2):
        ax.legend(loc="upper left")
    fig.suptitle(
        f"L={records[0]['L']}, U={records[0]['U']:g}: how good must the one trial be?", fontsize=15
    )
    out = path.with_name(path.stem + "_ladder.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")
    if show:
        plt.show()


def dense_trial_ops(trial, exact, eps):
    """The trial as a dense sector vector with the zero-free guide g = sqrt(<T|phi>^2 + eps^2 <T|H|phi>^2).

    Returns (DenseTrial pytree, trial_ops, meas_ops, parts) where parts(walker) = (<T|phi>, <T|H|phi>).
    eps = 0 gives g = |<T|phi>|, the plain importance function as long as no walker crosses a node.
    """
    from typing import NamedTuple

    import jax
    import jax.numpy as jnp
    from jax import tree_util

    from trot.core.ops import MeasOps, TrialOps, k_energy

    hf = exact.hf
    L = trial.norb
    nup, ndn = exact.nelec
    t = exact.amplitudes(trial.tensors)
    rows_a = np.asarray(hf.sector_basis(L, nup)[0])
    rows_b = np.asarray(hf.sector_basis(L, ndn)[0])
    shape = (len(rows_a), len(rows_b))

    @tree_util.register_pytree_node_class
    class DenseTrial(NamedTuple):
        t: jax.Array
        ht: jax.Array
        rdm1: jax.Array

        def tree_flatten(self):
            return (self.t, self.ht, self.rdm1), None

        @classmethod
        def tree_unflatten(cls, aux, children):
            return cls(*children)

    data = DenseTrial(
        jnp.asarray(t.reshape(shape)),
        jnp.asarray((exact.H @ t).reshape(shape)),
        jnp.asarray(trial.rdm1),
    )

    def parts(walker, trial_data):
        da = jnp.linalg.det(walker[0][rows_a])
        db = jnp.linalg.det(walker[1][rows_b])
        return da @ trial_data.t @ db, da @ trial_data.ht @ db

    def guide(walker, trial_data):
        o, h = parts(walker, trial_data)
        return jnp.sqrt(o**2 + eps**2 * h**2)

    def energy(walker, ham_data, meas_ctx, trial_data):
        o, h = parts(walker, trial_data)
        return h / o

    trial_ops = TrialOps(overlap=jax.jit(guide), get_rdm1=lambda d: d.rdm1)
    meas_ops = MeasOps(
        overlap=jax.jit(guide),
        build_meas_ctx=lambda h, d: None,
        kernels={k_energy: jax.jit(energy)},
    )
    return data, trial_ops, meas_ops, parts


def make_bridge_block(parts, sign0):
    """blocks.block for the zero-free guide: energy = sum w <T|H|phi>/g / sum w <T|phi>/g, the block's
    'weight' is the signed denominator (times the start overlap's sign, so it is positive while no walker has
    crossed a node) and 'sign' its ratio to the unsigned one. Random stream and SR as in blocks.block.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax

    from trot import walkers as wk
    from trot.prop.blocks import BlockObs

    def block(
        state,
        *,
        sys,
        params,
        ham_data,
        trial_data,
        trial_ops,
        meas_ops,
        meas_ctx,
        prop_ops,
        prop_ctx,
        sr_fn=wk.stochastic_reconfiguration,
        observable_names=(),
    ):
        def body(c, _):
            c = prop_ops.step(
                c,
                params=params,
                ham_data=ham_data,
                trial_data=trial_data,
                trial_ops=trial_ops,
                meas_ops=meas_ops,
                prop_ctx=prop_ctx,
                meas_ctx=meas_ctx,
            )
            return c, None

        state, _ = lax.scan(body, state, None, length=params.n_prop_steps)
        walkers = wk.orthonormalize(state.walkers, sys.walker_kind)
        g = jax.vmap(meas_ops.overlap, in_axes=(0, None))(walkers, trial_data)
        o, h = jax.vmap(parts, in_axes=(0, None))(walkers, trial_data)
        state = state._replace(walkers=walkers, overlaps=g)
        w = state.weights
        num, den, unsigned = jnp.sum(w * h / g), jnp.sum(w * o / g), jnp.sum(w * jnp.abs(o) / g)
        e_block = num / den
        alpha = jnp.asarray(params.shift_ema, dtype=jnp.result_type(e_block))
        state = state._replace(e_estimate=(1.0 - alpha) * state.e_estimate + alpha * e_block)
        key_next, key_sr = jax.random.split(state.rng_key)
        zeta = jax.random.uniform(key_sr)
        w_sr, weights_sr = sr_fn(state.walkers, state.weights, zeta, sys.walker_kind)
        state = state._replace(
            walkers=w_sr,
            weights=weights_sr,
            overlaps=jax.vmap(meas_ops.overlap, in_axes=(0, None))(w_sr, trial_data),
            rng_key=key_next,
        )
        scalars = {"energy": e_block, "weight": sign0 * den, "sign": sign0 * den / unsigned}
        return state, BlockObs(scalars=scalars, observables={})

    return block


def bridge(args):
    """The bridge analogue on one trial (default chi=4): leak test and standard run for each eps."""
    import jax
    from jax import lax

    from trot import walkers as wk
    from trot.driver import run_qmc
    from trot.prop import cpmc_slow
    from trot.prop.types import QmcParams

    out = Path(args.out)
    blocks_path = out.with_name(out.stem + "_blocks.jsonl")
    h1, ham, sys_ = hubbard_chain(args.L, args.U)
    nelec = tuple(int(n) for n in sys_.nelec)
    if args.L > 10:
        raise SystemExit("--bridge uses dense sector vectors: L <= 10")
    exact = ExactSector(h1, args.U, nelec, args.dt)
    trial = dmrg(ham, sys_, args.bridge_chi).trial
    print(
        f"bridge L={args.L} U={args.U:g} chi={args.bridge_chi}: exact E0 {exact.E0:.10f}",
        flush=True,
    )
    for eps in args.bridge:
        data, trial_ops, meas_ops, parts = dense_trial_ops(trial, exact, eps)
        prop_ops = cpmc_slow.make_prop_ops(ham, "unrestricted")
        record = dict(L=args.L, U=args.U, chi=args.bridge_chi, eps=eps, e_ref=exact.E0)

        # leak test: no SR, cap off, damping 0
        chunk = 50
        lp = leak_params(args, chunk)
        pctx = prop_ops.build_prop_ctx(ham, None, lp)
        state = prop_ops.init_prop_state(
            sys=sys_,
            ham_data=ham,
            trial_ops=trial_ops,
            trial_data=data,
            meas_ops=meas_ops,
            params=lp,
            meas_ctx=None,
        )

        @jax.jit
        def advance(s):
            def body(c, _):
                c = prop_ops.step(
                    c,
                    params=lp,
                    ham_data=ham,
                    trial_data=data,
                    trial_ops=trial_ops,
                    meas_ops=meas_ops,
                    prop_ctx=pctx,
                    meas_ctx=None,
                )
                return c, None

            s, _ = lax.scan(body, s, None, length=chunk)
            walkers = wk.orthonormalize(s.walkers, "unrestricted")
            g = jax.vmap(meas_ops.overlap, in_axes=(0, None))(walkers, data)
            o, h = jax.vmap(parts, in_axes=(0, None))(walkers, data)
            return s._replace(walkers=walkers, overlaps=g), o / g, h / g

        print(f"eps={eps:g}: leak test", flush=True)
        n_chunks = int(round(args.leak_tau / (chunk * args.dt)))
        rows = leak_test(
            state,
            t_amp=exact.amplitudes(trial.tensors),
            exact=exact,
            advance=advance,
            chunk=chunk,
            n_chunks=n_chunks,
        )
        record.update(leak=rows, **leak_summary(rows))

        # standard run with SR and the bridge block
        params = QmcParams(
            n_walkers=args.walkers,
            n_eql_blocks=args.eql,
            n_blocks=args.blocks,
            dt=args.dt,
            n_prop_steps=args.steps,
            weight_floor=args.floor,
            seed=args.seed,
        )
        o0, _ = parts(
            (
                jax.numpy.asarray(np.asarray(state.walkers[0][0])),
                jax.numpy.asarray(np.asarray(state.walkers[1][0])),
            ),
            data,
        )
        tag = f"bridge_L{args.L}_U{args.U:g}_chi{args.bridge_chi}_eps{eps:g}_s{args.seed}"
        block_fn = make_bridge_block(parts, float(np.sign(o0)))
        start = time.perf_counter()
        result = run_qmc(
            sys=sys_,
            params=params,
            ham_data=ham,
            trial_data=data,
            trial_ops=trial_ops,
            meas_ops=meas_ops,
            prop_ops=prop_ops,
            prop_ctx=prop_ops.build_prop_ctx(ham, None, params),  # pyright: ignore
            block_fn=block_logger(blocks_path, args.eql, tag, block_fn),
        )
        record.update(
            tag=tag,
            cpmc=float(result.mean_energy),
            cpmc_error=float(result.stderr_energy),
            seconds=time.perf_counter() - start,
            n_walkers=args.walkers,
        )
        record["bias"] = record["cpmc"] - exact.E0
        with out.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(
            f"RESULT bridge eps={eps:g}: CPMC-E0 {record['bias']:+.5f} +/- {record['cpmc_error']:.5f}  "
            f"leak norm ratio tau2 {record['norm_ratio_tau2']:.3f} tau4 {record['norm_ratio_tau4']:.3f}  "
            f"max w/mean {record['max_weight_ratio']:.0f}  min sign {record['min_sign']:+.3f}  "
            f"{record['seconds']:.0f} s",
            flush=True,
        )


def self_test():
    """p_S on states with known spin, the 90-degree weight against the projection, and a 2-block run."""
    from trot.trial.mps import make_mps_trial, mps_trial_from_sd, rotate_spin

    h1, ham, sys_ = hubbard_chain(6, 4.0)
    p = spin_distribution(dmrg(ham, sys_, 64).trial.tensors)  # exact ground state: a singlet
    assert abs(p[0] - 1.0) < 1e-8 and np.all(np.abs(p[1:]) < 1e-8), p
    h1, ham, sys_ = hubbard_chain(8, 12.0)
    Ca, Cb, mf = uhf_orbitals(h1, 12.0, (4, 4))
    uhf = mps_trial_from_sd(Ca, Cb)
    d = diagnostics(uhf.tensors, h1, 12.0, 0.0)
    assert abs(d["s_s1"] - mf.spin_square()[0]) < 1e-8, (d["s_s1"], mf.spin_square()[0])
    assert abs(d["e_var"] - mf.e_tot) < 1e-10, (d["e_var"], mf.e_tot)
    rotated = make_mps_trial(rotate_spin(uhf.tensors, R90), nelec=(4, 4))
    assert abs(d["w90"] - rotated.sector_weight) < 1e-10, (d["w90"], rotated.sector_weight)
    p_rot = spin_distribution(rotated.tensors)
    assert np.all(np.abs(p_rot[1::2]) < 1e-8), "the 90-degree rotation must remove every odd S"
    print(
        f"self-test: singlet p_0 = 1; UHF <S(S+1)> {d['s_s1']:.6f} = pyscf; w90 {d['w90']:.6f} = projection; "
        f"rotated UHF has no odd S (p_S {np.round(p_rot[:5], 4)})"
    )
    args = argparse.Namespace(
        walkers=4, eql=1, blocks=2, dt=0.01, steps=2, floor=1e-8, seed=1, walker_chi=None
    )
    with tempfile.TemporaryDirectory() as directory:
        summary = run(ham, sys_, rotated, args, "selftest", Path(directory) / "blocks.jsonl")
    assert np.isfinite(summary["cpmc"])
    print(f"self-test passed (2-block run: E = {summary['cpmc']:.6f})")
    self_test_ladder_and_bridge()


def self_test_ladder_and_bridge():
    """Leak test exact at short tau; dense overlaps = MPS overlaps; eps = 0 bridge block = blocks.block."""
    import jax
    import jax.numpy as jnp

    from trot.prop import blocks, cpmc_slow
    from trot.prop.types import QmcParamsMps
    from trot.trial.mps import make_mps_trial_ops, make_walker_plan

    h1, ham, sys_ = hubbard_chain(8, 8.0)
    exact = ExactSector(h1, 8.0, (4, 4), 0.01)
    trial = dmrg(ham, sys_, 4).trial
    args = argparse.Namespace(leak_walkers=2000, leak_tau=0.5, dt=0.01, floor=1e-8, seed=1)
    rows = mps_leak_test(ham, sys_, trial, exact, args)
    assert abs(rows[0]["norm_ratio"] - 1.0) < 5e-3, rows[0]
    assert abs(rows[0]["energy"] - rows[0]["energy_exact"]) < 0.02, rows[0]

    data, trial_ops, meas_ops, parts = dense_trial_ops(trial, exact, 0.0)
    params = QmcParamsMps(
        n_walkers=5,
        dt=0.01,
        n_prop_steps=5,
        orbital_plan="maximal",
        walker_channel_chi=None,
        seed=3,
    )
    plan = make_walker_plan(ham, trial, sys_, params)
    mps_ops = make_mps_trial_ops(plan)
    hf = exact.hf
    rdm1 = np.asarray(trial.rdm1)
    from trot.trial.mps import natural_orbitals

    Ra, Rb = natural_orbitals(rdm1[0], 4)[0], natural_orbitals(rdm1[1], 4)[0]
    walkers = hf.random_field_walkers(h1, 8.0, 0.1, Ra, Rb, n=5, steps=10, seed=2)
    for wa, wb in walkers:
        w = (jnp.asarray(wa), jnp.asarray(wb))
        o, _ = parts(w, data)
        assert abs(float(o) / float(mps_ops.overlap(w, trial)) - 1.0) < 1e-10

    prop_ops = cpmc_slow.make_prop_ops(ham, "unrestricted")
    pctx = prop_ops.build_prop_ctx(ham, None, params)
    state = prop_ops.init_prop_state(
        sys=sys_,
        ham_data=ham,
        trial_ops=trial_ops,
        trial_data=data,
        meas_ops=meas_ops,
        params=params,
        meas_ctx=None,
    )
    o0, _ = parts((state.walkers[0][0], state.walkers[1][0]), data)
    common = dict(
        sys=sys_,
        params=params,
        ham_data=ham,
        trial_data=data,
        trial_ops=trial_ops,
        meas_ops=meas_ops,
        meas_ctx=None,
        prop_ops=prop_ops,
        prop_ctx=pctx,
    )
    s_bridge, obs_bridge = jax.jit(
        lambda s: make_bridge_block(parts, float(np.sign(o0)))(s, **common)
    )(state)
    plain_meas = type(meas_ops)(
        overlap=jax.jit(lambda w, d: parts(w, d)[0]),
        build_meas_ctx=lambda h, d: None,
        kernels=meas_ops.kernels,
    )
    plain_trial = type(trial_ops)(overlap=plain_meas.overlap, get_rdm1=trial_ops.get_rdm1)
    common.update(trial_ops=plain_trial, meas_ops=plain_meas)
    state_plain = state._replace(
        overlaps=jax.vmap(plain_meas.overlap, in_axes=(0, None))(state.walkers, data)
    )
    s_plain, obs_plain = jax.jit(lambda s: blocks.block(s, **common))(state_plain)
    assert abs(float(obs_bridge.scalars["energy"]) - float(obs_plain.scalars["energy"])) < 1e-12, (
        float(obs_bridge.scalars["energy"]),
        float(obs_plain.scalars["energy"]),
    )
    for a, b in zip(s_bridge.walkers, s_plain.walkers):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert abs(float(obs_bridge.scalars["sign"]) - 1.0) < 1e-12
    print(
        f"self-test (ladder/bridge) passed: leak norm ratio at tau 0.5 = {rows[0]['norm_ratio']:.4f}; dense "
        f"overlaps = MPS overlaps; eps=0 bridge block = blocks.block (E = {float(obs_plain.scalars['energy']):.10f})"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--U", type=float, default=8.0)
    parser.add_argument(
        "--chi",
        nargs="*",
        default=["2", "4", "6"],
        help="DMRG trials: bond dimension; bond:seed for another random start (6:1); "
        "bond@warm for DMRG started from the compressed bond-warm state (16@64, wall-free)",
    )
    parser.add_argument("--uhf", action="store_true", help="also the pyscf UHF trial")
    parser.add_argument(
        "--chi-ref", type=int, default=400, help="bond dimension of the reference DMRG"
    )
    parser.add_argument("--walkers", type=int, default=100)
    parser.add_argument("--eql", type=int, default=40, help="equilibration blocks (5k)")
    parser.add_argument("--blocks", type=int, default=80, help="sampling blocks (10k)")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=20, help="propagation steps per block")
    parser.add_argument("--floor", type=float, default=1e-8, help="weight floor")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--walker-chi",
        type=lambda s: None if s.lower() == "none" else int(s),
        default=None,
        help="walker channel bond dimension (none: exact walkers, maximal orbital plan)",
    )
    parser.add_argument(
        "--rotated-start",
        choices=("plain", "rotated"),
        default="plain",
        help="walker start of the rotated run: the plain trial's (the same walkers in both runs) "
        "or the rotated trial's 1-RDM (trot's GHF convention)",
    )
    parser.add_argument("--propagator", choices=("fast", "slow"), default="fast")
    parser.add_argument(
        "--energy-clip",
        type=float,
        default=None,
        help="local-energy clip around the running estimate (default: trot's sqrt(2/dt); inf: off)",
    )
    parser.add_argument(
        "--weight-cap", type=float, default=None, help="walker weight cap (default: 100; inf: off)"
    )
    parser.add_argument("--out", default="rotated_trial_study.jsonl")
    parser.add_argument(
        "--ladder",
        action="store_true",
        help="chi ladder (L <= 10): diagnostics, standard CPMC and the no-SR weight-leak test per --chi",
    )
    parser.add_argument(
        "--bridge",
        type=float,
        nargs="*",
        default=None,
        metavar="EPS",
        help="bridge analogue (L <= 10): zero-free guide sqrt(<T|phi>^2 + eps^2 <T|H|phi>^2)",
    )
    parser.add_argument(
        "--bridge-chi", type=int, default=4, help="trial bond dimension for --bridge"
    )
    parser.add_argument("--leak-walkers", type=int, default=2000)
    parser.add_argument("--leak-tau", type=float, default=4.0)
    parser.add_argument(
        "--plot", metavar="JSONL", help="plot the runs of a results file instead of running"
    )
    parser.add_argument("--plot-ladder", metavar="JSONL", help="plot a --ladder results file")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    from trot.config import configure_once

    configure_once()
    if args.self_test:
        self_test()
    elif args.plot:
        plot(args.plot, args.show)
    elif args.plot_ladder:
        plot_ladder(args.plot_ladder, args.show)
    elif args.ladder:
        ladder(args)
    elif args.bridge is not None:
        bridge(args)
    else:
        study(args)


if __name__ == "__main__":
    main()
