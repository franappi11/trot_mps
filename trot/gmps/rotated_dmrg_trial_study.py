"""Does rotating a spin-broken trial remove the CPMC bias? Small-system study on the Hubbard chain.

Every trial (pyblock3 DMRG at the given bond dimensions; with --uhf also the pyscf UHF determinant) runs
twice with the same seed: as it is ("plain") and rotated by R_y(pi/2), i.e. projected onto the walkers'
S_z = 0 sector after the rotation ("rotated", rotate_mps_trial(T, R)). The rotation removes
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
    from trot.trial.mps import mps_trial_from_sd
    from trot.trial.mps_rotation import rotate_mps_trial

    nelec = tuple(int(n) for n in sys_.nelec)
    if rotated_start not in ("plain", "rotated"):
        raise ValueError(f"rotated_start must be 'plain' or 'rotated', got {rotated_start!r}")

    def start(t):
        return t.rdm1 if rotated_start == "plain" else None  # None: the rotated trial's rdm1, before projection

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
        (name, t, rotate_mps_trial(t, R90, nelec=nelec, rdm1=start(t)))
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
    )
    # the ops of trot.gmps.driver.make_mps_cpmc_ops, built here so the study does not import the driver
    plan = make_walker_plan(ham, trial, sys_, params)
    trial_ops = make_mps_trial_ops(plan)
    meas_ops = make_mps_meas_ops_hubbard(plan, energy_kernel=params.energy_kernel)
    prop_ops = make_prop_ops(ham, sys_, plan)
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


def self_test():
    """p_S on states with known spin, the 90-degree weight against the projection, and a 2-block run."""
    from trot.trial.mps import mps_trial_from_sd
    from trot.trial.mps_rotation import rotate_mps_trial

    h1, ham, sys_ = hubbard_chain(6, 4.0)
    p = spin_distribution(dmrg(ham, sys_, 64).trial.tensors)  # exact ground state: a singlet
    assert abs(p[0] - 1.0) < 1e-8 and np.all(np.abs(p[1:]) < 1e-8), p
    h1, ham, sys_ = hubbard_chain(8, 12.0)
    Ca, Cb, mf = uhf_orbitals(h1, 12.0, (4, 4))
    uhf = mps_trial_from_sd(Ca, Cb)
    d = diagnostics(uhf.tensors, h1, 12.0, 0.0)
    assert abs(d["s_s1"] - mf.spin_square()[0]) < 1e-8, (d["s_s1"], mf.spin_square()[0])
    assert abs(d["e_var"] - mf.e_tot) < 1e-10, (d["e_var"], mf.e_tot)
    rotated = rotate_mps_trial(uhf, R90)
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
        "--plot", metavar="JSONL", help="plot the runs of a results file instead of running"
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    from trot.config import configure_once

    configure_once()
    if args.self_test:
        self_test()
    elif args.plot:
        plot(args.plot, args.show)
    else:
        study(args)


if __name__ == "__main__":
    main()
