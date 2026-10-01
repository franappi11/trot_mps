"""CPMC with a DMRG trial for the square-lattice Hubbard model.

mps_cpmc_new.py is lattice-agnostic apart from its open-chain hopping matrix and
its nearest-neighbour chain MPO. This script supplies both for an Lx x Ly lattice
and reuses everything else from mps_cpmc_new unchanged.

Sites are ordered x*Ly + y, with the short side Ly running fastest. y-hops then have
range 1 and x-hops range Ly, so the MPO width and the entanglement across a cut both
scale with Ly. Boundaries are open, periodic or antiperiodic per direction.
Antiperiodic flips the sign of the wrap bonds: h1 stays real, which the walker MPS
code requires, and the free-fermion shell degeneracy of the torus is lifted.

    ~/.trot/bin/python mps_cpmc_2d.py cpmc Lx=4 Ly=2 n_up=4 n_down=4 trial_chi=256
    ~/.trot/bin/python mps_cpmc_2d.py dmrg Lx=4 Ly=4 n_up=8 n_down=8 dmrg_bdims=500,1000,2000 variational=False

Arguments are Config fields as key=value. Values are Python literals or bare strings.
The dmrg mode appends the high-bond-dimension reference to result_json. There,
e_mps is <H> of the final MPS, which is variational. e_davidson is the two-site
energy before the last truncation, so it sits below e_mps by an amount set by the
discarded weight.
"""
from __future__ import annotations

import ast
import sys
import time
from dataclasses import asdict, dataclass

import jax.numpy as jnp
import numpy as np

from pyblock3.algebra.mpe import MPE
from pyblock3.fcidump import FCIDUMP
from pyblock3.hamiltonian import Hamiltonian

from trot.core.ops import MeasOps, k_energy
from trot.core.system import System
from trot.gmps import mps_cpmc_new as m
from trot.ham.hubbard import HamHubbard, square_hopping_matrix
from trot.meas.mps import CHANNEL_CHARGE, apply_mpo, hubbard_mpo_from_h1, trial_times_h
from trot.prop import blocks
from trot.prop.types import QmcParams
from trot.trial.auto import make_auto_trial_ops
from trot.trial.uhf import UhfTrial, get_rdm1 as uhf_get_rdm1

__all__ = [
    "CHANNEL_CHARGE",
    "Config",
    "FCIDUMP",
    "HamHubbard",
    "Hamiltonian",
    "MPE",
    "MeasOps",
    "QmcParams",
    "System",
    "UhfTrial",
    "apply_mpo",
    "asdict",
    "ast",
    "blocks",
    "build_dmrg_hamiltonian",
    "config_from_args",
    "dataclass",
    "densified_trial",
    "describe",
    "dmrg_schedule",
    "hubbard_mpo_from_h1",
    "jnp",
    "k_energy",
    "lattice_hopping",
    "m",
    "main",
    "make_auto_trial_ops",
    "make_blocked_energy",
    "np",
    "reference",
    "run_dmrg",
    "square_hopping_matrix",
    "sys",
    "time",
    "trial_times_h",
    "uhf_get_rdm1",
]


@dataclass(frozen=True)
class Config:
    Lx: int = 4
    Ly: int = 2
    boundary_x: str = "open"  # open, periodic, antiperiodic
    boundary_y: str = "open"
    n_up: int = 4
    n_down: int = 4
    hopping: float = 1.0
    interaction: float = 4.0
    trial_chi: int = 64
    # Bond-dimension ramp, one entry per sweep, with the last entry repeating. If set it
    # replaces trial_chi, and noise stays on until the ramp is done.
    dmrg_bdims: tuple[int, ...] = ()
    dmrg_sweeps: int = 14
    dmrg_tol: float = 1.0e-6
    dmrg_seed: int = 0
    # rank_exact is exact for every walker. The adaptive plan is only guaranteed for the
    # reference, which a degenerate Fermi shell makes arbitrary.
    orbital_plan: str = "rank_exact"
    occupation_tolerance: float = 1.0e-10
    walker_channel_chi: int | None = None
    walker_cutoff: float = 0.0
    plan_reference: str = "natural"  # natural, rhf
    walker_start: str = "natural"  # natural, rhf
    n_walkers: int = 32
    n_blocks: int = 40
    n_equilibration: int = 15
    n_steps: int = 20
    dt: float = 0.01
    weight_floor: float = 1.0e-8
    seed: int = 1234
    n_chunks: int = 1
    # Local energy through charge-blocked contractions. False uses make_walker_ops' dense
    # contraction with a compressed H|trial>.
    blocked_energy: bool = True
    variational: bool = True  # dmrg mode: also <trial|H|trial> through the dense MPO
    result_json: str = ""
    block_log: str = ""
    tag: str = ""

    @property
    def n_sites(self) -> int:
        return self.Lx * self.Ly


def lattice_hopping(cfg: Config) -> np.ndarray:
    return square_hopping_matrix(cfg.Lx, cfg.Ly, cfg.hopping, cfg.boundary_x, cfg.boundary_y)


def build_dmrg_hamiltonian(cfg: Config, h1):
    n = cfg.n_sites
    g2 = np.zeros((n,) * 4)
    i = np.arange(n)
    g2[i, i, i, i] = cfg.interaction
    fcidump = FCIDUMP(pg="c1", n_sites=n, n_elec=cfg.n_up + cfg.n_down,
                      twos=cfg.n_up - cfg.n_down, ipg=0, h1e=h1, g2e=g2)
    return Hamiltonian(fcidump, flat=True)


def dmrg_schedule(cfg: Config):
    """Per-sweep bond dimensions and noises. Without a ramp this is m.run_dmrg's schedule."""
    if not cfg.dmrg_bdims:
        return [cfg.trial_chi] * cfg.dmrg_sweeps, [1.0e-5] * min(6, cfg.dmrg_sweeps) + [0.0]
    ramp = list(cfg.dmrg_bdims)
    if len(ramp) >= cfg.dmrg_sweeps:
        raise ValueError("dmrg_sweeps must exceed len(dmrg_bdims) so the final bond gets noiseless sweeps")
    bdims = ramp + [ramp[-1]] * (cfg.dmrg_sweeps - len(ramp))
    return bdims, [1.0e-5] * max(6, len(ramp)) + [0.0]


def run_dmrg(hamiltonian, cfg: Config, iprint=-1):
    """Returns the MPS, the last Davidson energy, the per-sweep energies and <H> of the MPS.

    Davidson energies belong to the two-site wavefunction before truncation. Only the
    last value, from pyblock3's block-sparse MPO, is variational for the returned MPS.
    """
    np.random.seed(cfg.dmrg_seed)
    bdims, noises = dmrg_schedule(cfg)
    mpo, _ = hamiltonian.build_qc_mpo().compress(cutoff=1.0e-12)
    mps = hamiltonian.build_mps(bdims[0])
    result = MPE(mps, mpo, mps).dmrg(bdims=bdims, noises=noises, dav_thrds=[1.0e-10],
                                     iprint=iprint, n_sweeps=cfg.dmrg_sweeps, tol=cfg.dmrg_tol)
    energies = [float(e) for e in result.energies]
    mps_energy = float(MPE(mps, mpo, mps)[0:2].expectation) / float(mps @ mps)
    return mps, energies[-1], energies, mps_energy


def describe(cfg: Config, h1):
    """Print the lattice and the free-fermion gaps at the Fermi level."""
    eps = np.linalg.eigvalsh(h1)
    gap = lambda N: eps[N] - eps[N - 1] if 0 < N < len(eps) else float("inf")
    print(f"{cfg.Lx}x{cfg.Ly} lattice, boundaries x={cfg.boundary_x} y={cfg.boundary_y}, "
          f"({cfg.n_up},{cfg.n_down}), U={cfg.interaction}")
    print(f"free-fermion gap eps_N - eps_N-1: {gap(cfg.n_up):.3e}, {gap(cfg.n_down):.3e}"
          " (about 0 means the rhf determinant is an arbitrary pick in a degenerate shell)")


def densified_trial(cfg: Config, h1, iprint=-1):
    hamiltonian = build_dmrg_hamiltonian(cfg, h1)
    dmrg_mps, dmrg_energy, sweep_energies, mps_energy = run_dmrg(hamiltonian, cfg, iprint)
    trial_np, trial_charges = m.densify_with_charges(dmrg_mps, cfg.n_sites)
    return trial_np, trial_charges, dmrg_energy, sweep_energies, mps_energy


def make_blocked_energy(ops, Ra, Rb, trial_np, Htrial_np, Htrial_charges):
    """<H trial|walker> / <trial|walker>, both through charge-blocked contractions.

    This replaces make_walker_ops' dense energy, which contracts the full d=4 walker MPS
    with a compressed H|trial> and dominates the 4x4 runtime.
    """
    _, qa, _, qb, _ = ops.convert(jnp.asarray(Ra), jnp.asarray(Rb))
    walker_charges = m.combined_charges(qa, qb)
    layouts = []
    for tensors, plan in ((Htrial_np, m.make_contraction_plan(walker_charges, Htrial_charges)),
                          (trial_np, ops.overlap_plan)):
        layouts.append((m.make_channel_block_maps(plan, qa, qb), m.extract_fixed_blocks(tensors, plan), plan))

    def energy(walker, _ham=None, _ctx=None, _trial_data=None):
        alpha, _, beta, _, _ = ops.convert(*walker)
        numerator, denominator = (
            m.blocked_contract_from_blocks(m.extract_channel_blocks(alpha, beta, maps), blocks, plan)
            for maps, blocks, plan in layouts)
        return numerator / denominator
    return energy


def reference(cfg: Config):
    """High-bond-dimension DMRG energy, appended to result_json."""
    h1 = lattice_hopping(cfg)
    describe(cfg, h1)
    start = time.perf_counter()
    trial_np, trial_charges, dmrg_energy, sweep_energies, mps_energy = densified_trial(cfg, h1, iprint=0)
    record = asdict(cfg)
    record.update(kind="dmrg", n_sites=cfg.n_sites, e_davidson=dmrg_energy, e_mps=mps_energy,
                  sweep_energies=sweep_energies, bdims=dmrg_schedule(cfg)[0],
                  bond_dims=[int(A.shape[0]) for A in trial_np] + [int(trial_np[-1].shape[-1])])
    if cfg.variational:
        W = hubbard_mpo_from_h1(h1, cfg.interaction)
        Htrial_np, _ = trial_times_h(W, trial_np, trial_charges)
        trial, Htrial = tuple(map(jnp.asarray, trial_np)), tuple(map(jnp.asarray, Htrial_np))
        record.update(e_variational=float(m.contract_real(Htrial, trial) / m.contract_real(trial, trial)),
                      mpo_bond=W.shape[1])
    record["seconds"] = time.perf_counter() - start
    print({k: record[k] for k in ("e_davidson", "e_mps", "e_variational", "bond_dims", "seconds") if k in record})
    m.save_result(cfg.result_json, record)
    return record


def main(cfg: Config):
    """mps_cpmc_new.main on the square lattice. The walker machinery is unchanged."""
    for option in ("plan_reference", "walker_start"):
        if getattr(cfg, option) not in ("rhf", "natural"):
            raise ValueError(f"{option} must be 'rhf' or 'natural'")
    h1 = lattice_hopping(cfg)
    n = cfg.n_sites
    ham = HamHubbard(h1=jnp.asarray(h1), u=cfg.interaction)
    system = System(norb=n, nelec=(cfg.n_up, cfg.n_down), walker_kind="unrestricted")
    _, orbitals = np.linalg.eigh(h1)
    Ca, Cb = orbitals[:, :cfg.n_up].copy(), orbitals[:, :cfg.n_down].copy()
    describe(cfg, h1)

    trial_np, trial_charges, dmrg_energy, _, mps_energy = densified_trial(cfg, h1)
    trial = tuple(jnp.asarray(A) for A in trial_np)
    np.testing.assert_allclose(float(m.contract_real(trial, trial)), 1.0, atol=1e-10)

    W = hubbard_mpo_from_h1(h1, cfg.interaction)
    mpo_bond = W.shape[1]
    Htrial_np, Htrial_charges = trial_times_h(W, trial_np, trial_charges)
    Htrial = tuple(jnp.asarray(A) for A in Htrial_np)
    trial_energy = float(m.contract_real(Htrial, trial) / m.contract_real(trial, trial))
    print(f"DMRG Davidson energy={dmrg_energy:.12f} (two-site, before truncation); trial energy: "
          f"pyblock3 {mps_energy:.12f}, dense MPO {trial_energy:.12f}")
    print("trial bonds:", [A.shape[0] for A in trial] + [trial[-1].shape[-1]])
    print(f"H|trial> bonds (uncompressed, MPO bond {mpo_bond}):",
          [A.shape[0] for A in Htrial] + [Htrial[-1].shape[-1]])

    determinants = {"rhf": (Ca, Cb)}
    if "natural" in (cfg.plan_reference, cfg.walker_start):
        gamma_a, gamma_b = m.one_rdm(trial_np)
        np.testing.assert_allclose([np.trace(gamma_a), np.trace(gamma_b)], [cfg.n_up, cfg.n_down], atol=1e-8)
        Na, occupations = m.natural_orbitals(gamma_a, cfg.n_up)
        Nb, _ = m.natural_orbitals(gamma_b, cfg.n_down)
        determinants["natural"] = (Na, Nb)
        if cfg.n_up < n:
            print(f"trial natural orbitals: alpha gap n_N - n_N+1 = {occupations[cfg.n_up - 1] - occupations[cfg.n_up]:.3f}")

    Ra, Rb = determinants[cfg.plan_reference]
    plan_a = m.make_orbital_plan(Ra, cfg.orbital_plan, cfg.occupation_tolerance)
    plan_b = m.make_orbital_plan(Rb, cfg.orbital_plan, cfg.occupation_tolerance)
    gates = lambda plan: int(plan.block_sizes.sum() - len(plan.block_sizes))
    print(f"plan reference: {cfg.plan_reference} determinant; gates {gates(plan_a)}, {gates(plan_b)}")

    Sa, Sb = determinants[cfg.walker_start]
    trial_data = UhfTrial(mo_coeff_a=jnp.asarray(Sa), mo_coeff_b=jnp.asarray(Sb))
    Pa, Pb = Sa @ Sa.T, Sb @ Sb.T
    initial_energy = float(np.sum(h1 * (Pa + Pb)) + cfg.interaction * np.diag(Pa) @ np.diag(Pb))

    def plan_fidelity(S, plan):
        _, rows = m.channel_angles(S, plan, xp=np)
        return float(np.linalg.det(np.stack([rows[i] for i in np.flatnonzero(plan.occupation)]))) ** 2
    print(f"walkers start from the {cfg.walker_start} determinant, E={initial_energy:.12f}; orbital-plan infidelity "
          f"{1 - plan_fidelity(Sa, plan_a):.1e}, {1 - plan_fidelity(Sb, plan_b):.1e}")

    bond_a = bond_b = None
    if cfg.walker_channel_chi is not None or cfg.walker_cutoff:
        bond_a = m.plan_bonds(Ra, plan_a, cfg.walker_channel_chi, cfg.walker_cutoff)
        bond_b = m.plan_bonds(Rb, plan_b, cfg.walker_channel_chi, cfg.walker_cutoff)
        print(f"walker truncation reference discarded weights: {bond_a.reference_discarded_weight:.3e}, "
              f"{bond_b.reference_discarded_weight:.3e}")

    dense_Htrial = None if cfg.blocked_energy else tuple(map(jnp.asarray, m.compress_mps(Htrial_np)))
    ops = m.make_walker_ops(Ra, Rb, plan_a, plan_b, bond_a, bond_b, trial_np, trial_charges, dense_Htrial)
    energy = (make_blocked_energy(ops, Ra, Rb, trial_np, Htrial_np, Htrial_charges)
              if cfg.blocked_energy else ops.energy)
    report = m.contraction_report(ops.overlap_plan)
    print("walker bonds:", [len(q) for q in ops.walker_charges])
    print("overlap environment entries:", report,
          f"dense/padded={report['dense']/report['padded_blocks']:.2f}x")

    prop_ops = m.make_fast_prop_ops(ham, system.walker_kind, ops.overlap, ops.sweep)
    params = QmcParams(dt=cfg.dt, n_walkers=cfg.n_walkers, n_prop_steps=cfg.n_steps,
                       n_blocks=cfg.n_blocks, n_eql_blocks=cfg.n_equilibration,
                       weight_floor=cfg.weight_floor, seed=cfg.seed, n_chunks=cfg.n_chunks)
    trial_ops = make_auto_trial_ops(system, overlap_u=ops.overlap, get_rdm1=uhf_get_rdm1)
    measurement = MeasOps(overlap=ops.overlap, kernels={k_energy: energy})

    block_fn = blocks.block
    if cfg.block_log:
        block_fn = m.make_block_logger(cfg.block_log, cfg.n_equilibration, cfg.tag)

    start = time.perf_counter()
    mean, error, _, _, collapsed = m.run_qmc_fixed_chunks(
        sys=system, params=params, ham_data=ham, trial_data=trial_data,
        meas_ops=measurement, trial_ops=trial_ops, prop_ops=prop_ops, block_fn=block_fn)
    elapsed = time.perf_counter() - start

    scalar = lambda x: None if x is None else float(x)
    print(f"CPMC energy = {scalar(mean)} +/- {scalar(error)}; elapsed={elapsed:.1f} s")
    record = asdict(cfg)
    record.update(
        kind="mps_2d", n_sites=n, initial_energy=initial_energy, dmrg_energy=dmrg_energy, mps_energy=mps_energy,
        trial_energy=trial_energy, cpmc_energy=scalar(mean), cpmc_error=scalar(error),
        seconds=elapsed, walker_d4_chi=max(map(len, ops.walker_charges)), collapsed_after_block=collapsed,
        gates=gates(plan_a), mpo_bond=mpo_bond)
    m.save_result(cfg.result_json, record)
    return record


def config_from_args(pairs):
    values = {}
    for pair in pairs:
        key, value = pair.split("=", 1)
        try:
            values[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            values[key] = value  # bare strings such as boundary_x=periodic
    if isinstance(values.get("dmrg_bdims"), int):
        values["dmrg_bdims"] = (values["dmrg_bdims"],)
    defaults = Config()
    for key, value in values.items():
        default = getattr(defaults, key)
        if isinstance(default, (int, float)) and not isinstance(default, bool) and isinstance(value, str):
            raise ValueError(f"{key}={value!r} is not a number (one key=value per argument)")
    return Config(**values)


if __name__ == "__main__":
    mode, *pairs = sys.argv[1:] or ["cpmc"]
    if mode not in ("cpmc", "dmrg"):
        raise SystemExit("usage: mps_cpmc_2d.py {cpmc|dmrg} [field=value ...]")
    (main if mode == "cpmc" else reference)(config_from_args(pairs))
