from trot.config import configure_once
configure_once()  # float64
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from trot.prop.blocks import load_prop_state_npz

# python error_analysis_8x8.py [walkers.npz [reference chi]]
WALKERS = sys.argv[1] if len(sys.argv) > 1 else "/mnt/ceph/users/fnappi/trot_walkers/uhf_8x8_doped_U8/walkers_01100.npz"
ref_chi = int(sys.argv[2]) if len(sys.argv) > 2 else 1024  # the old way's reference walker
n_walkers = 8  # walkers scored the new way (own plan, against the exact walker)
n_check = 2  # the first n_check of them are also scored the old way (against the reference walker)
chis = [32, 64, 128, 256, 512]
tolerance = 1e-10  # make_orbital_plan's occupation tolerance, channel_to_gmps' default


def own_plan_losses(job):
    """New way, one spin channel of one walker, planned on the walker itself (its own adaptive gates and its own kept
    counts at every chi), as in walker_truncation_check._channel_norms' w_own: the infidelity against the exact walker
    is 1 - (1 - plan_loss)(1 - discarded), with plan_loss = 1 - |gauge|^2 of the plan (~0 for its own) and discarded
    the norm plan_bonds' truncation loses at that chi."""
    from trot.gmps import utils
    from trot.gmps.driver import _plan_fidelity

    i, spin, C, plan_chis = job
    start = time.perf_counter()
    Q = np.linalg.qr(np.asarray(C, dtype=float))[0]
    plan = utils.make_orbital_plan(Q, "adaptive", tolerance)
    plan_loss = 1.0 - _plan_fidelity(Q, plan)
    discarded = [utils.plan_bonds(Q, plan, chi).reference_discarded_weight for chi in plan_chis]
    return i, spin, plan_loss, discarded, time.perf_counter() - start


def convert_channel(job):
    """Old way, one spin channel of one walker at one chi: channel_to_gmps' MPS and its discarded norm, made by the same
    algorithm in NumPy (utils.make_orbital_plan, plan_bonds, engine.channel_mps_host, the host reference of the device
    conversion) so that the worker processes do not each start JAX (the first run was OOM-killed at 8 GB)."""
    from trot.gmps import engine, utils

    i, spin, C, chi = job
    start = time.perf_counter()
    Q = np.linalg.qr(np.asarray(C, dtype=float))[0]
    plan = utils.make_orbital_plan(Q, "adaptive", tolerance)
    bond = utils.plan_bonds(Q, plan, chi)
    tensors, _, _ = engine.channel_mps_host(Q, plan, bond)
    return i, spin, chi, tensors, bond.reference_discarded_weight, time.perf_counter() - start


def overlap(bra, ket):  # the notebook's _overlap_tensors (the prefactors cancel in the fidelity)
    env = np.ones((1, 1))
    for A, B in zip(bra, ket):
        env = np.einsum("ab,asc,bsd->cd", env, A.conj(), B, optimize=True)
    return env[0, 0]


def main():
    np.set_printoptions(precision=10, linewidth=200, threshold=100000)  # full arrays, to copy for plotting offline
    state = load_prop_state_npz(WALKERS)

    wu, wd = (np.asarray(w) for w in state.walkers)  # unrestricted: (n_walkers, norb, nup) and (n_walkers, norb, ndn)
    weights = state.weights  # (n_walkers,)
    print(WALKERS)
    print(wu.shape, wd.shape, weights.shape)
    n_sites = wu.shape[1]
    cuts = np.arange(1, n_sites)
    center_cut = n_sites // 2
    w_abs = np.abs(np.asarray(weights))

    entropy_by_cut = np.zeros((wu.shape[0], len(cuts)))  # (n_walkers, n_cuts), both spins summed
    for walker in (wu, wd):
        q, _ = np.linalg.qr(walker, mode="reduced")  # orthonormal columns spanning each walker
        for k, cut in enumerate(cuts):
            nu = np.clip(np.linalg.svd(q[:, :cut, :], compute_uv=False) ** 2, 0.0, 1.0)
            active = (nu > 1e-14) & (nu < 1 - 1e-14)
            nu = np.where(active, nu, 0.5)
            entropy_by_cut[:, k] += np.sum(np.where(active, -(nu * np.log(nu) + (1 - nu) * np.log1p(-nu)), 0.0), axis=1)

    probs = np.array([0.10, 0.50, 0.90])
    cut_mean = np.average(entropy_by_cut, axis=0, weights=w_abs)
    cut_q = np.empty((len(cuts), len(probs)))  # weighted quantiles over walkers
    for k in range(len(cuts)):
        order = np.argsort(entropy_by_cut[:, k])
        cum = (np.cumsum(w_abs[order]) - 0.5 * w_abs[order]) / np.sum(w_abs)
        cut_q[k] = np.interp(probs, cum, entropy_by_cut[order, k])

    max_entropy = entropy_by_cut.max(axis=1)  # most entangled cut of each walker
    print(f"center cut  : mean S = {cut_mean[center_cut - 1]:.4f}, exp(S) = {np.exp(cut_mean[center_cut - 1]):.3f}")
    print(f"maximum cut : mean S = {np.average(max_entropy, weights=w_abs):.4f}, exp(S) = {np.exp(np.average(max_entropy, weights=w_abs)):.3f}")
    print(f"largest S of any walker at any cut: {entropy_by_cut.max():.4f} (exp = {np.exp(entropy_by_cut.max()):.2f})")
    print("center_cut", center_cut)
    print("cuts", repr(cuts))
    print("cut_mean", repr(cut_mean))
    print("cut_q10", repr(cut_q[:, 0]))
    print("cut_q50", repr(cut_q[:, 1]))
    print("cut_q90", repr(cut_q[:, 2]), flush=True)

    # infidelity of the walker bond, one process per core (spawn, one BLAS thread each, as walker_truncation_check)
    ids = np.linspace(0, wu.shape[0] - 1, n_walkers).round().astype(int)  # spread over the population
    plan_chis = sorted(set(chis) | {ref_chi})
    check_chis = [chi for chi in chis if chi < ref_chi]
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    workers = len(os.sched_getaffinity(0))
    plan_jobs = [(i, spin, (wu, wd)[spin][k], plan_chis) for i, k in enumerate(ids) for spin in range(2)]
    convert_jobs = [(i, spin, (wu, wd)[spin][ids[i]], chi)
                    for i in range(n_check) for spin in range(2) for chi in [ref_chi, *check_chis]]
    plan_loss = np.zeros((n_walkers, 2))
    discarded = np.zeros((n_walkers, 2, len(plan_chis)))
    mps = {}
    print(f"\nnew way: {len(plan_jobs)} channels of walkers {ids.tolist()}, bonds {plan_chis}; old way: "
          f"{len(convert_jobs)} conversions of walkers {ids[:n_check].tolist()} at bonds {[ref_chi, *check_chis]}; "
          f"{workers} processes", flush=True)
    start = time.perf_counter()
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(convert_channel, job): "old" for job in convert_jobs}  # the slow ones first
        futures.update({pool.submit(own_plan_losses, job): "new" for job in plan_jobs})
        for done, future in enumerate(as_completed(futures), 1):
            elapsed = f"({done}/{len(futures)}, {time.perf_counter() - start:.0f} s elapsed)"
            if futures[future] == "new":
                i, spin, loss, d, seconds = future.result()
                plan_loss[i, spin], discarded[i, spin] = loss, d
                print(f"new, walker {ids[i]} {('up', 'dn')[spin]}: plan loss {loss:.2e}, discarded "
                      f"{np.array2string(np.array(d))}, {seconds:.0f} s {elapsed}", flush=True)
            else:
                i, spin, chi, tensors, dropped, seconds = future.result()
                mps[i, spin, chi] = tensors
                print(f"old, walker {ids[i]} {('up', 'dn')[spin]}: chi {chi} discarded {dropped:.4e}, "
                      f"{seconds:.0f} s {elapsed}", flush=True)

    # 1 - prod over both channels of (1 - plan loss)(1 - discarded), through log1p/expm1 to keep the small values
    infidelities = -np.expm1(np.sum(np.log1p(-plan_loss[:, :, None]) + np.log1p(-discarded), axis=1))  # (walker, chi)
    median_infidelity = np.median(infidelities, axis=0)
    worst_infidelity = np.max(infidelities, axis=0)
    print("\nnew way (own plan, against the exact walker): largest plan loss of any channel", plan_loss.max())
    print("chi_w", repr(np.array(plan_chis)))
    print("median infidelity", repr(median_infidelity))
    print("worst infidelity ", repr(worst_infidelity))
    for k, chi in enumerate(plan_chis):
        print(f"infidelity per walker, chi_w {chi}", repr(infidelities[:, k]))

    # the old way, the notebook's: against the reference walker at ref_chi, up and down channel
    print(f"\nold way vs new way (the old way's reference is truncated at chi {ref_chi}, the new way's is exact):")
    for i in range(n_check):
        reference_new = infidelities[i, plan_chis.index(ref_chi)]
        print(f"walker {ids[i]}: the chi {ref_chi} reference itself is {reference_new:.4e} from the exact walker")
        for chi in check_chis:
            fidelity = 1.0
            for spin in range(2):
                ref, walker = mps[i, spin, ref_chi], mps[i, spin, chi]
                fidelity *= abs(overlap(ref, walker)) ** 2 / (abs(overlap(ref, ref)) * abs(overlap(walker, walker)))
            old, new = 1.0 - fidelity, infidelities[i, plan_chis.index(chi)]
            print(f"  chi_w {chi:4d}: old {old:.6e}   new {new:.6e}   old/new {old / new:.6f}   old - new {old - new:+.3e}")


if __name__ == "__main__":  # the spawned workers import this file, and must not run main()
    main()
