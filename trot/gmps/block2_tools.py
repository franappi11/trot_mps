"""Dense charge-labelled MPS (trot's trial format, or sd_to_gmps output) into a block2 SZ MPS.

The helpers of slurm/dmrg_trial_8x8_block2.py (unchanged, except that to_block2_mps takes the DMRGDriver instead of
reading a global), so the 6x6 block2 script can use them: site_blocks keeps each site's nonzero (left sector, p)
blocks, prune_sectors drops the sectors that do not run from the vacuum to the target, right_canonicalize brings the
blocks into the form block2 expects, and to_block2_mps builds the block2 MPS. Needs block2 (~/.trot312_block2).
"""

import block2 as bf
import block2.sz as bx
import numpy as np
from block2 import SZ as BSZ
from block2 import Global, VectorInt
from block2.sz import MPSInfo, StateInfo, VectorStateInfo

PHYSICAL = [(0, 0), (1, 0), (0, 1), (1, 1)]  # (n_up, n_dn) of the local states |0>, |up>, |dn>, |up dn>


def shift(sector, p):
    """The (n_up, n_dn) sector after adding the local state p."""
    return (sector[0] + PHYSICAL[p][0], sector[1] + PHYSICAL[p][1])


def site_blocks(A, q_left, q_right):
    """One site's dense tensor (left, 4, right) as {(left sector, p): matrix (rows in that sector, columns in the
    right sector)}, the nonzero blocks only (the blocks gmps_to_pyblock3 makes, as plain arrays)."""
    A, blocks = np.asarray(A), {}
    for left in sorted(set(tuple(int(x) for x in row) for row in q_left)):
        rows = np.flatnonzero((q_left == left).all(axis=1))
        for p in range(4):
            cols = np.flatnonzero((q_right == shift(left, p)).all(axis=1))
            block = A[np.ix_(rows, [p], cols)][:, 0, :]
            if np.any(block):
                blocks[(left, p)] = block
    return blocks


def right_canonicalize(sites):
    """Sector-wise LQ sweep from the last site to site 1 (in place), what pyblock3's canonicalize(0) does: sites 1..N-1
    become right-canonical and site 0 keeps the norm, the form block2 expects ("K" at site 0, then "R"). The L factor
    of each left sector goes into the right index of the previous site; sectors no block reaches are dropped."""
    for i in range(len(sites) - 1, 0, -1):
        blocks, new, factors = sites[i], {}, {}
        for s in sorted({left for left, _ in blocks}):
            keys = [(s, p) for p in range(4) if (s, p) in blocks]
            Q_t, R_t = np.linalg.qr(np.hstack([blocks[k] for k in keys]).T)  # M^T = Q_t R_t, so M = R_t^T Q_t^T
            Q, col = Q_t.T, 0
            for k in keys:
                width = blocks[k].shape[1]
                new[k] = Q[:, col:col + width]
                col += width
            factors[s] = R_t.T  # (rows, k)
        sites[i] = new
        sites[i - 1] = {(s, p): B @ factors[shift(s, p)] for (s, p), B in sites[i - 1].items() if shift(s, p) in factors}


def prune_sectors(sites):
    """Drop the charge sectors that do not run from the vacuum to the target: a bond sector one neighbour never reaches
    or never leaves (zero-amplitude completion vectors). Forward and backward until nothing changes."""
    changed = True
    while changed:
        changed = False
        for i in range(len(sites) - 1):  # site i+1 keeps the left sectors that site i reaches
            reached = {shift(s, p) for (s, p) in sites[i]}
            kept = {k: B for k, B in sites[i + 1].items() if k[0] in reached}
            changed |= len(kept) != len(sites[i + 1])
            sites[i + 1] = kept
        for i in range(len(sites) - 1, 0, -1):  # site i-1 keeps the right sectors that site i starts from
            starts = {s for (s, p) in sites[i]}
            kept = {k: B for k, B in sites[i - 1].items() if shift(*k) in starts}
            changed |= len(kept) != len(sites[i - 1])
            sites[i - 1] = kept


def to_block2_mps(sites, n_up, n_dn, driver, tag="KET"):
    """Right-canonical sector blocks as a block2 MPS (centre at site 0, one-site form): the block2 half of
    pyblock3.block2.io.MPSTools.to_block2, fed from the blocks instead of a pyblock3 MPS. It saves the MPS info
    into the driver's scratch (Global.frame), so the DMRGDriver must exist."""
    n = len(sites)
    left = [{s: B.shape[0] for (s, _), B in site.items()} for site in sites]
    right = [{shift(s, p): B.shape[1] for (s, p), B in site.items()} for site in sites]
    for i in range(n - 1):
        if left[i + 1] != right[i]:
            common = set(left[i + 1]) & set(right[i])
            raise AssertionError(
                f"bond {i + 1}: only in site {i}'s right leg {sorted(set(right[i]) - set(left[i + 1]))}; "
                f"only in site {i + 1}'s left leg {sorted(set(left[i + 1]) - set(right[i]))}; "
                f"dimensions differ (sector, right, left) "
                f"{[(s, right[i][s], left[i + 1][s]) for s in sorted(common) if right[i][s] != left[i + 1][s]]}")
    assert left[0] == {(0, 0): 1} and right[-1] == {(n_up, n_dn): 1}, "the MPS does not run from vacuum to target"

    q = lambda s: BSZ(s[0] + s[1], s[0] - s[1], 0)

    def state_info(dims):  # {(n_up, n_dn): number of states} -> block2 StateInfo
        info = StateInfo()
        info.allocate(len(dims))
        for ix, (s, count) in enumerate(sorted(dims.items())):
            info.quanta[ix] = q(s)
            info.n_states[ix] = count
        info.sort_states()
        return info

    vacuum, target = BSZ(0, 0, 0), BSZ(n_up + n_dn, n_up - n_dn, 0)
    basis = [state_info({s: 1 for s in PHYSICAL}) for _ in range(n)]  # the complete local basis on every site
    minfo = MPSInfo(n, vacuum, target, VectorStateInfo(basis))
    minfo.set_bond_dimension_full_fci(vacuum, vacuum)
    minfo.left_dims[0] = StateInfo(vacuum)
    for i in range(n):
        info = minfo.left_dims[i + 1]  # bond after site i
        info.allocate(len(right[i]))
        for ix, (s, count) in enumerate(sorted(right[i].items())):
            info.quanta[ix] = q(s)
            info.n_states[ix] = count
        info.sort_states()
    minfo.right_dims[n] = StateInfo(vacuum)
    for i in range(n):
        info = minfo.right_dims[i]  # bond before site i, labelled by what the right part carries
        info.allocate(len(left[i]))
        for ix, (s, count) in enumerate(sorted(left[i].items())):
            info.quanta[ix] = target - q(s)
            info.n_states[ix] = count
        info.sort_states()
    minfo.tag = tag
    minfo.bond_dim = minfo.get_max_bond_dimension()
    minfo.save_mutable()
    minfo.save_data(f"{Global.frame.save_dir}/{tag}-mps_info.bin")

    tensors = []
    for site, local in zip(sites, basis):
        tensor = bx.SparseTensor()
        tensor.data = bx.VectorVectorPSSTensor([bx.VectorPSSTensor() for _ in range(local.n)])
        for (s, p), B in site.items():
            im = local.find_state(q(PHYSICAL[p]))
            assert im != -1
            tensor.data[im].append(((q(s), q(shift(s, p))), bf.Tensor(VectorInt([B.shape[0], 1, B.shape[1]]))))
            np.array(tensor.data[im][-1][1], copy=False)[:] = B[:, None, :]
        tensors.append(tensor)
    umps = bx.UnfusedMPS()
    umps.info = minfo
    umps.n_sites = n
    umps.canonical_form = "K" + "R" * (n - 1)
    umps.center = 0
    umps.dot = 1
    umps.tensors = bx.VectorSpTensor(tensors)
    ket = umps.finalize()
    ket, _ = driver.adjust_mps(ket, dot=2)  # two-site form: the first sweep contracts sites 0 and 1
    return ket
