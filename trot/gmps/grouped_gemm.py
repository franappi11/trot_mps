"""The dense engine's trial GEMMs on the environment rows that can be nonzero: a grouped GEMM in one Pallas kernel.

A dense environment X (rows, Dt) goes through the trial's site as one GEMM per trial sector N,
X[:, start_N:stop_N] @ W_N (dense_engine._trial_apply, written into the zero (rows, 4, D') output). Its rows are the
walker's bond pairs (a, b), times the MPO channels w for the energy (mpo_contract); its columns t the trial's bond
states, in particle-number sectors. Row r has an electron number N(r) = n_a + n_b (+ the channel's offset), and X[r, t]
is exactly zero unless t lies in the sector of N(r): the walker tensors are snapped to their bond labels
(dense_conversion._snap) and the trial is stored in sector blocks, so every term of such an entry has an exact zero
factor. Of sector N's GEMM only its c_N rows can contribute; the others multiply zeros into zeros. On the doped 8x8 at
chi_T 2048 and walker bond 64 that is 81-83% of the multiply-adds (job 7366493).

c_N is data: it differs from walker to walker and changes as they propagate. XLA compiles static shapes, so it cannot
give each sector's GEMM a walker-dependent number of rows without a fixed capacity per sector (grouped_trial_gemm.ipynb
shows the alternatives and why they fail). Here the rows are sorted by sector (perm), cut into tiles of R rows that
never cross a sector, and one kernel launch multiplies every tile by its own sector's block. The launch grid is static
(at most ceil(rows / R) + G tiles for G sectors, whatever the walker) and each program reads its tile's sector and rows
at run time. Every output entry is the same sum of the same products as in _trial_apply; only the order of the sum
over a sector's columns can differ (rounding).

    spec = trial_spec(site.forward, site.right_widths, layout.sectors[i])  # static, per site and direction
    packed = pack_blocks(T[0])                                              # the site's blocks as one flat array
    g = row_sectors(n_a, n_b, spec, offsets)                                # each row's sector (G: none)
    Z = grouped_trial_apply(X, g, spec, packed)  # == _trial_apply(X, T[0], site.forward, site.right_widths)

On a GPU the kernel is compiled by Triton; elsewhere (the CPU tests, the notebook) it runs in Pallas' interpreter.
The engine uses it with TROT_TRIAL_GEMM=grouped (dense_engine.TRIAL_GEMM) for the MPO energy's and szsz's trial GEMMs,
the walker's labels coming from dense_converter(plan, labelled=True); profile_6x6_engines.py --stages grouped times it
against _trial_apply and the alternatives (cuBLAS behind fixed caps: no faster; Triton float64 is about half cuBLAS's
rate, and the dense (rows, 4, D') output costs ~3 ms per 8x8 energy call to write, job 7372537).

Further down, each with its own section: the fused kernel (fused_trial_apply, TROT_TRIAL_GEMM=fused), the walker
contractions on label blocks (label_contract, blocked), the walker-first right environments' trial GEMM
(summed_trial_apply, TROT_PROP_GEMM=grouped / envs) and the trial GEMM in cuBLAS through a custom call
(sector_products, cublas_fused_apply; TROT_TRIAL_GEMM=cublas).
"""
from __future__ import annotations

import ctypes
import math
import os
from functools import partial
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

N_PHYSICAL = (0, 1, 1, 2)  # electrons of the local states p = n_alpha + 2 n_beta (as dense_engine, which imports this)

# Tiles of the GPU kernels: R rows of one sector, BN output columns, BK of the sector's columns per step. Powers of 2
# (Triton); a float64 dot needs R >= 16, BN >= 8, BK >= 16. TILE_ROWS: multiply_adds' default row tile.
TILE_ROWS = 64


def _tile(name, default):
    """A tile (R, BK, BN, num_warps, num_stages) from the environment variable `name` ("64,32,128,8,2"), else
    default."""
    value = os.environ.get(name, "").strip()
    return tuple(int(x) for x in value.split(",")) if value else default


# The fastest tiles on the 8x8 at chi_T 2048, walker bond 64 (H100): (R, BK, BN, num_warps, num_stages). LARGE_TILE:
# stage 1's kernel (grouped_trial_apply) on the energy's 139264 rows, SMALL_TILE on the overlap's 4096 (job 7372094).
# FUSED_TILE: the fused kernel's on the energy's and szsz's rows (job 7375438: energy step 21.2 -> 19.8 ms against
# LARGE_TILE, szsz a tie; 7388160: (32, 16, 128, 4, 3) 15.6 -> 15.0 ms against (32, 32, 64, 4, 2), szsz 7.7 -> 7.5).
# CONTRACT_TILE: label_contract's (7375438: energy step 19.8 -> 15.5 ms against (32, 32, 64, 4, 2), szsz transfer
# 9.7 -> 7.7 ms; its neighbours in 7388160 within 0.3%). SUMMED_TILE: summed_trial_apply's (the propagation's
# walker-first right environments, the walkers pooled: ~51000 rows at 200 walkers of bond 16; not timed, LARGE_TILE's
# values). EPILOGUE_TILE: the fused kernel reading cuBLAS's products (products_fused_apply; not timed, BK unused).
# TINY_TILE (not timed) below 2048 rows. tile_for picks by the row count. Each can be set from the environment,
# TROT_TILE_LARGE, _SMALL, _TINY, _FUSED, _CONTRACT, _SUMMED, _EPILOGUE = "R,BK,BN,num_warps,num_stages"
# (profile_6x6_engines.py --tile-sweep times alternatives).
LARGE_TILE = _tile("TROT_TILE_LARGE", (64, 32, 128, 8, 2))
SMALL_TILE = _tile("TROT_TILE_SMALL", (64, 16, 64, 4, 3))
TINY_TILE = _tile("TROT_TILE_TINY", (16, 32, 64, 4, 2))
FUSED_TILE = _tile("TROT_TILE_FUSED", (32, 16, 128, 4, 3))
CONTRACT_TILE = _tile("TROT_TILE_CONTRACT", (16, 16, 64, 4, 3))
SUMMED_TILE = _tile("TROT_TILE_SUMMED", (64, 32, 128, 8, 2))
EPILOGUE_TILE = _tile("TROT_TILE_EPILOGUE", (32, 16, 128, 4, 2))
KINDS = ("grouped", "fused", "summed", "epilogue")  # the kernels with a large-row tile of their own


def tile_for(rows, kind="grouped") -> dict:
    """The tile options of a kernel of this kind (KINDS) on this many rows: from 32768 rows its own large tile
    (LARGE_TILE for grouped_trial_apply, FUSED_TILE, SUMMED_TILE, EPILOGUE_TILE), from 2048 SMALL_TILE, else
    TINY_TILE."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
    large = dict(grouped=LARGE_TILE, fused=FUSED_TILE, summed=SUMMED_TILE, epilogue=EPILOGUE_TILE)[kind]
    R, BK, BN, warps, stages = large if rows >= 32768 else SMALL_TILE if rows >= 2048 else TINY_TILE
    return dict(R=R, BK=BK, BN=BN, num_warps=warps, num_stages=stages)


def _tile_options(rows, kind, R, BK, BN, num_warps, num_stages) -> dict:
    """The given tile options, the missing ones from tile_for(rows, kind)."""
    given, default = dict(R=R, BK=BK, BN=BN, num_warps=num_warps, num_stages=num_stages), tile_for(rows, kind)
    return {k: default[k] if v is None else v for k, v in given.items()}


class TrialSpec(NamedTuple):
    """One site's sector blocks in one direction, as _trial_apply(X, matrices, entries, out_widths) uses them; all
    static. Per GEMM (sector) s: N[s] the electron number of the rows it takes, [start[s], start[s] + width[s]) its
    columns of X, ncols[s] the columns of its block (the four physical states' target sectors side by side), w_off[s]
    the block's offset in pack_blocks' flat array and c_off[s] its first entry in cmap. cmap: per block column, its
    column of the flat (4 D') output (p D' + the target sector's start + the column within it). in_cols = Dt (X's
    columns), out_cols = 4 D'."""
    N: tuple
    start: tuple
    width: tuple
    ncols: tuple
    w_off: tuple
    c_off: tuple
    cmap: tuple
    in_cols: int
    out_cols: int


def trial_spec(entries, out_widths, in_sectors) -> TrialSpec:
    """The TrialSpec of _trial_apply(X, matrices, entries, out_widths): entries a SiteLayout's forward (backward)
    entries (start, stop, cols), out_widths the other bond's sector widths (right_widths, left_widths), in_sectors the
    (N, start, stop) of the bond X's columns are on (SectorLayout.sectors[i], sectors[i + 1])."""
    N_of = {start: N for N, start, _ in in_sectors}
    d_out = int(sum(out_widths))
    out_start = np.concatenate(([0], np.cumsum(out_widths)))[:-1].astype(int)
    fields = dict(N=[], start=[], width=[], ncols=[], w_off=[], c_off=[])
    cmap, w_off = [], 0
    for start, stop, cols in entries:
        n_cols = int(sum(width for *_, width in cols))
        fields["N"].append(int(N_of[start]))
        fields["start"].append(int(start))
        fields["width"].append(int(stop - start))
        fields["ncols"].append(n_cols)
        fields["w_off"].append(w_off)
        fields["c_off"].append(len(cmap))
        for p, k, _, width in cols:  # in the order of their offsets: the block's columns
            cmap.extend(int(p * d_out + out_start[k] + j) for j in range(width))
        w_off += (stop - start) * n_cols
    return TrialSpec(**{k: tuple(v) for k, v in fields.items()}, cmap=tuple(cmap),
                     in_cols=int(sum(stop - start for _, start, stop in in_sectors)), out_cols=4 * d_out)


def forward_specs(layout) -> tuple:
    """Per site the TrialSpec of the left environments' trial GEMMs (_forward, _forward_mpo_gather, _left_transfer)."""
    return tuple(trial_spec(site.forward, site.right_widths, layout.sectors[i]) for i, site in enumerate(layout.sites))


def backward_specs(layout) -> tuple:
    """Per site the TrialSpec of the right environments' trial GEMMs (_backward, _right_transfer)."""
    return tuple(trial_spec(site.backward, site.left_widths, layout.sectors[i + 1])
                 for i, site in enumerate(layout.sites))


class FusedSpec(NamedTuple):
    """A TrialSpec with what the fused kernel (fused_trial_apply) and the block contractions (label_contract) need on
    top, all static. Per input sector s and physical state q (the trial's): piece_off[4 s + q] the column offset of q's
    target block in s's block (-1: none) and piece_target[4 s + q] that target's index among the output bond's sectors;
    out_start, out_width those sectors' columns of the output bond (d_out in all); in_index[N - n_lo] the input sector
    of electron number N, out_index[N - out_lo] the output sector (-1: none)."""
    trial: TrialSpec
    piece_off: tuple
    piece_target: tuple
    out_start: tuple
    out_width: tuple
    d_out: int
    n_lo: int
    in_index: tuple
    out_lo: int
    out_index: tuple


def _number_index(N):
    """(lo, index) with index[n - lo] the position of electron number n in N (-1: none)."""
    lo = min(N) if len(N) else 0
    index = np.full((max(N) - lo + 1) if len(N) else 1, -1)
    index[np.asarray(N, int) - lo] = np.arange(len(N))
    return int(lo), tuple(int(x) for x in index)


def fused_spec(entries, out_widths, in_sectors, out_sectors) -> FusedSpec:
    """The FusedSpec of a site's entries in one direction (arguments as trial_spec; out_sectors the output bond's
    (N, start, stop))."""
    trial = trial_spec(entries, out_widths, in_sectors)
    G = len(trial.N)
    offset, target = np.full((G, 4), -1), np.full((G, 4), -1)
    for s, (_, _, cols) in enumerate(entries):
        for q, k, column, _ in cols:
            offset[s, q], target[s, q] = column, k
    n_lo, in_index = _number_index(trial.N)
    out_lo, out_index = _number_index([N for N, _, _ in out_sectors])
    starts = np.concatenate(([0], np.cumsum(out_widths)))[:-1]
    return FusedSpec(trial, tuple(int(x) for x in offset.ravel()), tuple(int(x) for x in target.ravel()),
                     tuple(int(x) for x in starts), tuple(int(x) for x in out_widths), int(sum(out_widths)), n_lo,
                     in_index, out_lo, out_index)


class SummedSpec(NamedTuple):
    """One site's forward blocks as the walker-first right environments multiply them (dense_engine._backward_summed:
    the walker tensors first, the sum over the physical index inside the GEMM), all static. Per left sector s: N[s]
    its electron number (the rows (a, b) with n_a + n_b = N[s] go through it), out_start[s], out_width[s] its columns
    of the (rows, Dt) output, K[s] the inner dimension (the right sectors of the four physical states side by side,
    the forward block's columns), w_off[s] the transposed block's offset in pack_blocks([W.T for W in forward]) and
    k_off[s] its first entry in kmap. kmap: per inner index the column p Dt' + t' of the walker-first operand (p the
    physical state, t' the right bond's state). in_cols = 4 Dt', out_cols = Dt."""
    N: tuple
    out_start: tuple
    out_width: tuple
    K: tuple
    w_off: tuple
    k_off: tuple
    kmap: tuple
    in_cols: int
    out_cols: int


def summed_spec(entries, right_widths, left_sectors) -> SummedSpec:
    """The SummedSpec of a site: entries its SiteLayout's forward (start, stop, cols), right_widths its right bond's
    sector widths, left_sectors the left bond's (N, start, stop)."""
    N_of = {start: N for N, start, _ in left_sectors}
    d_right = int(sum(right_widths))
    right_start = np.concatenate(([0], np.cumsum(right_widths)))[:-1].astype(int)
    fields = dict(N=[], out_start=[], out_width=[], K=[], w_off=[], k_off=[])
    kmap, w_off = [], 0
    for start, stop, cols in entries:
        K = int(sum(width for *_, width in cols))
        fields["N"].append(int(N_of[start]))
        fields["out_start"].append(int(start))
        fields["out_width"].append(int(stop - start))
        fields["K"].append(K)
        fields["w_off"].append(w_off)
        fields["k_off"].append(len(kmap))
        for p, k, _, width in cols:  # in the order of their offsets: the block's columns
            kmap.extend(int(p * d_right + right_start[k] + j) for j in range(width))
        w_off += (stop - start) * K
    return SummedSpec(**{k: tuple(v) for k, v in fields.items()}, kmap=tuple(kmap), in_cols=4 * d_right,
                      out_cols=int(sum(stop - start for _, start, stop in left_sectors)))


_SPECS = {}  # id(layout) -> (layout, {kind: specs}): a SectorLayout holds a dict, so it is not hashable


def layout_specs(layout, direction, fused=False):
    """Per site the TrialSpecs (forward_specs, backward_specs) or with fused the FusedSpecs of a layout in a direction
    ("forward" or "backward"), or the SummedSpecs ("summed": the forward blocks as the walker-first right environments
    use them), made once per layout (the engine asks at every trace)."""
    cached = _SPECS.get(id(layout))
    if cached is None or cached[0] is not layout:
        cached = _SPECS[id(layout)] = (layout, {})
    key = (direction, bool(fused))
    if key not in cached[1] and direction == "summed":
        cached[1][key] = tuple(summed_spec(site.forward, site.right_widths, layout.sectors[i])
                               for i, site in enumerate(layout.sites))
    elif key not in cached[1]:
        if direction == "forward":
            args = [(site.forward, site.right_widths, layout.sectors[i], layout.sectors[i + 1])
                    for i, site in enumerate(layout.sites)]
        else:
            args = [(site.backward, site.left_widths, layout.sectors[i + 1], layout.sectors[i])
                    for i, site in enumerate(layout.sites)]
        cached[1][key] = tuple(fused_spec(*a) if fused else trial_spec(*a[:3]) for a in args)
    return cached[1][key]


def pack_blocks(matrices):
    """A site's sector blocks (sector_trial_blocks' forward or backward matrices, (width, ncols) each, in the order of
    the spec's sectors) as one flat array: block s at w_off[s], row-major."""
    return jnp.concatenate([jnp.ravel(W) for W in matrices])


def mpo_channel_offsets(gathers) -> list:
    """Per bond of mpo_contract's environments (a, b, w, t) the electron-number offset of each channel w, in the
    environments' channel order: E[a, b, w, t] can be nonzero only if N(t) = n_a + n_b + offset[w]. From the sites'
    MpoGather (dense_engine.mpo_gathers): output channel j at walker state p takes the columns 4 w + q, so
    offset_j = offset_w + n(q) - n(p), the same for all its terms (raises otherwise; a channel without terms is zero
    and gets 0). channel_offsets gives the same inside traced code."""
    n = np.asarray(N_PHYSICAL)
    offsets = [np.zeros(1, np.int64)]
    for site, gather in enumerate(gathers):
        single_src, single_coef, multi_src, multi_coef = (np.asarray(a) for a in gather)
        n1, out = single_src.shape[-1], []
        for j in range(n1 + multi_src.shape[-1]):
            if j < n1:
                terms = [(single_src[p, j], p) for p in range(4) if single_coef[p, j] != 0]
            else:
                terms = [(multi_src[t, p, j - n1], p) for t in range(multi_src.shape[0]) for p in range(4)
                         if multi_coef[t, p, j - n1] != 0]
            values = {int(offsets[-1][src // 4] + n[src % 4] - n[p]) for src, p in terms}
            if len(values) > 1:
                raise ValueError(f"site {site}, channel {j}: terms with different electron-number offsets {values}")
            out.append(values.pop() if values else 0)
        offsets.append(np.asarray(out, np.int64))
    return offsets


def channel_offsets(gather, previous):
    """One step of mpo_channel_offsets in traced code: the electron-number offsets of the channels after a site, from
    its MpoGather (device arrays) and the offsets of the channels before it, each channel's from its first term (a
    channel without terms gets previous[0] + 0: it is zero everywhere)."""
    n = jnp.asarray(N_PHYSICAL, jnp.int32)
    previous = jnp.asarray(previous, jnp.int32)
    out = []
    for src, coef, n_out in ((gather.single_src, gather.single_coef, gather.single_src.shape[-1]),
                             (gather.multi_src, gather.multi_coef, gather.multi_src.shape[-1])):
        if not n_out:
            continue
        src, coef = src.reshape(-1, n_out), coef.reshape(-1, n_out)  # rows (term, p) as t * 4 + p, or p alone
        first = jnp.argmax(coef != 0, axis=0)
        source = jnp.take_along_axis(src, first[None], axis=0)[0]
        out.append(previous[source // 4] + n[source % 4] - n[first % 4])
    return jnp.concatenate(out)


def row_sectors(n_a, n_b, spec: TrialSpec, offsets=None):
    """Per row of X = E.reshape(Da * Db [* Dw], Dt) the index of its sector in spec, or G = len(spec.N) for a row
    whose electron number is no sector's (it is zero in every column). n_a, n_b: the labels of the walker's alpha and
    beta bond states (the converter's charges: floats holding integers); offsets: the channels' electron-number
    offsets (mpo_channel_offsets) for the MPO environments, None for the overlap's."""
    n = jnp.round(n_a).astype(jnp.int32)[:, None] + jnp.round(n_b).astype(jnp.int32)[None, :]
    if offsets is not None:
        n = n[:, :, None] + jnp.asarray(offsets, jnp.int32)[None, None, :]
    n = n.reshape(-1)
    G = len(spec.N)
    if not G:
        return jnp.zeros(n.shape, jnp.int32)
    lo, hi = min(spec.N), max(spec.N)
    table = np.full(hi - lo + 1, G, np.int32)
    table[np.asarray(spec.N) - lo] = np.arange(G)
    inside = (n >= lo) & (n <= hi)
    return jnp.where(inside, jnp.asarray(table)[jnp.clip(n - lo, 0, hi - lo)], G)


def tile_tables(g, n_sectors, R):
    """The kernel's schedule for rows labelled g (rows,) with n_sectors sectors (label n_sectors: none): perm, the rows
    sorted by sector (the unmatched ones last), and per tile its sector, its first position in perm and its number of
    rows (0: an idle tile). Sector s gets ceil(c_s / R) tiles, so there are at most ceil(rows / R) + n_sectors of
    them, whatever the counts c_s: that is the static number of tiles (the launch grid)."""
    rows = g.shape[0]
    counts = jnp.bincount(g, length=n_sectors + 1)[:n_sectors]
    perm = jnp.argsort(g, stable=True)
    first = jnp.cumsum(counts) - counts
    tiles = (counts + R - 1) // R
    tile_end = jnp.cumsum(tiles)
    t = jnp.arange(-(-rows // R) + n_sectors)
    sector = jnp.minimum(jnp.searchsorted(tile_end, t, side="right"), n_sectors - 1)
    local = t - (tile_end - tiles)[sector]
    n_rows = jnp.where(t < tile_end[-1], jnp.clip(counts[sector] - local * R, 0, R), 0)
    return perm, sector, first[sector] + local * R, n_rows


def _kernel(tile_sector, tile_row0, tile_rows, perm, start, width, ncols, w_off, c_off, x, w, cmap, z_in, z, *,
            R, BK, BN, in_cols, out_cols, trash, presorted):
    """One program: tile t = program_id(0), up to R rows of one sector s (consecutive in perm), times that sector's
    block, output columns [BN c, BN (c + 1)) for c = program_id(1), written into z (the flat (rows, out_cols) output,
    whatever z_in held where nothing is written) at the rows' own places. Every array is flat and read through computed
    indices: x[row * in_cols + start_s + k] (the rows gathered by perm; presorted: x's rows are already in perm's
    order, so the tile's rows are its positions), w[w_off_s + k * ncols_s + j], and the output columns through cmap.
    Masked lanes (rows past the tile's count, columns past the block's) touch nothing; their indices are kept in range
    for the interpreter, which writes them to the extra element `trash`."""
    del z_in  # what z starts from (aliased to z)
    t, c = pl.program_id(0), pl.program_id(1)
    n_rows, s = tile_rows[t], tile_sector[t]
    n_cols = ncols[s]

    @pl.when((n_rows > 0) & (c * BN < n_cols))
    def _():
        r = jnp.arange(R)
        row_ok = r < n_rows
        position = jnp.where(row_ok, tile_row0[t] + r, 0)
        rows = position if presorted else perm[position]  # the tile's rows of x and of z
        j = c * BN + jnp.arange(BN)
        col_ok = j < n_cols
        x0, k_end, w0 = start[s], width[s], w_off[s]

        def step(kb, acc):
            k = kb * BK + jnp.arange(BK)
            k_ok = k < k_end
            x_ok = row_ok[:, None] & k_ok[None, :]
            xa = plgpu.load(x.at[jnp.where(x_ok, rows[:, None] * in_cols + x0 + k[None, :], 0)], mask=x_ok, other=0.0)
            w_ok = k_ok[:, None] & col_ok[None, :]
            wb = plgpu.load(w.at[jnp.where(w_ok, w0 + k[:, None] * n_cols + j[None, :], 0)], mask=w_ok, other=0.0)
            return acc + jnp.dot(xa, wb, preferred_element_type=acc.dtype)

        acc = lax.fori_loop(0, lax.div(k_end + BK - 1, jnp.asarray(BK, k_end.dtype)), step,
                            jnp.zeros((R, BN), z.dtype))
        dest = cmap[jnp.where(col_ok, c_off[s] + j, 0)]
        z_ok = row_ok[:, None] & col_ok[None, :]
        plgpu.store(z.at[jnp.where(z_ok, rows[:, None] * out_cols + dest[None, :], trash)], acc, mask=z_ok)


def _zeros_after(x, size):
    """A zero vector of `size` that XLA cannot build before x exists: 0 * x[0] (x[0] replaced by 0 unless finite),
    broadcast. A plain jnp.zeros has no operands, so XLA may schedule it anywhere: in the energy sweep of the
    walker-start probe it built every site's (rows, 4 D') output buffer at the start, 436 GB on the 8x8 at chi_T 2048
    (job 7372844). 0 * x[0] is not folded away for floats. The finite guard matters: x can be an array whose entries
    outside the rows' sectors were never written (label_contract's intermediates), so x[0] may hold anything; the
    interpreter fills such memory with NaN, which made the block contractions' zeros NaN (job 7374739)."""
    x0 = x.reshape(-1)[0]
    return jnp.broadcast_to(jnp.zeros((), x.dtype) * jnp.where(jnp.isfinite(x0), x0, jnp.zeros((), x.dtype)), (size,))


def grouped_trial_apply(X, g, spec: TrialSpec, packed, *, R=None, BK=None, BN=None, num_warps=None, num_stages=None,
                        interpret=None, init=None, presorted=False):
    """_trial_apply(X, matrices, entries, out_widths) on the rows that can be nonzero: X (rows, Dt), g (rows,) the
    rows' sectors (row_sectors), spec and packed (= pack_blocks(matrices)) the site's. Returns (rows, 4, D') like
    _trial_apply. R, BK, BN: the tile, num_warps, num_stages: Triton's (each not given from tile_for of the row count);
    interpret: Pallas' interpreter (default: off on a GPU, on elsewhere). init: an array of the output's size to write
    the rows of g's sectors into (its other entries are kept; default zeros). presorted: X's rows are already in sector
    order (X = X0[perm], perm from tile_tables(g, ...), g still X0's labels); the output then comes in that order too.

    vmap it over walkers: without init and presorted the walkers' rows are pooled, one launch multiplying the rows
    (walker, row) of all of them (_pooled, a custom_vmap rule; spec and packed shared, the tile from the pooled row
    count), so a sector's rows of every walker fill the same tiles. The propagation's walkers have a few dozen rows per
    sector each: in tiles of their own the grouped propagation was 8% slower than the dense GEMMs (job 7375440). Each
    row's sums are the ones it gets alone; with another tile only their order changes (rounding)."""
    options = dict(R=R, BK=BK, BN=BN, num_warps=num_warps, num_stages=num_stages, interpret=interpret)
    if init is not None or presorted:
        return _grouped_apply(X, g, spec, packed, init=init, presorted=presorted, **options)
    return _pooled(lambda X, g, packed: _grouped_apply(X, g, spec, packed, **options))(X, g, packed)


def _pooled(impl):
    """impl(X, g, packed) (a kernel's launch on rows whose first axis X shares with g) whose vmap pools the walkers:
    X (walkers, n, ...) and g (walkers, rows) go through one call as (walkers n, ...) and (walkers rows,) (pooled
    again under a further vmap), the output (walkers rows, ...) split back; a batched packed (every walker its own
    blocks) keeps the kernel's own batching (a grid axis per walker)."""

    @jax.custom_batching.custom_vmap
    def apply(X, g, packed):
        return impl(X, g, packed)

    @apply.def_vmap
    def pool(axis_size, in_batched, X, g, packed):
        x_batched, g_batched, packed_batched = in_batched
        if packed_batched:
            return jax.vmap(impl, in_axes=(0 if x_batched else None, 0 if g_batched else None, 0))(X, g, packed), True
        X = X if x_batched else jnp.broadcast_to(X, (axis_size,) + X.shape)
        g = g if g_batched else jnp.broadcast_to(g, (axis_size,) + g.shape)
        out = apply(X.reshape((-1,) + X.shape[2:]), g.reshape(-1), packed)
        return out.reshape((axis_size, -1) + out.shape[1:]), True

    return apply


def _grouped_apply(X, g, spec: TrialSpec, packed, *, R=None, BK=None, BN=None, num_warps=None, num_stages=None,
                   interpret=None, init=None, presorted=False):
    """grouped_trial_apply on one batch of rows (the kernel's launch)."""
    rows, in_cols = X.shape
    tile = _tile_options(rows, "grouped", R, BK, BN, num_warps, num_stages)
    R, BK, BN, num_warps, num_stages = (tile[k] for k in ("R", "BK", "BN", "num_warps", "num_stages"))
    if in_cols != spec.in_cols:
        raise ValueError(f"X has {in_cols} columns, the spec's bond {spec.in_cols}")
    size = rows * spec.out_cols
    G = len(spec.N)
    if not G or not X.size:
        base = jnp.zeros(size, X.dtype) if init is None else jnp.reshape(init, (size,)).astype(X.dtype)
        return base.reshape(rows, 4, spec.out_cols // 4)
    base = _zeros_after(X, size) if init is None else jnp.reshape(init, (size,)).astype(X.dtype)
    interpret = jax.default_backend() != "gpu" if interpret is None else bool(interpret)
    extra = 1 if interpret else 0  # the interpreter also writes masked lanes: to one extra element
    if extra:
        base = jnp.concatenate([base, jnp.zeros(1, X.dtype)])
    perm, tile_sector, tile_row0, tile_rows = tile_tables(g, G, R)
    tables = [jnp.asarray(np.asarray(v, np.int64)) for v in (spec.start, spec.width, spec.ncols, spec.w_off,
                                                             spec.c_off)]
    args = (tile_sector, tile_row0, tile_rows, perm, *tables, X.reshape(-1), packed,
            jnp.asarray(np.asarray(spec.cmap, np.int64)), base)
    kernel = partial(_kernel, R=R, BK=BK, BN=BN, in_cols=in_cols, out_cols=spec.out_cols, trash=size if extra else 0,
                     presorted=bool(presorted))
    options = {} if interpret else dict(compiler_params=plgpu.CompilerParams(num_warps=num_warps,
                                                                             num_stages=num_stages))
    z = pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((size + extra,), X.dtype),
                       grid=(tile_rows.shape[0], -(-max(spec.ncols) // BN)), input_output_aliases={len(args) - 1: 0},
                       interpret=interpret, name="grouped_trial_gemm", **options)(*args)
    return (z[:size] if extra else z).reshape(rows, 4, spec.out_cols // 4)


# ============================================================================
# The walker-first right environments: the trial GEMM with the sum over the physical index inside
# ============================================================================
#
# dense_engine._backward_summed contracts R (c, d, t') with the walker tensors first, X (a, n_alpha, b, n_beta, t'),
# and then, per left sector N, one GEMM of X's rows (a, b) and the columns (p, t') of the right sectors N + n(p) side
# by side with the forward block's transpose: R'[a, b, t] for t in N. X[a, ., b, ., t'] is zero outside the right
# sectors n_a + n_b + n(p), so a row (a, b) has a nonzero product only with the left sector of n_a + n_b (the walker's
# labels on the left bond). summed_trial_apply multiplies each row by that sector's block alone, reading X in its own
# layout (no transpose) through the column map kmap; tiles as grouped_trial_apply's, the walkers pooled under vmap.

def _summed_kernel(tile_sector, tile_row0, tile_rows, perm, out_start, out_width, K, w_off, k_off, x, w, kmap, z_in, z,
                   *, R, BK, BN, D_low, S_high, S_low, S_alpha, S_beta, d_right, out_cols, trash):
    """One program: tile t = program_id(0), up to R rows (a, b) of one left sector s (consecutive in perm), output
    columns [BN c, BN (c + 1)) of the sector (c = program_id(1)). Row r = a D_low + b (a absorbs the pooled walkers)
    starts at a S_high + b S_low in x; inner index k of the sector is column kmap[k_off + k] = p d_right + t', at
    (p % 2) S_alpha + (p // 2) S_beta + t' from the row's start (x is (a, n_alpha, b, n_beta, t')). The block is read
    transposed, w[w_off + k out_width + j]; the result goes to z[r, out_start + j]."""
    del z_in  # the zeros z starts from (aliased to z)
    t, c = pl.program_id(0), pl.program_id(1)
    n_rows, s = tile_rows[t], tile_sector[t]
    width = out_width[s]

    @pl.when((n_rows > 0) & (c * BN < width))
    def _():
        r = jnp.arange(R)
        row_ok = r < n_rows
        rows = perm[jnp.where(row_ok, tile_row0[t] + r, 0)]
        base = _div(rows, D_low) * S_high + _rem(rows, D_low) * S_low
        j = c * BN + jnp.arange(BN)
        col_ok = j < width
        k_end, w0, m0 = K[s], w_off[s], k_off[s]

        def step(kb, acc):
            k = kb * BK + jnp.arange(BK)
            k_ok = k < k_end
            column = kmap[jnp.where(k_ok, m0 + k, 0)]
            p, tp = _div(column, d_right), _rem(column, d_right)
            offset = _rem(p, 2) * S_alpha + _div(p, 2) * S_beta + tp
            x_ok = row_ok[:, None] & k_ok[None, :]
            xa = plgpu.load(x.at[jnp.where(x_ok, base[:, None] + offset[None, :], 0)], mask=x_ok, other=0.0)
            w_ok = k_ok[:, None] & col_ok[None, :]
            wb = plgpu.load(w.at[jnp.where(w_ok, w0 + k[:, None] * width + j[None, :], 0)], mask=w_ok, other=0.0)
            return acc + jnp.dot(xa, wb, preferred_element_type=acc.dtype)

        acc = lax.fori_loop(0, lax.div(k_end + BK - 1, jnp.asarray(BK, k_end.dtype)), step,
                            jnp.zeros((R, BN), z.dtype))
        z_ok = row_ok[:, None] & col_ok[None, :]
        plgpu.store(z.at[jnp.where(z_ok, rows[:, None] * out_cols + out_start[s] + j[None, :], trash)], acc, mask=z_ok)


def summed_trial_apply(X, g, sspec: SummedSpec, packed, *, R=None, BK=None, BN=None, num_warps=None, num_stages=None,
                       interpret=None):
    """The walker-first right environment's trial GEMM on the rows that can be nonzero: X (Da, 2, Db, 2, Dt') as
    _backward_summed's (a, n_alpha, b, n_beta, t'), g (Da Db,) the rows' left sectors (row_sectors(n_a, n_b, sspec),
    labels of the left bond), packed = pack_blocks([W.T for W in the site's forward blocks]). Returns (Da Db, Dt), the
    rows outside every sector zero: _backward_summed's result, each row's sum in another order. Tile options as
    grouped_trial_apply's (those not given from tile_for(rows, "summed")); vmap it over walkers (pooled)."""
    options = dict(R=R, BK=BK, BN=BN, num_warps=num_warps, num_stages=num_stages, interpret=interpret)
    return _pooled(lambda X, g, packed: _summed_apply(X, g, sspec, packed, **options))(X, g, packed)


def _summed_apply(X, g, sspec: SummedSpec, packed, *, R=None, BK=None, BN=None, num_warps=None, num_stages=None,
                  interpret=None):
    """summed_trial_apply on one batch of rows (the kernel's launch)."""
    Da, _, Db, _, d_right = X.shape
    rows = Da * Db
    if 4 * d_right != sspec.in_cols:
        raise ValueError(f"X has right bond {d_right}, the spec's {sspec.in_cols // 4}")
    tile = _tile_options(rows, "summed", R, BK, BN, num_warps, num_stages)
    R, BK, BN, num_warps, num_stages = (tile[k] for k in ("R", "BK", "BN", "num_warps", "num_stages"))
    size = rows * sspec.out_cols
    if not sspec.N or not sspec.kmap or not X.size:
        return jnp.zeros((rows, sspec.out_cols), X.dtype)
    base = _zeros_after(X, size)
    interpret = jax.default_backend() != "gpu" if interpret is None else bool(interpret)
    extra = 1 if interpret else 0
    if extra:
        base = jnp.concatenate([base, jnp.zeros(1, X.dtype)])
    perm, tile_sector, tile_row0, tile_rows = tile_tables(g, len(sspec.N), R)
    tables = [jnp.asarray(np.asarray(v, np.int64)) for v in (sspec.out_start, sspec.out_width, sspec.K, sspec.w_off,
                                                             sspec.k_off)]
    args = (tile_sector, tile_row0, tile_rows, perm, *tables, X.reshape(-1), packed,
            jnp.asarray(np.asarray(sspec.kmap, np.int64)), base)
    kernel = partial(_summed_kernel, R=R, BK=BK, BN=BN, D_low=Db, S_high=2 * Db * 2 * d_right, S_low=2 * d_right,
                     S_alpha=Db * 2 * d_right, S_beta=d_right, d_right=d_right, out_cols=sspec.out_cols,
                     trash=size if extra else 0)
    options = {} if interpret else dict(compiler_params=plgpu.CompilerParams(num_warps=num_warps,
                                                                             num_stages=num_stages))
    z = pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((size + extra,), X.dtype),
                       grid=(tile_rows.shape[0], -(-max(sspec.out_width) // BN)), input_output_aliases={len(args) - 1: 0},
                       interpret=interpret, name="summed_trial_gemm", **options)(*args)
    return (z[:size] if extra else z).reshape(rows, sspec.out_cols)


def _pieces(spec: TrialSpec, s):
    """Sector s's block columns as runs (lo, hi, dest): columns [lo, hi) go to output columns [dest, dest + hi - lo)
    (one run per physical state's target sector)."""
    dest = np.asarray(spec.cmap[spec.c_off[s]:spec.c_off[s] + spec.ncols[s]])
    edges = np.concatenate(([0], np.flatnonzero(np.diff(dest) != 1) + 1, [len(dest)]))
    return [(int(lo), int(hi), int(dest[lo])) for lo, hi in zip(edges[:-1], edges[1:])]


def hybrid_trial_apply(X, g, spec: TrialSpec, matrices, packed, caps, **kernel):
    """grouped_trial_apply with each sector's first caps[s] rows through XLA's GEMM (cuBLAS on a GPU): gathered into
    a (caps[s], width) operand (rows past the walker's count read as zeros), multiplied by the block, and written back
    at their rows (the padding's writes dropped). A walker with more rows in a sector than its cap sends the rest
    through the grouped kernel (grouped_trial_apply, written into the same output; its launch does nothing when no
    sector overflows). So caps decide only which path a row takes, never the result. kernel: grouped_trial_apply's
    tile and Triton options."""
    rows, G = X.shape[0], len(spec.N)
    caps = tuple(int(c) for c in caps)
    if len(caps) != G:
        raise ValueError(f"{len(caps)} caps for {G} sectors")
    counts = jnp.bincount(g, length=G + 1)[:G]
    perm = jnp.argsort(g, stable=True)
    first = jnp.cumsum(counts) - counts
    out = _zeros_after(X, rows * spec.out_cols).reshape(rows, spec.out_cols)
    for s in range(G):
        if not caps[s]:
            continue
        j = jnp.arange(caps[s])
        r = jnp.where(j < counts[s], perm[jnp.minimum(first[s] + j, rows - 1)], rows)  # past the count: out of range
        A = X.at[r, spec.start[s]:spec.start[s] + spec.width[s]].get(mode="fill", fill_value=0.0)
        Y = A @ matrices[s]
        for lo, hi, dest in _pieces(spec, s):
            out = out.at[r, dest:dest + hi - lo].set(Y[:, lo:hi], mode="drop")
    position = jnp.zeros(rows, perm.dtype).at[perm].set(jnp.arange(rows, dtype=perm.dtype))
    sector = jnp.minimum(g, G - 1)
    rest = jnp.where((g < G) & (position - first[sector] >= jnp.asarray(caps)[sector]), g, G)
    return grouped_trial_apply(X, rest, spec, packed, init=out, **kernel)


# ============================================================================
# Stage 2: the trial GEMM fused with what follows it (the MPO gather, the szsz weights, the sign, the layout)
# ============================================================================
#
# _forward_mpo_gather writes Z = the trial GEMM's (rows (a, b, w), (q, t')) and then, in a separate pass over the dense,
# mostly-zero array, makes each output (p, v) from its MPO terms (w, q, coefficient), applies the fermionic sign
# signs[b, n_alpha] and transposes to (a, n_alpha, b, n_beta, v, t') for the walker GEMMs; the szsz transfers do the
# same with weights. fused_trial_apply computes that final array directly: one program per (tile of walker pairs (a, b)
# with one n_a + n_b, output (p, v), block of output columns) sums, over the output's terms, the coefficient times the
# rows (a, b, w) of X in the term's input sector times the trial block's columns of q, and stores with the sign at the
# rows' places in the walker GEMMs' layout. Z is never written.


def mpo_terms(gather):
    """Per output (p, v) of a site's MpoGather (device arrays) its terms for fused_trial_apply: (count (4 Dv,),
    src (4 Dv Kmax,), coef (4 Dv Kmax,), Kmax) with pv = p Dv + v, src = 4 w + q, the nonzero terms first; the outputs
    v in the gather's channel order (single, then multi), as _forward_mpo_gather's intermediate."""
    n1, n2 = gather.single_src.shape[-1], gather.multi_src.shape[-1]
    K = max(1, gather.multi_src.shape[0]) if n2 else 1
    src, coef = [], []
    if n1:
        src.append(jnp.zeros((K, 4, n1), jnp.int32).at[0].set(gather.single_src.astype(jnp.int32)))
        coef.append(jnp.zeros((K, 4, n1), gather.single_coef.dtype).at[0].set(gather.single_coef))
    if n2:
        src.append(gather.multi_src.astype(jnp.int32))
        coef.append(gather.multi_coef)
    src, coef = jnp.concatenate(src, axis=2), jnp.concatenate(coef, axis=2)  # (K, 4, Dv)
    order = jnp.argsort((coef == 0).astype(jnp.int32), axis=0, stable=True)  # the nonzero terms first
    src, coef = jnp.take_along_axis(src, order, axis=0), jnp.take_along_axis(coef, order, axis=0)
    count = jnp.sum(coef != 0, axis=0).astype(jnp.int32)
    return count.reshape(-1), src.transpose(1, 2, 0).reshape(-1), coef.transpose(1, 2, 0).reshape(-1), K


def weight_terms(members, weights):
    """The szsz transfers' terms for fused_trial_apply: output (p, v = m W + w) from (member m, q = p) with the
    coefficient weights[w, p] (W, 4); one term, none where the weight is 0 (S^z on the empty and the doubly occupied
    site)."""
    weights = jnp.asarray(weights)
    W = weights.shape[0]
    v, p = jnp.arange(members * W)[None, :], jnp.arange(4)[:, None]
    src = (4 * (v // W) + p).astype(jnp.int32)
    coef = weights[v % W, p]
    return (coef != 0).astype(jnp.int32).reshape(-1), src.reshape(-1), coef.reshape(-1), 1


def _div(a, b):
    return lax.div(a, jnp.asarray(b, a.dtype))  # non-negative integers: no floor_divide (sign fixes) in the kernel


def _rem(a, b):
    return lax.rem(a, jnp.asarray(b, a.dtype))


def _fused_kernel(tile_group, tile_row0, tile_rows, perm, off_in, off_out, term_count, term_src, term_coef, signs,
                  in_index, start, width, w_off, ncols, piece_off, out_index, out_start, out_width, x, w, *refs,
                  R, BK, BN, Dw, Dv, Kmax, d_first, d_second, in_cols, d_out, n_lo, n_index, out_lo, n_out_index,
                  layout, signed, fill, products, trash):
    """One program: tile t = program_id(0) of walker pairs (first, second) with one label sum M, output
    pv = program_id(1) = p Dv + v, output columns [BN c, BN (c + 1)) of its sector (c = program_id(2)). The output
    sector is that of M + off_out[v] + n(p) ("left": the next bond; "right" walks backwards: M + off_out[v] - n(p)),
    from the labels alone. For each of the output's terms (src = 4 w + q, coef): the rows (pair, w) of x in the input
    sector of M + off_in[w] times the columns of q's block in that sector, times coef, summed; then the sign
    signs[second, n_alpha] (signed) and the store at the output rows of `layout`: "left" (first, n_alpha, second,
    n_beta, v), "right" (n_beta, first, second, n_alpha, v), p = n_alpha + 2 n_beta. With fill (z starts as zeros,
    aliased) an output without terms is skipped; without, its rows are written as zeros, so that every row whose sector
    exists is written (the block contractions read those and nothing else). (The sector used to come from the first
    term's target, which skipped multi-term outputs whose first term had no input sector: jobs 7374600-01.)
    products: the GEMM is already done (sector_products): x is Z (rows, in_cols = ldz), w the rows' positions in Z, and
    a term reads Z[w[row], q's columns] instead of multiplying."""
    z = refs[-1]  # with fill refs = (z_in, z), z_in the aliased zeros
    t, pv, c = pl.program_id(0), pl.program_id(1), pl.program_id(2)
    n_rows, count, M = tile_rows[t], term_count[pv], tile_group[t]
    p, v = _div(pv, Dv), _rem(pv, Dv)
    i, jb = _rem(p, 2), _div(p, 2)

    def sector(src):  # the input sector of a term for this tile's rows, -1 for none
        N = M + off_in[_div(src, 4)] - n_lo
        return jnp.where((N >= 0) & (N < n_index), in_index[jnp.clip(N, 0, n_index - 1)], -1)

    N_out = M + off_out[v] + (i + jb if layout == "left" else -(i + jb)) - out_lo
    k_out = jnp.where((N_out >= 0) & (N_out < n_out_index), out_index[jnp.clip(N_out, 0, n_out_index - 1)], -1)
    w_out = out_width[jnp.maximum(k_out, 0)]
    active = (n_rows > 0) & (k_out >= 0) & (c * BN < w_out)

    @pl.when((active & (count > 0)) if fill else active)
    def _():
        r = jnp.arange(R)
        row_ok = r < n_rows
        pair = perm[jnp.where(row_ok, tile_row0[t] + r, 0)]
        j = c * BN + jnp.arange(BN)
        col_ok = j < w_out

        def term(m, acc):
            src, coef = term_src[pv * Kmax + m], term_coef[pv * Kmax + m]
            s = sector(src)
            sv = jnp.maximum(s, 0)
            off = piece_off[sv * 4 + _rem(src, 4)]
            x0, k_end, w0, n_cols = start[sv], width[sv], w_off[sv], ncols[sv]
            rows = pair * Dw + _div(src, 4)
            if products:  # the row's product with its sector's block: Z[position, q's columns]
                z_ok = row_ok[:, None] & col_ok[None, :] & ((s >= 0) & (off >= 0))
                position = w[jnp.where(row_ok, rows, 0)]
                return acc + coef * plgpu.load(x.at[jnp.where(z_ok, position[:, None] * in_cols + off + j[None, :], 0)],
                                               mask=z_ok, other=0.0)

            def step(kb, part):
                k = kb * BK + jnp.arange(BK)
                k_ok = k < k_end
                x_ok = row_ok[:, None] & k_ok[None, :]
                xa = plgpu.load(x.at[jnp.where(x_ok, rows[:, None] * in_cols + x0 + k[None, :], 0)], mask=x_ok,
                                other=0.0)
                w_ok = k_ok[:, None] & col_ok[None, :]
                wb = plgpu.load(w.at[jnp.where(w_ok, w0 + k[:, None] * n_cols + off + j[None, :], 0)], mask=w_ok,
                                other=0.0)
                return part + jnp.dot(xa, wb, preferred_element_type=part.dtype)

            steps = jnp.where((s >= 0) & (off >= 0), lax.div(k_end + BK - 1, jnp.asarray(BK, k_end.dtype)), 0)
            return acc + coef * lax.fori_loop(0, steps, step, jnp.zeros((R, BN), z.dtype))

        acc = lax.fori_loop(0, count, term, jnp.zeros((R, BN), z.dtype))
        first, second = _div(pair, d_second), _rem(pair, d_second)
        if signed:
            acc = acc * signs[jnp.where(row_ok, second * 2 + i, 0)][:, None]
        if layout == "left":
            row = (((first * 2 + i) * d_second + second) * 2 + jb) * Dv + v
        else:
            row = (((jb * d_first + first) * d_second + second) * 2 + i) * Dv + v
        z_ok = row_ok[:, None] & col_ok[None, :]
        plgpu.store(z.at[jnp.where(z_ok, row[:, None] * d_out + out_start[k_out] + j[None, :], trash)], acc,
                    mask=z_ok)


def fused_trial_apply(X, n_first, n_second, offsets, terms, signs, fspec: FusedSpec, packed, *, n_groups,
                      layout="left", offsets_out=None, fill=True, R=None, BK=None, BN=None, num_warps=None,
                      num_stages=None, interpret=None):
    """The trial GEMM fused with its MPO gather / weights, sign and layout (see above): X (rows, Dt) with rows
    (first, second, w), n_first, n_second the walker bond labels of first and second, offsets the channels w's
    electron-number offsets (None: all 0), offsets_out the output channels v's (None: all 0), terms mpo_terms(gather)
    or weight_terms(members, weights), signs the (D_second, 2) sign factors or None, fspec and packed the site's
    (FusedSpec, pack_blocks of its blocks in the same direction), n_groups a bound on the label sums
    n_first + n_second (2 L + 1: labels count electrons). Returns layout "left" (D_first, 2, D_second, 2, Dv, d_out)
    or "right" (2, D_first, D_second, 2, Dv, d_out). fill: zeros wherever no term reaches; without, only the rows
    whose output sector exists are written, each in that sector's columns, and the rest of the array is undefined
    (for the block contractions, which read nothing else). Tile and Triton options as grouped_trial_apply (those not
    given from tile_for(rows, "fused")); vmap it over walkers (a grid axis per walker)."""
    if X.shape[1] != fspec.trial.in_cols:
        raise ValueError(f"X {X.shape} has not the spec's {fspec.trial.in_cols} columns")
    return _fused_call(X, packed, False, n_first, n_second, offsets, terms, signs, fspec, n_groups=n_groups,
                       layout=layout, offsets_out=offsets_out, fill=fill, R=R, BK=BK, BN=BN, num_warps=num_warps,
                       num_stages=num_stages, interpret=interpret)


def products_fused_apply(Z, positions, n_first, n_second, offsets, terms, signs, fspec: FusedSpec, *, n_groups,
                         layout="left", offsets_out=None, fill=True, R=None, BN=None, num_warps=None, num_stages=None,
                         interpret=None):
    """fused_trial_apply's work after the trial GEMM, from its products: Z (rows, ldz) as sector_products gives it
    (ldz at least the widest block), positions (rows,) each row's position in Z (rows (first, second, w) as X's; a row
    in no sector is never read). Same arguments otherwise and the same result. Tile options as fused_trial_apply's
    (those not given from tile_for(rows, "epilogue"); there is no inner dimension)."""
    if Z.shape[1] < max(fspec.trial.ncols, default=0) or positions.shape != Z.shape[:1]:
        raise ValueError(f"Z {Z.shape} and positions {positions.shape} do not hold the spec's products")
    return _fused_call(Z, positions, True, n_first, n_second, offsets, terms, signs, fspec, n_groups=n_groups,
                       layout=layout, offsets_out=offsets_out, fill=fill, R=R, BK=None, BN=BN, num_warps=num_warps,
                       num_stages=num_stages, interpret=interpret)


def _fused_call(X, w, products, n_first, n_second, offsets, terms, signs, fspec: FusedSpec, *, n_groups, layout,
                offsets_out, fill, R, BK, BN, num_warps, num_stages, interpret):
    """The fused kernel's launch: fused_trial_apply (products False: X the environment rows, w the packed blocks) or
    products_fused_apply (X = Z, w the rows' positions in it)."""
    if layout not in ("left", "right"):
        raise ValueError(f"layout must be 'left' or 'right', not {layout!r}")
    spec = fspec.trial
    d_first, d_second = n_first.shape[0], n_second.shape[0]
    rows, in_cols = X.shape
    tile = _tile_options(rows, "epilogue" if products else "fused", R, BK, BN, num_warps, num_stages)
    R, BK, BN, num_warps, num_stages = (tile[k] for k in ("R", "BK", "BN", "num_warps", "num_stages"))
    Dw = rows // (d_first * d_second)
    if Dw * d_first * d_second != rows:
        raise ValueError(f"X {X.shape} has not {d_first} x {d_second} x members rows")
    count, src, coef, Kmax = terms
    Dv = count.shape[0] // 4
    shape = ((d_first, 2, d_second, 2, Dv, fspec.d_out) if layout == "left"
             else (2, d_first, d_second, 2, Dv, fspec.d_out))
    size = math.prod(shape)
    if not spec.N or not X.size:
        return jnp.zeros(shape, X.dtype)
    interpret = jax.default_backend() != "gpu" if interpret is None else bool(interpret)
    extra = 1 if interpret else 0
    M = (jnp.round(n_first).astype(jnp.int32)[:, None] + jnp.round(n_second).astype(jnp.int32)[None, :]).reshape(-1)
    perm, group, row0, n_rows = tile_tables(jnp.where((M >= 0) & (M < n_groups), M, n_groups), n_groups, R)
    offsets = jnp.zeros(Dw, jnp.int32) if offsets is None else jnp.asarray(offsets, jnp.int32)
    offsets_out = jnp.zeros(Dv, jnp.int32) if offsets_out is None else jnp.asarray(offsets_out, jnp.int32)
    signed = signs is not None
    signs = jnp.reshape(signs, (-1,)).astype(X.dtype) if signed else jnp.zeros(1, X.dtype)
    static = [jnp.asarray(np.asarray(v, np.int64)) for v in (fspec.in_index, spec.start, spec.width, spec.w_off,
                                                             spec.ncols, fspec.piece_off, fspec.out_index,
                                                             fspec.out_start, fspec.out_width)]
    args = (group, row0, n_rows, perm, offsets, offsets_out, count, src, coef.astype(X.dtype), signs, *static,
            X.reshape(-1), w)
    aliases = {}
    if fill:
        base = _zeros_after(X, size)
        args += (jnp.concatenate([base, jnp.zeros(1, X.dtype)]) if extra else base,)
        aliases = {len(args) - 1: 0}
    kernel = partial(_fused_kernel, R=R, BK=BK, BN=BN, Dw=Dw, Dv=Dv, Kmax=Kmax, d_first=d_first, d_second=d_second,
                     in_cols=in_cols, d_out=fspec.d_out, n_lo=fspec.n_lo, n_index=len(fspec.in_index),
                     out_lo=fspec.out_lo, n_out_index=len(fspec.out_index), layout=layout, signed=signed,
                     fill=bool(fill), products=bool(products), trash=size if extra else 0)
    options = {} if interpret else dict(compiler_params=plgpu.CompilerParams(num_warps=num_warps,
                                                                             num_stages=num_stages))
    z = pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((size + extra,), X.dtype),
                       grid=(n_rows.shape[0], 4 * Dv, -(-max(fspec.out_width) // BN)), input_output_aliases=aliases,
                       interpret=interpret, name="fused_products" if products else "fused_trial_gemm", **options)(*args)
    return (z[:size] if extra else z).reshape(shape)


# ============================================================================
# The trial GEMM in cuBLAS: one DGEMM per sector on its rows, through an XLA FFI call (cublas_ext/)
# ============================================================================
#
# Triton's float64 GEMM runs at about half of cuBLAS's rate (job 7372537: 28-29 against 55 TFLOPS), and the fused
# kernel's trial GEMM is ~70% of the blocked energy and szsz steps (job 7388160: 10.5 of 14.6 ms, 5.3 of 7.9). XLA
# calls cuBLAS only with shapes known when it compiles, so the custom call cublas_ext/sector_products.cu takes the
# per-sector row counts (walker data) to the host, gathers each sector's rows into one packed operand and runs one
# cublasDgemm per sector into Z (rows, ldz): position i of the rows sorted by sector (perm) holds X[perm[i]]'s sector
# columns times its sector's block (ldz the widest block). products_fused_apply then does the rest of the fused kernel
# (MPO gather / weights, sign, layout) reading Z; cublas_fused_apply is the two together, a drop-in for
# fused_trial_apply (TROT_TRIAL_GEMM=cublas). The same products, summed in cuBLAS's order. The library is built by
# cublas_ext/build.sh into CUBLAS_LIB; off the GPU sector_products computes Z with XLA's dense GEMMs (the tests'
# reference).

CUBLAS_LIB = os.environ.get("TROT_CUBLAS_LIB",
                            str(Path(__file__).resolve().parent / "cublas_ext" / "libtrot_sector_products.so"))
_FFI_TARGET = "trot_sector_products"
_FFI = {}  # the loaded library once its target is registered


def cublas_available() -> bool:
    """Whether the cuBLAS sector products can run here: a GPU backend and the built library."""
    return jax.default_backend() == "gpu" and Path(CUBLAS_LIB).exists()


def _register_ffi():
    if _FFI:
        return
    if not Path(CUBLAS_LIB).exists():
        raise RuntimeError(f"the cuBLAS sector products need {CUBLAS_LIB}: build it first (sbatch "
                           "trot/gmps/cublas_ext/build.sh, see there), or set TROT_CUBLAS_LIB")
    library = ctypes.cdll.LoadLibrary(CUBLAS_LIB)
    jax.ffi.register_ffi_target(_FFI_TARGET, jax.ffi.pycapsule(library.TrotSectorProducts), platform="CUDA")
    _FFI["library"] = library


def sector_order(n_first, n_second, spec: TrialSpec, offsets, members):
    """The rows (first, second, member) by trial sector (row_sectors, offsets the members' or None): (perm, counts,
    positions), perm the rows sorted (stably, the rows in no sector last), counts (G,) each sector's rows, positions
    each row's place in perm."""
    g = row_sectors(n_first, n_second, spec, jnp.zeros(members, jnp.int32) if offsets is None else offsets)
    G = len(spec.N)
    perm = jnp.argsort(g, stable=True)
    counts = jnp.bincount(g, length=G + 1)[:G]
    positions = jnp.zeros(g.shape[0], perm.dtype).at[perm].set(jnp.arange(g.shape[0], dtype=perm.dtype))
    return perm, counts, positions


def sector_products(X, perm, counts, spec: TrialSpec, packed, use_ffi=None):
    """The trial GEMM's products by sector: X (rows, Dt), perm (rows,) the rows sorted by sector (stable, the rows in
    no sector last), counts (G,) the rows of each sector, spec and packed the site's (TrialSpec, pack_blocks). Returns
    Z (rows, ldz), ldz = max(spec.ncols): for position i of sector s's rows, Z[i, :ncols_s] = X[perm[i], start_s :
    start_s + width_s] @ W_s; every other entry undefined. use_ffi: the cuBLAS custom call (the default on a GPU; it
    needs the built library) or XLA's dense GEMMs (the default elsewhere). vmap: one call for all walkers (expand_dims).
    With "sequential" (a while loop of one call per walker) the 8x8 measurement asked XLA for a 90.75 GiB temp buffer
    (job 7390782, out of memory; the Triton path peaks at 62.8 GB, 7390781): the loop and its copies are the suspected
    cause, not confirmed."""
    ldz = max(spec.ncols, default=0)
    use_ffi = jax.default_backend() == "gpu" if use_ffi is None else bool(use_ffi)
    if not use_ffi or not spec.N:
        return _sector_products_reference(X, perm, counts, spec, packed, ldz)
    _register_ffi()
    table = lambda values: np.asarray(values, np.int64)
    call = jax.ffi.ffi_call(_FFI_TARGET, jax.ShapeDtypeStruct((X.shape[0], ldz), X.dtype), vmap_method="expand_dims")
    return call(X, perm.astype(jnp.int64), counts.astype(jnp.int64), packed, start=table(spec.start),
                width=table(spec.width), ncols=table(spec.ncols), w_off=table(spec.w_off),
                in_cols=np.int64(spec.in_cols), ldz=np.int64(ldz))


def _sector_products_reference(X, perm, counts, spec: TrialSpec, packed, ldz):
    """sector_products with XLA's GEMMs on every row (each sector's block times all rows, kept where the position is
    the sector's); the positions of no sector are zero."""
    rows = X.shape[0]
    sector = jnp.searchsorted(jnp.cumsum(counts), jnp.arange(rows), side="right")  # of each position of perm
    Xs = X[perm]
    Z = jnp.zeros((rows, ldz), X.dtype)
    for s, (start, width, ncols, w_off) in enumerate(zip(spec.start, spec.width, spec.ncols, spec.w_off)):
        W = packed[w_off:w_off + width * ncols].reshape(width, ncols)
        P = jnp.pad(Xs[:, start:start + width] @ W, ((0, 0), (0, ldz - ncols)))
        Z = jnp.where((sector == s)[:, None], P, Z)
    return Z


def cublas_fused_apply(X, n_first, n_second, offsets, terms, signs, fspec: FusedSpec, packed, *, n_groups,
                       layout="left", offsets_out=None, fill=True, use_ffi=None, R=None, BN=None, num_warps=None,
                       num_stages=None, interpret=None):
    """fused_trial_apply with the trial GEMM in cuBLAS (sector_products) and the rest in products_fused_apply: the
    same arguments and result (the products summed in cuBLAS's order). use_ffi: sector_products'. Tile options:
    products_fused_apply's."""
    spec = fspec.trial
    rows = X.shape[0]
    d_first, d_second = n_first.shape[0], n_second.shape[0]
    Dw = rows // (d_first * d_second)
    if X.shape[1] != spec.in_cols or Dw * d_first * d_second != rows:
        raise ValueError(f"X {X.shape} is not ({d_first} x {d_second} x members, {spec.in_cols})")
    if not spec.N or not X.size:
        return fused_trial_apply(X, n_first, n_second, offsets, terms, signs, fspec, packed, n_groups=n_groups,
                                 layout=layout, offsets_out=offsets_out, fill=fill, interpret=interpret)
    perm, counts, positions = sector_order(n_first, n_second, spec, offsets, Dw)
    Z = sector_products(X, perm, counts, spec, packed, use_ffi)
    return products_fused_apply(Z, positions, n_first, n_second, offsets, terms, signs, fspec, n_groups=n_groups,
                                layout=layout, offsets_out=offsets_out, fill=fill, R=R, BN=BN, num_warps=num_warps,
                                num_stages=num_stages, interpret=interpret)


# ============================================================================
# Stage 3: the walker contractions on the rows that can be nonzero (label blocks)
# ============================================================================
#
# After the fused trial GEMM, a site contracts its intermediate with the walker tensors: (a, n_alpha) into a' with the
# alpha tensor A[a, i, a'], then (b, n_beta) into b' with the beta tensor (the szsz right transfer does the mirror:
# beta first, then alpha with the sign). A[a, i, a'] vanishes unless n_a' = n_a + i, and every row of the intermediates
# has one sector of the trial bond (its labels plus the channel's offset), so an output row (a', rest) gets
# contributions only from the input rows (a, i, rest) with n_a + i = n_a', all in that one sector's columns.
# label_contract does exactly those: per (rest, block of columns), for each target label group, the walker tensor's
# block of the group's sources and targets times the sources' rows. Nothing else is read or written: the intermediates
# need no zero fill (fused_trial_apply with fill=False writes every row whose sector exists), only the site's output,
# which other code reads whole, is zero-filled.


def _label_groups(n_dst, n_src, sigma):
    """The target label groups of a contraction: targets x' grouped by their label (distinct values, at most D_x'
    groups), sources (x, s) by the group of n_src[x] + sigma s (sources reaching no target label are left out).
    Returns (n_groups, group label (D_x',), targets' perm (D_x',), first, count (D_x',), sources' perm (2 D_x,) as
    2 x + s, first, count (D_x',))."""
    n_dst = jnp.round(n_dst).astype(jnp.int32)
    n_src = jnp.round(n_src).astype(jnp.int32)
    D = n_dst.shape[0]
    perm = jnp.argsort(n_dst, stable=True)
    ordered = n_dst[perm]
    new = jnp.concatenate([jnp.ones(1, bool), ordered[1:] != ordered[:-1]])
    group_of_sorted = jnp.cumsum(new) - 1  # (D,): group index of each sorted target
    n_groups = group_of_sorted[-1] + 1
    labels = jnp.zeros(D, jnp.int32).at[group_of_sorted].set(ordered)  # each group's label value
    count = jnp.bincount(group_of_sorted, length=D)
    first = jnp.cumsum(count) - count
    value = (n_src[:, None] + sigma * jnp.arange(2)[None, :]).reshape(-1)  # source 2 x + s
    sorted_labels = jnp.where(jnp.arange(D) < n_groups, labels, jnp.iinfo(jnp.int32).max)  # unused groups last
    position = jnp.searchsorted(sorted_labels, value)
    found = (position < n_groups) & (labels[jnp.clip(position, 0, D - 1)] == value)
    src_group = jnp.where(found, position, D)
    src_count = jnp.bincount(src_group, length=D + 1)[:D]
    src_first = jnp.cumsum(src_count) - src_count
    return n_groups, labels, perm, first, count, jnp.argsort(src_group, stable=True), src_first, src_count


def _label_contract_kernel(n_groups, g_label, dst_perm, dst_first, dst_count, src_perm, src_first, src_count, lab1,
                           lab3, signs, out_index, out_start, out_width, x, w, *refs, R, BK, BN, D2, D3, c2,
                           in_strides, out_strides, w_strides, out_lo, n_out_index, sign_mode, keep_s, fill, trash):
    """One program: rest r = (r1, r2, r3) = program_id(0), columns [BN c, BN (c + 1)) of each row's sector
    (c = program_id(1)). For each target label group g (label L): the rows' sector is that of
    L + lab1[r1] + c2 r2 + lab3[r3] in the output bond; out[x', r, cols] = sum over the group's sources (x, s) of
    w[x, s, x'] x[x, s, r, cols] (keep_s: out[x', s, r, cols], each s apart), the targets R at a time and the sources
    BK at a time, with the sign of sign_mode: "rest_s" signs[r1, s] and "src_rest" signs[x, r2] on the source terms,
    "dst_rest" signs[x', r2] on the result. Strides: in (x, s, r1, r2, r3), out (x', s, r1, r2, r3) (s used with
    keep_s), w (x, s, x'); the columns are the last, contiguous index of x and out (offset out_start of the sector)."""
    out = refs[-1]
    S_x, S_s, S_1, S_2, S_3 = in_strides
    T_x, T_s, T_1, T_2, T_3 = out_strides
    U_x, U_s, U_d = w_strides
    r, c = pl.program_id(0), pl.program_id(1)
    r1, r23 = _div(r, D2 * D3), _rem(r, D2 * D3)
    r2, r3 = _div(r23, D3), _rem(r23, D3)
    rest = lab1[r1] + c2 * r2 + lab3[r3]
    in_base = r1 * S_1 + r2 * S_2 + r3 * S_3
    out_base = r1 * T_1 + r2 * T_2 + r3 * T_3

    def group(g, carry):
        N = g_label[g] + rest - out_lo
        k = jnp.where((N >= 0) & (N < n_out_index), out_index[jnp.clip(N, 0, n_out_index - 1)], -1)
        width = out_width[jnp.maximum(k, 0)]

        @pl.when((k >= 0) & (c * BN < width))
        def _():
            cols = c * BN + jnp.arange(BN)
            col_ok = cols < width
            col0 = out_start[k]
            n_dst, n_src, f_dst, f_src = dst_count[g], src_count[g], dst_first[g], src_first[g]

            def rows(rc, carry):
                rr = rc * R + jnp.arange(R)
                row_ok = rr < n_dst
                target = dst_perm[jnp.where(row_ok, f_dst + rr, 0)]
                n_acc = 2 if keep_s else 1

                def step(kc, accs):
                    kk = kc * BK + jnp.arange(BK)
                    k_ok = kk < n_src
                    source = src_perm[jnp.where(k_ok, f_src + kk, 0)]
                    xs, s = _div(source, 2), _rem(source, 2)
                    w_ok = k_ok[:, None] & row_ok[None, :]
                    ws = plgpu.load(w.at[jnp.where(w_ok, xs[:, None] * U_x + s[:, None] * U_s + target[None, :] * U_d,
                                                   0)], mask=w_ok, other=0.0)  # (BK, R)
                    if sign_mode == "rest_s":
                        ws = ws * signs[jnp.where(k_ok, r1 * 2 + s, 0)][:, None]
                    elif sign_mode == "src_rest":
                        ws = ws * signs[jnp.where(k_ok, xs * 2 + r2, 0)][:, None]
                    i_ok = k_ok[:, None] & col_ok[None, :]
                    xi = plgpu.load(x.at[jnp.where(i_ok, xs[:, None] * S_x + s[:, None] * S_s + in_base + col0
                                                   + cols[None, :], 0)], mask=i_ok, other=0.0)  # (BK, BN)
                    if not keep_s:
                        return (accs[0] + lax.dot_general(ws, xi, (((0,), (0,)), ((), ())),
                                                          preferred_element_type=accs[0].dtype),)
                    return tuple(accs[b] + lax.dot_general(jnp.where((s == b)[:, None], ws, 0.0), xi,
                                                           (((0,), (0,)), ((), ())),
                                                           preferred_element_type=accs[b].dtype) for b in range(2))

                accs = lax.fori_loop(0, lax.div(n_src + BK - 1, jnp.asarray(BK, n_src.dtype)), step,
                                     tuple(jnp.zeros((R, BN), out.dtype) for _ in range(n_acc)))
                z_ok = row_ok[:, None] & col_ok[None, :]
                for b, acc in enumerate(accs):
                    if sign_mode == "dst_rest":
                        acc = acc * signs[jnp.where(row_ok, target * 2 + r2, 0)][:, None]
                    where = target[:, None] * T_x + b * T_s + out_base + col0 + cols[None, :]
                    plgpu.store(out.at[jnp.where(z_ok, where, trash)], acc, mask=z_ok)
                return carry

            lax.fori_loop(0, lax.div(n_dst + R - 1, jnp.asarray(R, n_dst.dtype)), rows, 0)

        return carry

    lax.fori_loop(0, n_groups[0], group, 0)


def label_contract(IN, W, n_src, n_dst, sigma, rest, lab1, c2, lab3, signs, fspec: FusedSpec, *, in_strides,
                   out_strides, w_strides, out_size, fill, sign_mode=None, keep_s=False, R=None, BK=None, BN=None,
                   num_warps=None, num_stages=None, interpret=None):
    """One walker contraction on label blocks (see above), out[x', r1, r2, r3, t] =
    sum_{(x, s): n_src[x] + sigma s = n_dst[x']} W[x, s, x'] IN[x, s, r1, r2, r3, t] (keep_s: out[x', s, ...], each s
    apart; sign_mode as the kernel's, with signs (.., 2)) on the rows whose sector (n_dst[x'] + lab1[r1] + c2 r2 +
    lab3[r3], in fspec's output bond) exists, in its columns: IN and W flat with the given strides (t contiguous;
    out_strides (x', s, r1, r2, r3)), rest = (D1, D2, D3), out flat of out_size, zero-filled (fill) or undefined
    elsewhere. Rows of IN outside their sector's columns are never read. Tile: CONTRACT_TILE unless given."""
    if sign_mode not in (None, "rest_s", "src_rest", "dst_rest") or (sign_mode is None) != (signs is None):
        raise ValueError(f"sign_mode {sign_mode!r} with signs {'missing' if signs is None else 'given'}")
    tile = CONTRACT_TILE
    R, BK, BN = R or tile[0], BK or tile[1], BN or tile[2]
    num_warps, num_stages = num_warps or tile[3], num_stages or tile[4]
    D1, D2, D3 = rest
    interpret = jax.default_backend() != "gpu" if interpret is None else bool(interpret)
    extra = 1 if interpret else 0
    n_groups, g_label, dst_perm, dst_first, dst_count, src_perm, src_first, src_count = _label_groups(n_dst, n_src,
                                                                                                     sigma)
    static = [jnp.asarray(np.asarray(v, np.int64)) for v in (fspec.out_index, fspec.out_start, fspec.out_width)]
    args = (jnp.reshape(n_groups, (1,)), g_label, dst_perm, dst_first, dst_count, src_perm, src_first, src_count,
            jnp.round(jnp.asarray(lab1)).astype(jnp.int32), jnp.round(jnp.asarray(lab3)).astype(jnp.int32),
            jnp.zeros(1, IN.dtype) if signs is None else jnp.reshape(signs, (-1,)).astype(IN.dtype), *static,
            IN.reshape(-1), W.reshape(-1))
    aliases = {}
    if fill:
        base = _zeros_after(IN, out_size)
        args += (jnp.concatenate([base, jnp.zeros(1, IN.dtype)]) if extra else base,)
        aliases = {len(args) - 1: 0}
    kernel = partial(_label_contract_kernel, R=R, BK=BK, BN=BN, D2=D2, D3=D3, c2=int(c2), in_strides=in_strides,
                     out_strides=out_strides, w_strides=w_strides, out_lo=fspec.out_lo,
                     n_out_index=len(fspec.out_index), sign_mode=sign_mode, keep_s=bool(keep_s), fill=bool(fill),
                     trash=out_size if extra else 0)
    options = {} if interpret else dict(compiler_params=plgpu.CompilerParams(num_warps=num_warps,
                                                                             num_stages=num_stages))
    z = pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((out_size + extra,), IN.dtype),
                       grid=(D1 * D2 * D3, -(-max(fspec.out_width) // BN)), input_output_aliases=aliases,
                       interpret=interpret, name="label_contract", **options)(*args)
    return z[:out_size] if extra else z


def contract_left(Y, A, B, n_a, n_b, n_a2, n_b2, offsets_out, fspec: FusedSpec, fill=True):
    """_forward_mpo_gather's and _left_transfer's walker GEMMs on label blocks: Y (Da, 2, Db, 2, Dv, Dt) as
    fused_trial_apply(..., layout="left", fill=False) writes it (signs applied), A (Da, 2, Da2), B (Db, 2, Db2) the
    undressed walker tensors, labels of a, b on the site's left bond and a2, b2 on its right, offsets_out the channels'
    (None: 0). Returns (Da2, Db2, Dv, Dt), zeros outside the rows' sectors, as tensordot(A, Y) and the beta
    dot_general give it; without fill those entries are undefined (for a next fused kernel, which reads nothing
    else)."""
    H = contract_left_alpha(Y, A, n_a, n_b, n_a2, offsets_out, fspec)
    return contract_left_beta(H, B, n_b, n_a2, n_b2, offsets_out, fspec, fill)


def contract_left_alpha(Y, A, n_a, n_b, n_a2, offsets_out, fspec: FusedSpec):
    """contract_left's alpha contraction H[a2, b, j, v, t] = sum_{a, i} A[a, i, a2] Y[a, i, b, j, v, t], defined on the
    rows of sector n_a2 + n_b + j + off[v] (the rest undefined): (Da2, Db, 2, Dv, Dt)."""
    _, _, Db, _, Dv, Dt = Y.shape
    Da2 = A.shape[2]
    off = jnp.zeros(Dv, jnp.int32) if offsets_out is None else jnp.asarray(offsets_out, jnp.int32)
    H = label_contract(Y, A, n_a, n_a2, 1, (Db, 2, Dv), n_b, 1, off, None, fspec,
                       in_strides=(2 * Db * 2 * Dv * Dt, Db * 2 * Dv * Dt, 2 * Dv * Dt, Dv * Dt, Dt),
                       out_strides=(Db * 2 * Dv * Dt, 0, 2 * Dv * Dt, Dv * Dt, Dt), w_strides=(2 * Da2, Da2, 1),
                       out_size=Da2 * Db * 2 * Dv * Dt, fill=False)
    return H.reshape(Da2, Db, 2, Dv, Dt)


def contract_left_beta(H, B, n_b, n_a2, n_b2, offsets_out, fspec: FusedSpec, fill=True):
    """contract_left's beta contraction E[a2, b2, v, t] = sum_{b, j} B[b, j, b2] H[a2, b, j, v, t] on the rows of
    sector n_b2 + n_a2 + off[v], zeros elsewhere (without fill undefined): (Da2, Db2, Dv, Dt)."""
    Da2, Db, _, Dv, Dt = H.shape
    Db2 = B.shape[2]
    off = jnp.zeros(Dv, jnp.int32) if offsets_out is None else jnp.asarray(offsets_out, jnp.int32)
    E = label_contract(H, B, n_b, n_b2, 1, (Da2, 1, Dv), n_a2, 0, off, None, fspec,
                       in_strides=(2 * Dv * Dt, Dv * Dt, Db * 2 * Dv * Dt, 0, Dt),
                       out_strides=(Dv * Dt, 0, Db2 * Dv * Dt, 0, Dt), w_strides=(2 * Db2, Db2, 1),
                       out_size=Da2 * Db2 * Dv * Dt, fill=fill)
    return E.reshape(Da2, Db2, Dv, Dt)


def contract_right(V, A, B, signs, n_a, n_b, n_a2, n_b2, fspec: FusedSpec):
    """_right_transfer's walker GEMMs on label blocks: V (2, Db2, Da2, 2, Dv, Dt) as fused_trial_apply(...,
    layout="right", fill=False) writes it ((n_beta, b', a', n_alpha, v, t), no sign), A (Da, 2, Da2), B (Db, 2, Db2)
    the undressed walker tensors, signs (Db, 2) the sign factors (they go on the alpha contraction), labels of a, b on
    the site's left bond and a2, b2 (a', b') on its right. Returns (Db, Da, Dv, Dt), zeros outside the rows' sectors,
    as the beta tensordot and the signed alpha dot_general give it."""
    _, Db2, Da2, _, Dv, Dt = V.shape
    Da, Db = A.shape[0], B.shape[0]
    zero = jnp.zeros(Dv, jnp.int32)
    # V2[b, a2, i, v, t] = sum_{b2, j} B[b, j, b2] V[j, b2, a2, i, v, t]: rows of sector n_b + n_a2 - i
    V2 = label_contract(V, B, n_b2, n_b, -1, (Da2, 2, Dv), n_a2, -1, zero, None, fspec,
                        in_strides=(Da2 * 2 * Dv * Dt, Db2 * Da2 * 2 * Dv * Dt, 2 * Dv * Dt, Dv * Dt, Dt),
                        out_strides=(Da2 * 2 * Dv * Dt, 0, 2 * Dv * Dt, Dv * Dt, Dt), w_strides=(1, Db2, 2 * Db2),
                        out_size=Db * Da2 * 2 * Dv * Dt, fill=False)
    # out[b, a, v, t] = sum_{a2, i} A[a, i, a2] signs[b, i] V2[b, a2, i, v, t]: rows of sector n_a + n_b
    out = label_contract(V2, A, n_a2, n_a, -1, (Db, 1, Dv), n_b, 0, zero, signs, fspec,
                         in_strides=(2 * Dv * Dt, Dv * Dt, Da2 * 2 * Dv * Dt, 0, Dt),
                         out_strides=(Dv * Dt, 0, Da * Dv * Dt, 0, Dt), w_strides=(1, Da2, 2 * Da2),
                         out_size=Db * Da * Dv * Dt, fill=True, sign_mode="rest_s")
    return out.reshape(Db, Da, Dv, Dt)


def unit_terms():
    """weight_terms of one member and unit weights: fused_trial_apply as the plain trial GEMM in its layouts. Made per
    call (a few constants): a cached copy made inside one trace would leak into the next."""
    return weight_terms(1, np.ones((1, 4)))


def forward_walker_blocks(Z, A, B, signs, n_a, n_b, n_c, n_d, fspec: FusedSpec):
    """_forward's walker einsums on label blocks (the propagation's field sweep): Z (Da, 2, Db, 2, 1, Dt') as
    fused_trial_apply(..., unit_terms(), layout="left", fill=False) writes it, (a, n_alpha, b, n_beta, t'); A (Da, 2,
    Dc), B (Db, 2, Dd) the undressed walker tensors, signs (Db, 2) of b's labels; labels a, b on the site's left bond,
    c, d on its right. Returns (Dc, Dd, 4, Dt') with p = n_alpha + 2 n_beta kept, zeros outside the rows' sectors, as
    _forward's two einsums give it."""
    Da, _, Db, _, _, Dt = Z.shape
    Dc, Dd = A.shape[2], B.shape[2]
    zero = jnp.zeros(1, jnp.int32)
    # H[c, i, b, j, t] = sum_a A[a, i, c] Z[a, i, b, j, t], i kept: rows of sector n_c + n_b + j
    H = label_contract(Z, A, n_a, n_c, 1, (Db, 2, 1), n_b, 1, zero, None, fspec,
                       in_strides=(2 * Db * 2 * Dt, Db * 2 * Dt, 2 * Dt, Dt, 0),
                       out_strides=(2 * Db * 2 * Dt, Db * 2 * Dt, 2 * Dt, Dt, 0), w_strides=(2 * Dc, Dc, 1),
                       out_size=Dc * 2 * Db * 2 * Dt, fill=False, keep_s=True)
    # Y[c, d, j, i, t] = sum_b B[b, j, d] signs[b, i] H[c, i, b, j, t], j kept: rows of sector n_c + n_d
    Y = label_contract(H, B, n_b, n_d, 1, (Dc, 2, 1), n_c, 0, zero, signs, fspec,
                       in_strides=(2 * Dt, Dt, 2 * Db * 2 * Dt, Db * 2 * Dt, 0),
                       out_strides=(4 * Dt, 2 * Dt, Dd * 4 * Dt, Dt, 0), w_strides=(2 * Dd, Dd, 1),
                       out_size=Dc * Dd * 4 * Dt, fill=True, keep_s=True, sign_mode="src_rest")
    return Y.reshape(Dc, Dd, 4, Dt)


def backward_walker_blocks(U, A, B, signs, n_a, n_b, n_c, n_d, fspec: FusedSpec):
    """_backward's walker einsums on label blocks (the propagation's right environments): U (2, Dc, Dd, 2, 1, Dt) as
    fused_trial_apply(..., unit_terms(), layout="right", fill=False) writes it, (n_beta, c, d, n_alpha, t) with c, d
    the right bond's alpha and beta states; A (Da, 2, Dc), B (Db, 2, Dd) undressed, signs (Db, 2) of the left bond's
    beta labels; labels a, b on the left bond, c, d on the right. Returns (Da, Db, 4, Dt), p kept, zeros outside the
    rows' sectors, as _backward's two einsums give it."""
    _, Dc, Dd, _, _, Dt = U.shape
    Da, Db = A.shape[0], B.shape[0]
    zero = jnp.zeros(1, jnp.int32)
    # H[a, i, d, j, t] = sum_c A[a, i, c] U[j, c, d, i, t], i kept: rows of sector n_a + n_d - j
    H = label_contract(U, A, n_c, n_a, -1, (Dd, 2, 1), n_d, -1, zero, None, fspec,
                       in_strides=(Dd * 2 * Dt, Dt, 2 * Dt, Dc * Dd * 2 * Dt, 0),
                       out_strides=(2 * Dd * 2 * Dt, Dd * 2 * Dt, 2 * Dt, Dt, 0), w_strides=(1, Dc, 2 * Dc),
                       out_size=Da * 2 * Dd * 2 * Dt, fill=False, keep_s=True)
    # X[a, b, j, i, t] = sum_d B[b, j, d] signs[b, i] H[a, i, d, j, t], j kept: rows of sector n_a + n_b
    X = label_contract(H, B, n_d, n_b, -1, (Da, 2, 1), n_a, 0, zero, signs, fspec,
                       in_strides=(2 * Dt, Dt, 2 * Dd * 2 * Dt, Dd * 2 * Dt, 0),
                       out_strides=(4 * Dt, 2 * Dt, Db * 4 * Dt, Dt, 0), w_strides=(1, Dd, 2 * Dd),
                       out_size=Da * Db * 4 * Dt, fill=True, keep_s=True, sign_mode="dst_rest")
    return X.reshape(Da, Db, 4, Dt)


def _gemm_kernel(a, b, c, *, BK, K):
    def step(k, acc):
        return acc + jnp.dot(a[:, pl.ds(k * BK, BK)], b[pl.ds(k * BK, BK), :], preferred_element_type=acc.dtype)

    c[...] = lax.fori_loop(0, K // BK, step, jnp.zeros(c.shape, c.dtype))


def triton_gemm(A, B, *, BM=64, BN=64, BK=32, num_warps=4, num_stages=3, interpret=None):
    """A @ B as a plain Pallas GEMM on contiguous operands ((BM, BN) output tiles, BK per step; the shapes must be
    multiples of the tiles: pad first). A diagnostic of Triton's float64 GEMM rate without the grouped kernel's
    gathers and masks, to set against XLA's GEMM (cuBLAS) on the same operands."""
    (M, K), N = A.shape, B.shape[1]
    if M % BM or N % BN or K % BK:
        raise ValueError(f"({M}, {K}) @ ({K}, {N}) is not a multiple of the tile ({BM}, {BK}, {BN})")
    interpret = jax.default_backend() != "gpu" if interpret is None else bool(interpret)
    options = {} if interpret else dict(compiler_params=plgpu.CompilerParams(num_warps=num_warps,
                                                                             num_stages=num_stages))
    return pl.pallas_call(partial(_gemm_kernel, BK=BK, K=K), out_shape=jax.ShapeDtypeStruct((M, N), A.dtype),
                          grid=(M // BM, N // BN),
                          in_specs=[pl.BlockSpec((BM, K), lambda i, j: (i, 0)),
                                    pl.BlockSpec((K, BN), lambda i, j: (0, j))],
                          out_specs=pl.BlockSpec((BM, BN), lambda i, j: (i, j)),
                          interpret=interpret, name="triton_gemm", **options)(A, B)


# ============================================================================
# References and counts (NumPy)
# ============================================================================

def grouped_reference(X, g, spec: TrialSpec, matrices):
    """What the kernel computes, in NumPy, sector by sector: each sector's rows of X times its block, written into the
    zero (rows, 4 D') output at their own rows and the block columns' places (cmap)."""
    X, g = np.asarray(X), np.asarray(g)
    Z = np.zeros((X.shape[0], spec.out_cols), X.dtype)
    for s, W in enumerate(matrices):
        rows = np.flatnonzero(g == s)
        cols = np.asarray(spec.cmap[spec.c_off[s]:spec.c_off[s] + spec.ncols[s]])
        Z[np.ix_(rows, cols)] = X[rows, spec.start[s]:spec.start[s] + spec.width[s]] @ np.asarray(W)
    return Z.reshape(X.shape[0], 4, -1)


def multiply_adds(g, spec: TrialSpec, R=None) -> dict:
    """Multiply-adds of the trial GEMM for rows labelled g ((rows,) or (walkers, rows)): dense (_trial_apply: every
    row in every sector's GEMM), useful (each sector's own rows), tiled (those rows rounded up to whole tiles of R:
    what the kernel does)."""
    R = R or TILE_ROWS
    g = np.asarray(g)
    g = g.reshape(-1, g.shape[-1])
    per_row = np.asarray(spec.width) * np.asarray(spec.ncols)  # one row through each sector's GEMM
    counts = np.stack([np.bincount(x, minlength=len(spec.N) + 1)[:len(spec.N)] for x in g])
    return dict(dense=float(g.shape[1] * per_row.sum() * len(g)), useful=float((counts * per_row).sum()),
                tiled=float((np.ceil(counts / R) * R * per_row).sum()))
