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
Not used by the engine yet: the H100 benchmark (profile_6x6_engines.py --stages grouped) decides.
"""
from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

from trot.gmps.dense_engine import N_PHYSICAL

# Tile sizes of the GPU kernel: R rows of one sector, BN output columns, BK of the sector's columns per step. Powers of
# 2 (Triton); a float64 dot needs R >= 16, BN >= 8, BK >= 16.
TILE_ROWS, TILE_COLS, TILE_K = 64, 64, 32


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


def pack_blocks(matrices):
    """A site's sector blocks (sector_trial_blocks' forward or backward matrices, (width, ncols) each, in the order of
    the spec's sectors) as one flat array: block s at w_off[s], row-major."""
    return jnp.concatenate([jnp.ravel(W) for W in matrices])


def mpo_channel_offsets(gathers) -> list:
    """Per bond of mpo_contract's environments (a, b, w, t) the electron-number offset of each channel w, in the
    environments' channel order: E[a, b, w, t] can be nonzero only if N(t) = n_a + n_b + offset[w]. From the sites'
    MpoGather (dense_engine.mpo_gathers): output channel j at walker state p takes the columns 4 w + q, so
    offset_j = offset_w + n(q) - n(p), the same for all its terms (raises otherwise; a channel without terms is zero
    and gets 0)."""
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
            R, BK, BN, in_cols, out_cols, trash):
    """One program: tile t = program_id(0), up to R rows of one sector s (consecutive in perm), times that sector's
    block, output columns [BN c, BN (c + 1)) for c = program_id(1), written into z (the flat (rows, out_cols) output,
    zeros where nothing is written) at the rows' own places. Every array is flat and read through computed indices:
    x[row * in_cols + start_s + k] (the rows gathered by perm), w[w_off_s + k * ncols_s + j], and the output columns
    through cmap. Masked lanes (rows past the tile's count, columns past the block's) touch nothing; their indices are
    kept in range for the interpreter, which writes them to the extra element `trash`."""
    del z_in  # the zeros z starts from (aliased to z)
    t, c = pl.program_id(0), pl.program_id(1)
    n_rows, s = tile_rows[t], tile_sector[t]
    n_cols = ncols[s]

    @pl.when((n_rows > 0) & (c * BN < n_cols))
    def _():
        r = jnp.arange(R)
        row_ok = r < n_rows
        rows = perm[jnp.where(row_ok, tile_row0[t] + r, 0)]  # the tile's rows of x and of z
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


def grouped_trial_apply(X, g, spec: TrialSpec, packed, *, R=None, BK=None, BN=None, num_warps=4, num_stages=2,
                        interpret=None):
    """_trial_apply(X, matrices, entries, out_widths) on the rows that can be nonzero: X (rows, Dt), g (rows,) the
    rows' sectors (row_sectors), spec and packed (= pack_blocks(matrices)) the site's. Returns (rows, 4, D') like
    _trial_apply. vmap it over walkers (the launch grid gets a walker axis; spec and packed stay shared). R, BK, BN:
    the tile (default TILE_ROWS, TILE_K, TILE_COLS); num_warps, num_stages: Triton's; interpret: Pallas' interpreter
    (default: off on a GPU, on elsewhere)."""
    R, BK, BN = R or TILE_ROWS, BK or TILE_K, BN or TILE_COLS
    rows, in_cols = X.shape
    if in_cols != spec.in_cols:
        raise ValueError(f"X has {in_cols} columns, the spec's bond {spec.in_cols}")
    size = rows * spec.out_cols
    G = len(spec.N)
    if not G or not rows:
        return jnp.zeros((rows, 4, spec.out_cols // 4), X.dtype)
    interpret = jax.default_backend() != "gpu" if interpret is None else bool(interpret)
    extra = 1 if interpret else 0  # the interpreter also writes masked lanes: to one extra element
    perm, tile_sector, tile_row0, tile_rows = tile_tables(g, G, R)
    tables = [jnp.asarray(np.asarray(v, np.int64)) for v in (spec.start, spec.width, spec.ncols, spec.w_off,
                                                             spec.c_off)]
    args = (tile_sector, tile_row0, tile_rows, perm, *tables, X.reshape(-1), packed,
            jnp.asarray(np.asarray(spec.cmap, np.int64)), jnp.zeros(size + extra, X.dtype))
    kernel = partial(_kernel, R=R, BK=BK, BN=BN, in_cols=in_cols, out_cols=spec.out_cols, trash=size if extra else 0)
    options = {} if interpret else dict(compiler_params=plgpu.CompilerParams(num_warps=num_warps,
                                                                             num_stages=num_stages))
    z = pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((size + extra,), X.dtype),
                       grid=(tile_rows.shape[0], -(-max(spec.ncols) // BN)), input_output_aliases={len(args) - 1: 0},
                       interpret=interpret, name="grouped_trial_gemm", **options)(*args)
    return (z[:size] if extra else z).reshape(rows, 4, spec.out_cols // 4)


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
