"""The grouped trial GEMM (trot/gmps/grouped_gemm.py) against dense_engine._trial_apply.

The kernel runs in Pallas' interpreter on a CPU and is compiled by Triton on a GPU. Here the environments are
synthetic, built in the sector layout's format with the exact structure of real ones (row r nonzero only in the columns
of its own sector), to reach the schedule's edge cases: empty sectors, rows in no sector, counts that fill whole tiles,
sectors narrower than one inner step, blocks with fewer columns than a column tile, several walkers under vmap. The
environments of converted walkers (the premise, entries outside a row's sector are exactly zero, and the kernel against
_trial_apply at every site) are tested in test_mps_engine.py (test_grouped_trial_gemm_on_converted_walkers).

Also here, on the same synthetic sites: the walker-first right environments' kernel (summed_trial_apply, with the
walkers pooled under vmap), the rows' order by sector and their products (sector_order, sector_products with XLA's
GEMMs) and, on a GPU with the library built (cublas_ext/build.sh; skipped otherwise), the cuBLAS custom call. The
fused kernel reading those products (TROT_TRIAL_GEMM=cublas) is tested on converted walkers in test_mps_engine.py.

    python -m pytest tests/test_grouped_gemm.py
"""

from trot import config

config.configure_once()

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trot.gmps import dense_engine
from trot.gmps import grouped_gemm as gg

N0 = 10  # electron number of the first left sector
TILES = {"16": dict(R=16, BK=16, BN=16), "32": dict(R=32, BK=16, BN=32)}
CASES = {  # sector widths, rows, sectors left empty, rows in no sector
    "uneven": ((3, 40, 17, 1, 25), 203, (3,), 9),
    "single": ((50,), 37, (), 0),
    "empty-first": ((12, 30, 5), 120, (0,), 4),
}


def _site(rng, widths):
    """A site in SiteLayout's format: left sectors N = N0 + s of these widths, right sectors N = N0 + k (k = 0 .. G + 1)
    of random widths; physical state p of left sector s reaches right sector s + n(p) (state 0 always, the others with
    probability 0.8), so (p, right sector) is never reached from two left sectors, as in sector_layout. Returns
    (entries, out_widths, in_sectors, matrices)."""
    G = len(widths)
    out_widths = tuple(int(w) for w in rng.integers(1, 40, G + 2))
    starts = np.concatenate(([0], np.cumsum(widths)))
    in_sectors = tuple((N0 + s, int(starts[s]), int(starts[s + 1])) for s in range(G))
    entries = []
    for s, (_, start, stop) in enumerate(in_sectors):
        cols, offset = [], 0
        for p in range(4):
            if p == 0 or rng.random() < 0.8:
                k = s + dense_engine.N_PHYSICAL[p]
                cols.append((p, k, offset, out_widths[k]))
                offset += out_widths[k]
        entries.append((start, stop, tuple(cols)))
    matrices = [jnp.asarray(rng.standard_normal((stop - start, sum(c[3] for c in cols))))
                for start, stop, cols in entries]
    return tuple(entries), out_widths, in_sectors, matrices


def _labels(rng, rows, G, empty, n_none):
    """Each row's sector (G: none): n_none rows in no sector, the others spread over the sectors not in empty."""
    allowed = [s for s in range(G) if s not in empty]
    g = rng.choice(allowed, rows)
    g[rng.choice(rows, n_none, replace=False)] = G
    return g


def _environment(rng, g, in_sectors):
    """X (rows, Dt), row r random in the columns of its sector g[r] and exactly zero elsewhere."""
    X = np.zeros((len(g), in_sectors[-1][2]))
    for r, s in enumerate(g):
        if s < len(in_sectors):
            _, start, stop = in_sectors[s]
            X[r, start:stop] = rng.standard_normal(stop - start)
    return jnp.asarray(X)


def _assert_close(got, want):
    want = np.asarray(want)
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-12, atol=1e-13 * np.abs(want).max(initial=1.0))


@pytest.mark.parametrize("tiles", list(TILES))
@pytest.mark.parametrize("case", list(CASES))
def test_grouped_trial_apply_equals_trial_apply(case, tiles):
    """The kernel's output equals _trial_apply's (and the NumPy reference's) on an environment with the sector
    structure, for uneven sectors (narrower than an inner step, wider than two), an empty sector, rows in no sector."""
    widths, rows, empty, n_none = CASES[case]
    rng = np.random.default_rng(len(widths) * rows)
    entries, out_widths, in_sectors, matrices = _site(rng, widths)
    spec = gg.trial_spec(entries, out_widths, in_sectors)
    g = _labels(rng, rows, len(widths), empty, n_none)
    X = _environment(rng, g, in_sectors)
    want = dense_engine._trial_apply(X, matrices, entries, out_widths)
    got = gg.grouped_trial_apply(X, jnp.asarray(g), spec, gg.pack_blocks(matrices), **TILES[tiles])
    assert got.shape == want.shape
    _assert_close(got, want)
    _assert_close(gg.grouped_reference(X, g, spec, matrices), want)


@pytest.mark.parametrize("caps", ["exact", "half", "over", "zero-one"])
def test_hybrid_equals_trial_apply_whatever_the_caps(caps):
    """hybrid_trial_apply (XLA's GEMM on each sector's first caps[s] rows, the grouped kernel on the rest) equals
    _trial_apply for caps equal to the counts, below them (half the rows through the kernel), above them (padding
    rows dropped) and 0 for one sector (all its rows through the kernel)."""
    widths, rows, empty, n_none = CASES["uneven"]
    rng = np.random.default_rng(17)
    entries, out_widths, in_sectors, matrices = _site(rng, widths)
    spec = gg.trial_spec(entries, out_widths, in_sectors)
    g = _labels(rng, rows, len(widths), empty, n_none)
    X = _environment(rng, g, in_sectors)
    counts = np.bincount(g, minlength=len(widths) + 1)[:len(widths)]
    chosen = {"exact": counts, "half": counts // 2, "over": counts + 5,
              "zero-one": np.where(np.arange(len(counts)) == 1, 0, counts)}[caps]
    got = gg.hybrid_trial_apply(X, jnp.asarray(g), spec, matrices, gg.pack_blocks(matrices), tuple(chosen),
                                **TILES["16"])
    _assert_close(got, dense_engine._trial_apply(X, matrices, entries, out_widths))


def test_presorted_rows_give_the_sorted_output():
    """With X's rows sorted by sector beforehand (X[perm]) and presorted=True, the kernel reads the tiles' rows at
    their positions and returns the output in the same order: _trial_apply's output[perm]."""
    widths, rows, empty, n_none = CASES["uneven"]
    rng = np.random.default_rng(23)
    entries, out_widths, in_sectors, matrices = _site(rng, widths)
    spec = gg.trial_spec(entries, out_widths, in_sectors)
    g = _labels(rng, rows, len(widths), empty, n_none)
    X = _environment(rng, g, in_sectors)
    perm = np.asarray(gg.tile_tables(jnp.asarray(g), len(widths), 16)[0])
    got = gg.grouped_trial_apply(X[perm], jnp.asarray(g), spec, gg.pack_blocks(matrices), presorted=True,
                                 **TILES["16"])
    _assert_close(got, np.asarray(dense_engine._trial_apply(X, matrices, entries, out_widths))[perm])


def test_triton_gemm_is_a_gemm():
    """The plain Pallas GEMM of the diagnostic equals A @ B (shapes multiples of the tile)."""
    rng = np.random.default_rng(29)
    A, B = jnp.asarray(rng.standard_normal((128, 48))), jnp.asarray(rng.standard_normal((48, 64)))
    _assert_close(gg.triton_gemm(A, B, BM=64, BN=32, BK=16), np.asarray(A) @ np.asarray(B))


def test_counts_that_fill_whole_tiles():
    """Every sector's count a multiple of the tile's rows (no partial tile) and one sector with exactly one tile."""
    rng = np.random.default_rng(5)
    entries, out_widths, in_sectors, matrices = _site(rng, (20, 33, 8))
    spec = gg.trial_spec(entries, out_widths, in_sectors)
    g = rng.permutation(np.repeat([0, 1, 2], [48, 16, 32]))
    X = _environment(rng, g, in_sectors)
    got = gg.grouped_trial_apply(X, jnp.asarray(g), spec, gg.pack_blocks(matrices), **TILES["16"])
    _assert_close(got, dense_engine._trial_apply(X, matrices, entries, out_widths))


def test_vmap_over_walkers_with_different_counts():
    """Under vmap (the walkers' rows pooled into one launch, spec and blocks shared) each walker gets its own result,
    the same as alone."""
    rng = np.random.default_rng(11)
    widths = (7, 26, 19, 2)
    entries, out_widths, in_sectors, matrices = _site(rng, widths)
    spec, packed = gg.trial_spec(entries, out_widths, in_sectors), gg.pack_blocks(matrices)
    gs = np.stack([_labels(rng, 90, len(widths), empty, 3) for empty in ((), (1,), (0, 3))])
    Xs = jnp.stack([_environment(rng, g, in_sectors) for g in gs])
    got = jax.vmap(lambda X, g: gg.grouped_trial_apply(X, g, spec, packed, **TILES["16"]))(Xs, jnp.asarray(gs))
    for X, g, out in zip(Xs, gs, got):
        _assert_close(out, dense_engine._trial_apply(X, matrices, entries, out_widths))
        np.testing.assert_array_equal(np.asarray(out), np.asarray(
            gg.grouped_trial_apply(X, jnp.asarray(g), spec, packed, **TILES["16"])))


def test_nested_vmap_and_walker_blocks():
    """Two vmaps (walkers in chunks) pool all the rows; walkers with blocks of their own (packed batched) keep a
    launch each; walkers sharing their rows (X not batched) each get the shared result. Both equal _trial_apply."""
    rng = np.random.default_rng(12)
    widths = (11, 3, 30)
    entries, out_widths, in_sectors, matrices = _site(rng, widths)
    spec, packed = gg.trial_spec(entries, out_widths, in_sectors), gg.pack_blocks(matrices)
    gs = np.stack([_labels(rng, 40, len(widths), empty, 2) for empty in ((), (1,), (2,), (0,), (), (1, 2))])
    Xs = jnp.stack([_environment(rng, g, in_sectors) for g in gs])
    apply = lambda X, g, packed: gg.grouped_trial_apply(X, g, spec, packed, **TILES["16"])
    nested = jax.vmap(jax.vmap(apply, in_axes=(0, 0, None)), in_axes=(0, 0, None))
    got = nested(Xs.reshape(2, 3, *Xs.shape[1:]), jnp.asarray(gs).reshape(2, 3, -1), packed).reshape(6, *Xs.shape[1:2],
                                                                                                      4, -1)
    scaled = jnp.stack([packed, 2.0 * packed])  # two walkers with blocks of their own
    own = jax.vmap(apply)(Xs[:2], jnp.asarray(gs[:2]), scaled)
    shared = jax.vmap(apply, in_axes=(None, 0, None))(Xs[0], jnp.asarray(gs[:1]).repeat(2, axis=0), packed)
    for X, out in zip(Xs, got):
        _assert_close(out, dense_engine._trial_apply(X, matrices, entries, out_widths))
    for X, scale, out in zip(Xs[:2], (1.0, 2.0), own):
        _assert_close(out, dense_engine._trial_apply(X, [scale * W for W in matrices], entries, out_widths))
    for out in shared:
        _assert_close(out, dense_engine._trial_apply(Xs[0], matrices, entries, out_widths))


@pytest.mark.parametrize("R", [4, 16])
def test_tile_tables_cover_every_row_once(R):
    """perm sorts the rows by sector; the tiles cover each sector's rows exactly once, never cross a sector, never
    exceed R rows, fit the static bound ceil(rows / R) + G, and the tiles past the last are idle."""
    rng = np.random.default_rng(R)
    G, rows = 6, 101
    g = _labels(rng, rows, G, (2,), 5)
    perm, sector, row0, n = (np.asarray(x) for x in gg.tile_tables(jnp.asarray(g), G, R))
    assert len(n) == -(-rows // R) + G
    assert np.array_equal(np.sort(g, kind="stable"), g[perm])
    seen = []
    for s, r0, k in zip(sector, row0, n):
        assert 0 <= k <= R
        if k:
            rows_t = perm[r0:r0 + k]
            assert np.all(g[rows_t] == s)
            seen.extend(rows_t.tolist())
    assert sorted(seen) == sorted(np.flatnonzero(g < G).tolist())
    active = np.flatnonzero(n)
    assert np.array_equal(active, np.arange(len(active)))  # the active tiles first, then idle ones


def test_row_sectors_follow_the_labels_and_channel_offsets():
    """row_sectors: row (a, b, w) in E.reshape(Da * Db * Dw, Dt)'s order gets the sector of n_a + n_b + offset_w,
    or G when no sector has that electron number."""
    rng = np.random.default_rng(3)
    entries, out_widths, in_sectors, _ = _site(rng, (4, 9, 6))
    spec = gg.trial_spec(entries, out_widths, in_sectors)
    n_a, n_b, offsets = rng.integers(3, 8, 5), rng.integers(4, 9, 4), np.array([-1, 0, 1])
    g = np.asarray(gg.row_sectors(jnp.asarray(n_a, float), jnp.asarray(n_b, float), spec, offsets))
    want = [spec.N.index(N) if N in spec.N else len(spec.N)
            for N in (n_a[:, None, None] + n_b[None, :, None] + offsets[None, None, :]).ravel()]
    assert g.tolist() == want
    g2 = np.asarray(gg.row_sectors(jnp.asarray(n_a, float), jnp.asarray(n_b, float), spec))
    assert g2.tolist() == [spec.N.index(N) if N in spec.N else len(spec.N) for N in (n_a[:, None] + n_b).ravel()]


def test_multiply_adds_count_the_sectors_rows():
    """multiply_adds: dense = every row through every GEMM, useful = each sector's rows, tiled = rounded up to tiles."""
    rng = np.random.default_rng(2)
    entries, out_widths, in_sectors, _ = _site(rng, (5, 11))
    spec = gg.trial_spec(entries, out_widths, in_sectors)
    g = np.array([0] * 3 + [1] * 17 + [2] * 4)
    per = np.asarray(spec.width) * np.asarray(spec.ncols)
    m = gg.multiply_adds(g, spec, R=8)
    assert m["dense"] == len(g) * per.sum()
    assert m["useful"] == 3 * per[0] + 17 * per[1]
    assert m["tiled"] == 8 * per[0] + 24 * per[1]


def test_tile_for_picks_each_kinds_large_tile_by_rows():
    """tile_for: from 32768 rows the kind's own tile, from 2048 SMALL_TILE, else TINY_TILE; an unknown kind refused;
    _tile_options keeps what is given."""
    as_tile = lambda d: (d["R"], d["BK"], d["BN"], d["num_warps"], d["num_stages"])
    for kind, tile in (("grouped", gg.LARGE_TILE), ("fused", gg.FUSED_TILE), ("summed", gg.SUMMED_TILE),
                       ("epilogue", gg.EPILOGUE_TILE)):
        assert as_tile(gg.tile_for(32768, kind)) == tuple(tile)
        assert as_tile(gg.tile_for(2048, kind)) == tuple(gg.SMALL_TILE)
        assert as_tile(gg.tile_for(2047, kind)) == tuple(gg.TINY_TILE)
    with pytest.raises(ValueError, match="kind"):
        gg.tile_for(10, "other")
    options = gg._tile_options(10, "fused", 32, None, None, 8, None)
    assert (options["R"], options["num_warps"]) == (32, 8) and options["BK"] == gg.TINY_TILE[1]


# ============================================================================
# summed_trial_apply: the walker-first right environments' trial GEMM
# ============================================================================

SUMMED_CASES = {  # left sector widths, Da, Db
    "uneven": ((3, 40, 17, 1, 25), 9, 7),
    "single": ((50,), 4, 5),
    "two": ((70, 33), 6, 11),
}


def _summed_operand(rng, Da, Db, G, right_widths):
    """Labels n_a (Da,), n_b (Db,) whose sums run from N0 - 1 to N0 + G (some rows in no left sector) and the
    walker-first operand X (Da, 2, Db, 2, Dt') = (a, n_alpha, b, n_beta, t'): row (a, b) random in the right sector of
    n_a + n_b + n(p) for p = n_alpha + 2 n_beta (where it exists), exactly zero elsewhere, as _backward_summed's."""
    n_a, n_b = rng.integers(0, 3, Da), rng.integers(N0 - 1, N0 + G - 1, Db)
    starts = np.concatenate(([0], np.cumsum(right_widths)))
    X = np.zeros((Da, 2, Db, 2, int(starts[-1])))
    for a in range(Da):
        for b in range(Db):
            for p in range(4):
                k = n_a[a] + n_b[b] - N0 + dense_engine.N_PHYSICAL[p]
                if 0 <= k < len(right_widths):
                    X[a, p % 2, b, p // 2, starts[k]:starts[k + 1]] = rng.standard_normal(right_widths[k])
    return n_a.astype(float), n_b.astype(float), X


def _summed_reference(X, entries, right_widths, matrices, left_width):
    """_backward_summed's trial GEMM with NumPy: per left sector the operand's columns (p, right sector of the
    piece) side by side, times the forward block's transpose, for every row."""
    Da, _, Db, _, _ = X.shape
    starts = np.concatenate(([0], np.cumsum(right_widths)))
    out = np.zeros((Da * Db, left_width))
    for (start, stop, cols), W in zip(entries, matrices):
        G = np.concatenate([X[:, p % 2, :, p // 2, starts[k]:starts[k] + width].reshape(Da * Db, width)
                            for p, k, _, width in cols], axis=1)
        out[:, start:stop] = G @ np.asarray(W).T
    return out


def _summed_site(case, seed):
    widths, Da, Db = SUMMED_CASES[case]
    rng = np.random.default_rng(seed)
    entries, right_widths, left_sectors, matrices = _site(rng, widths)
    sspec = gg.summed_spec(entries, right_widths, left_sectors)
    return rng, (Da, Db, len(widths)), entries, right_widths, left_sectors, matrices, sspec


def test_summed_spec_maps_the_forward_blocks_columns():
    """summed_spec: per left sector its electron number, columns, inner size (the pieces' widths), the transposed
    block's offset and the column map p Dt' + the right sector's start + j in the block's column order."""
    _, _, entries, right_widths, left_sectors, matrices, sspec = _summed_site("uneven", 1)
    d_right, starts = sum(right_widths), np.concatenate(([0], np.cumsum(right_widths)))
    assert sspec.in_cols == 4 * d_right and sspec.out_cols == left_sectors[-1][2]
    w_off = 0
    for s, ((start, stop, cols), W) in enumerate(zip(entries, matrices)):
        assert (sspec.N[s], sspec.out_start[s], sspec.out_width[s]) == (N0 + s, start, stop - start)
        assert sspec.K[s] == W.shape[1] and sspec.w_off[s] == w_off
        kmap = sspec.kmap[sspec.k_off[s]:sspec.k_off[s] + sspec.K[s]]
        assert list(kmap) == [p * d_right + starts[k] + j for p, k, _, width in cols for j in range(width)]
        w_off += W.size


@pytest.mark.parametrize("tiles", list(TILES))
@pytest.mark.parametrize("case", list(SUMMED_CASES))
def test_summed_trial_apply_equals_the_sector_gemms(case, tiles):
    """summed_trial_apply (each row (a, b) through the left sector of n_a + n_b alone, the operand read in its
    (a, n_alpha, b, n_beta, t') layout through the column map) equals every row through every sector's GEMM, zeros
    for the rows in no sector, for uneven sectors, a missing physical piece and one sector."""
    rng, (Da, Db, G), entries, right_widths, left_sectors, matrices, sspec = _summed_site(case, 2)
    n_a, n_b, X = _summed_operand(rng, Da, Db, G, right_widths)
    g = gg.row_sectors(jnp.asarray(n_a), jnp.asarray(n_b), sspec)
    got = gg.summed_trial_apply(jnp.asarray(X), g, sspec, gg.pack_blocks([W.T for W in matrices]), **TILES[tiles])
    assert got.shape == (Da * Db, sspec.out_cols)
    _assert_close(got, _summed_reference(X, entries, right_widths, matrices, sspec.out_cols))


def test_summed_trial_apply_pools_walkers_under_vmap():
    """Under vmap the walkers' rows go through one launch (the pooled tiles mix walkers); each walker's result is
    bitwise its own alone, and equals the reference; vmapped twice it is the same."""
    rng, (Da, Db, G), entries, right_widths, left_sectors, matrices, sspec = _summed_site("uneven", 3)
    packed = gg.pack_blocks([W.T for W in matrices])
    walkers = [_summed_operand(rng, Da, Db, G, right_widths) for _ in range(4)]
    Xs = jnp.stack([jnp.asarray(X) for *_, X in walkers])
    gs = jnp.stack([gg.row_sectors(jnp.asarray(a), jnp.asarray(b), sspec) for a, b, _ in walkers])
    apply = lambda X, g: gg.summed_trial_apply(X, g, sspec, packed, **TILES["16"])
    got = jax.vmap(apply)(Xs, gs)
    twice = jax.vmap(jax.vmap(apply))(Xs.reshape(2, 2, *Xs.shape[1:]), gs.reshape(2, 2, -1))
    for k, (_, _, X) in enumerate(walkers):
        np.testing.assert_array_equal(np.asarray(got[k]), np.asarray(apply(Xs[k], gs[k])))
        np.testing.assert_array_equal(np.asarray(twice[k // 2, k % 2]), np.asarray(got[k]))
        _assert_close(got[k], _summed_reference(X, entries, right_widths, matrices, sspec.out_cols))


def test_summed_trial_apply_refuses_another_bond():
    _, (Da, Db, _), *_, sspec = _summed_site("two", 4)
    with pytest.raises(ValueError, match="right bond"):
        gg.summed_trial_apply(jnp.zeros((Da, 2, Db, 2, sspec.in_cols // 4 + 1)), jnp.zeros(Da * Db, jnp.int32),
                              sspec, jnp.zeros(1))


# ============================================================================
# The cuBLAS sector products (sector_order, sector_products, the custom call)
# ============================================================================

def test_sector_order_sorts_the_rows_by_sector():
    """sector_order: perm sorts the rows (a, b, member) by row_sectors' sector, stably, the rows in no sector last;
    counts per sector; positions inverts perm."""
    rng = np.random.default_rng(6)
    entries, out_widths, in_sectors, _ = _site(rng, (4, 9, 6))
    spec = gg.trial_spec(entries, out_widths, in_sectors)
    n_a, n_b, offsets = rng.integers(3, 8, 5), rng.integers(4, 9, 4), np.array([-1, 0, 1])
    perm, counts, positions = (np.asarray(x) for x in gg.sector_order(
        jnp.asarray(n_a, float), jnp.asarray(n_b, float), spec, offsets, 3))
    g = np.asarray(gg.row_sectors(jnp.asarray(n_a, float), jnp.asarray(n_b, float), spec, offsets))
    assert np.array_equal(perm, np.argsort(g, kind="stable"))
    assert counts.tolist() == np.bincount(g, minlength=len(spec.N) + 1)[:len(spec.N)].tolist()
    assert np.array_equal(positions[perm], np.arange(len(g)))
    perm0, _, _ = gg.sector_order(jnp.asarray(n_a, float), jnp.asarray(n_b, float), spec, None, 1)
    g0 = np.asarray(gg.row_sectors(jnp.asarray(n_a, float), jnp.asarray(n_b, float), spec))
    assert np.array_equal(np.asarray(perm0), np.argsort(g0, kind="stable"))


def _products_walker(rng, case, in_sectors):
    """A walker of a CASES site: (labels g, environment X, perm sorting g stably, counts per sector)."""
    widths, rows, empty, n_none = CASES[case]
    g = _labels(rng, rows, len(widths), empty, n_none)
    X = _environment(rng, g, in_sectors)
    return g, X, np.argsort(g, kind="stable"), np.bincount(g, minlength=len(widths) + 1)[:len(widths)]


def _products_case(case, seed=7):
    rng = np.random.default_rng(seed)
    entries, out_widths, in_sectors, matrices = _site(rng, CASES[case][0])
    spec = gg.trial_spec(entries, out_widths, in_sectors)
    return (spec, matrices) + _products_walker(rng, case, in_sectors) + (rng, in_sectors)


def _assert_products(Z, X, g, perm, counts, spec, matrices):
    """Z's defined entries: position i of sector s's rows holds X[perm[i]]'s sector columns times block s."""
    Z, X = np.asarray(Z), np.asarray(X)
    assert Z.shape == (len(g), max(spec.ncols))
    for i, r in enumerate(perm[:int(counts.sum())]):
        s = g[r]
        want = X[r, spec.start[s]:spec.start[s] + spec.width[s]] @ np.asarray(matrices[s])
        np.testing.assert_allclose(Z[i, :spec.ncols[s]], want, rtol=1e-12, atol=1e-13 * np.abs(want).max(initial=1.0))


@pytest.mark.parametrize("case", list(CASES))
def test_reference_sector_products_are_each_rows_sector_product(case):
    """sector_products with XLA's GEMMs (the default off the GPU): each sorted position holds its row's product with
    its own sector's block, for empty sectors and rows in no sector too."""
    spec, matrices, g, X, perm, counts, *_ = _products_case(case)
    Z = gg.sector_products(X, jnp.asarray(perm), jnp.asarray(counts), spec, gg.pack_blocks(matrices), use_ffi=False)
    _assert_products(Z, X, g, perm, counts, spec, matrices)


def test_sector_products_without_the_library_say_how_to_build_it(monkeypatch):
    """The custom call asked for without its library: an error naming the build script, before any device call."""
    spec, matrices, g, X, perm, counts, *_ = _products_case("single")
    monkeypatch.setattr(gg, "CUBLAS_LIB", "/nonexistent/libtrot_sector_products.so")
    monkeypatch.setattr(gg, "_FFI", {})
    with pytest.raises(RuntimeError, match="build.sh"):
        gg.sector_products(X, jnp.asarray(perm), jnp.asarray(counts), spec, gg.pack_blocks(matrices), use_ffi=True)


def _require_cublas():
    if not gg.cublas_available():
        pytest.skip(f"needs a GPU and the cuBLAS library {gg.CUBLAS_LIB} (sbatch trot/gmps/cublas_ext/build.sh)")


@pytest.mark.parametrize("case", list(CASES))
def test_cublas_sector_products_equal_the_rows_products(case):
    """The custom call (cuBLAS, one DGEMM per sector on its gathered rows): each sorted position's product, eagerly,
    under jit and under vmap (one call per walker), for empty sectors and rows in no sector."""
    _require_cublas()
    spec, matrices, g, X, perm, counts, rng, in_sectors = _products_case(case)
    packed = gg.pack_blocks(matrices)
    call = lambda X, perm, counts: gg.sector_products(X, perm, counts, spec, packed, use_ffi=True)
    for Z in (call(jnp.asarray(X), jnp.asarray(perm), jnp.asarray(counts)),
              jax.jit(call)(jnp.asarray(X), jnp.asarray(perm), jnp.asarray(counts))):
        _assert_products(Z, X, g, perm, counts, spec, matrices)
    g2, X2, perm2, counts2 = _products_walker(rng, case, in_sectors)  # a second walker of the same site
    Xs, perms = jnp.stack([X, X2]), jnp.stack([jnp.asarray(perm), jnp.asarray(perm2)])
    countss = jnp.stack([jnp.asarray(counts), jnp.asarray(counts2)])
    Zs = jax.jit(jax.vmap(call))(Xs, perms, countss)  # one call, the blocks shared (a leading 1)
    _assert_products(Zs[0], X, g, perm, counts, spec, matrices)
    _assert_products(Zs[1], X2, g2, perm2, counts2, spec, matrices)
    nested = jax.jit(jax.vmap(jax.vmap(call)))(Xs[None], perms[None], countss[None])  # two leading batch axes
    _assert_products(nested[0, 1], X2, g2, perm2, counts2, spec, matrices)
    own = jax.jit(jax.vmap(lambda X, perm, counts, packed: gg.sector_products(X, perm, counts, spec, packed,
                                                                              use_ffi=True)))(
        Xs, perms, countss, jnp.stack([packed, 2.0 * packed]))  # every walker its own blocks
    _assert_products(own[0], X, g, perm, counts, spec, matrices)
    _assert_products(own[1], X2, g2, perm2, counts2, spec, [2.0 * W for W in matrices])


def test_cublas_sector_products_with_every_row_in_no_sector():
    """No row in any sector: nothing to multiply, the call returns (Z undefined, never read)."""
    _require_cublas()
    spec, matrices, g, X, perm, counts, *_ = _products_case("uneven")
    G = len(spec.N)
    Z = gg.sector_products(jnp.asarray(X), jnp.arange(len(g)), jnp.zeros(G, jnp.int64), spec,
                           gg.pack_blocks(matrices), use_ffi=True)
    assert Z.shape == (len(g), max(spec.ncols))
