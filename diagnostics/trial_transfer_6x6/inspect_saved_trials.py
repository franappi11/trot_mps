"""Compare saved 6x6 trial metadata and RDMs without loading the large MPS tensors.

Run with the trot Python environment. Outputs go next to this script.
These are unrotated-trial diagnostics; they do not measure CPMC constraint error.
"""
from pathlib import Path
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TRIALS = Path("/Users/fnappi/trot_trials/uhf_dmrg_6x6_U8")
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
records, arrays = {}, {}
for chi in (1024, 2048):
    path = TRIALS / f"dmrg_chi{chi}.npz"
    with np.load(path, allow_pickle=False) as z:
        d, h = z["rdm1"], z["h1"]
        energy, u = float(z["energy"]), float(z["u"])
        np.testing.assert_allclose(d, d.swapaxes(1, 2), atol=1e-12)
        np.testing.assert_allclose(np.trace(d, axis1=1, axis2=2), z["nelec"], atol=1e-8)
        if arrays:
            np.testing.assert_allclose(h, arrays[1024]["h1"])
        diagonal = np.diagonal(d, axis1=1, axis2=2)
        density = diagonal.sum(axis=0)
        magnetization = (diagonal[0] - diagonal[1]) / 2
        occupations = np.linalg.eigvalsh(d)[:, ::-1]
        kinetic = float(np.einsum("ij,sji->", h, d))
        arrays[chi] = dict(rdm1=d, h1=h, density=density,
                           magnetization=magnetization, occupations=occupations)
        records[chi] = dict(
            path=str(path), energy=energy,
            last_sweep_energy=float(z["sweep_energies"][-1]),
            kinetic_energy=kinetic,
            mean_double_occupancy_inferred_from_saved_E_and_RDM=(energy - kinetic) / (u * 36),
            density_std=float(density.std()),
            rms_static_magnetization=float(np.sqrt(np.mean(magnetization**2))),
            natural_nonidempotency=float(np.sum(occupations * (1 - occupations))),
            occupied_cutoff_gaps=(occupations[:, 15] - occupations[:, 16]).tolist(),
            density=density.reshape(6, 6).tolist(),
            magnetization=magnetization.reshape(6, 6).tolist(),
        )

d0, d1 = arrays[1024]["rdm1"], arrays[2048]["rdm1"]
reference_fidelities = []
for spin in (0, 1):
    a = np.linalg.eigh(d0[spin])[1][:, -16:]
    b = np.linalg.eigh(d1[spin])[1][:, -16:]
    s = np.linalg.svd(a.T @ b, compute_uv=False)
    reference_fidelities.append(float(np.prod(s**2)))
comparison = dict(
    relative_rdm_frobenius_difference=float(np.linalg.norm(d1-d0) / np.linalg.norm(d1)),
    max_site_density_difference=float(np.max(abs(arrays[2048]["density"]-arrays[1024]["density"]))),
    natural_reference_determinant_fidelity_by_spin=reference_fidelities,
    natural_reference_determinant_joint_fidelity=float(np.prod(reference_fidelities)),
    note="The determinant fidelities are NOT fidelities between the two many-body MPS trials.",
)

runs = []
for path in sorted((ROOT / "trot/gmps").glob("ex_6x6_doped_diag-*.out")):
    text = path.read_text()
    args = re.search(r"^args:\s*(.*)$", text, re.M).group(1)
    def option(name):
        match = re.search(rf"--{name}\s+(\S+)", args)
        return match.group(1) if match else None
    final = re.findall(r"E = ([-+0-9.e]+) \+/- ([0-9.e+-]+)", text)[-1]
    runs.append(dict(
        file=str(path), chi=int(option("chi")), chi_w=int(option("chi-w")),
        chi_m=int(option("measure-chi")),
        plan=re.search(r"^walker plan: (\w+)", text, re.M).group(1),
        n_eql=int(re.findall(r"^\[eql\s+\d+/(\d+)\]", text, re.M)[-1]),
        n_sampling=int(re.findall(r"^\[blk\s+\d+/(\d+)\]", text, re.M)[-1]),
        energy=float(final[0]), reported_se=float(final[1]),
    ))
report = dict(trials=records, comparison=comparison, runs=runs,
              limitations=["Uses saved RDMs and energies, without independently contracting the MPS.",
                           "No S^2, trial energy variance, MPS fidelity, or walker overlap signs measured.",
                           "Production rotates the trial by 90 degrees; these RDMs are before rotation.",
                           "rank_exact removes orbital-plan truncation, not finite walker-bond truncation."])
(OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

fig, axes = plt.subplots(2, 3, figsize=(13, 7.3), layout="constrained")
for row, chi in enumerate((1024, 2048)):
    a = arrays[chi]
    for column, key, cmap, vmin, vmax, label in (
        (0, "density", "viridis", .8, 1., "site density"),
        (1, "magnetization", "RdBu_r", -.28, .28, "static magnetization ⟨Sz⟩"),
    ):
        im = axes[row, column].imshow(a[key].reshape(6, 6), cmap=cmap, vmin=vmin, vmax=vmax)
        axes[row, column].set(title=f"χ = {chi}: {label}", xlabel="x", ylabel="y")
        fig.colorbar(im, ax=axes[row, column], shrink=.85)
    for spin, style in ((0, "-"), (1, "--")):
        axes[0, 2].plot(np.arange(1, 37), a["occupations"][spin], style,
                        color=f"C{row}", label=f"χ={chi}, {'up' if spin == 0 else 'down'}")
    axes[1, 2].plot(np.arange(6), a["density"].reshape(6, 6).mean(axis=1), "o-", label=f"χ={chi}")
axes[0, 2].set(title="Natural occupations", xlabel="orbital (sorted)", ylabel="occupation")
axes[0, 2].axvline(16.5, color="gray", linestyle=":")
axes[1, 2].set(title="Charge profile averaged over x", xlabel="y", ylabel="mean site density")
for ax in axes[:, 2]:
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
fig.suptitle("Saved unrotated 6×6 trials: one-body diagnostics", fontsize=14)
for extension in ("png", "pdf"):
    fig.savefig(OUT / f"trial_rdm_comparison.{extension}", dpi=160)
plt.close(fig)
print(json.dumps(dict(trials={k: {n: v for n, v in r.items() if n not in ('density', 'magnetization')}
                             for k, r in records.items()}, comparison=comparison), indent=2))
print("Saved summary.json and trial_rdm_comparison.{png,pdf} in", OUT)
