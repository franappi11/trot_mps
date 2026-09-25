#!/bin/bash
# Submit one SLURM job per (L, U, chi_w) combination so they run in parallel,
# plus one DMRG reference job per (L, U). Everything lands in the one --out
# folder: results.jsonl, blocks.jsonl, one log per run and reference.jsonl,
# which sweep_plots.ipynb reads.
#
# Takes the same options as run_mps_sweep.py; --L, --U and --chi-w may list
# several values, everything else is passed to every job unchanged.
#
#   ./submit_mps_sweep.sh --L 32 48 --U 4 8 --chi-w 2 4 6 \
#       --trial-chi 8 --walkers 200 --eql 60 --blocks 200 --out sweep_T8
#
# sbatch options go before a "--" separator:
#   ./submit_mps_sweep.sh --time=2-00:00:00 --cpus-per-task=8 -- --L 48 --chi-w 2 4
#
# REF_CHI=N sets the DMRG reference bond dimension (default 200); NO_REF=1
# skips the reference jobs (e.g. when reference.jsonl already has them).
# DRY_RUN=1 prints the sbatch commands instead of submitting them.
# PYTHON=/path/to/python is forwarded to the jobs (see run_mps_sweep.sh).

cd "$(dirname "$(readlink -f "$0")")" || exit 1

sbatch_opts=()
if [[ " $* " == *" -- "* ]]; then
    while [[ "$1" != "--" ]]; do sbatch_opts+=("$1"); shift; done
    shift
fi

Ls=(32); Us=(4); chi_ws=(2 4 6); out=sweep; rest=()
while (($#)); do
    case "$1" in
        --L|--U|--chi-w)
            opt="$1"; shift; vals=()
            while (($#)) && [[ "$1" != --* ]]; do vals+=("$1"); shift; done
            case "$opt" in
                --L) Ls=("${vals[@]}") ;;
                --U) Us=("${vals[@]}") ;;
                --chi-w) chi_ws=("${vals[@]}") ;;
            esac ;;
        --out) out="$2"; rest+=("$1" "$2"); shift 2 ;;
        *) rest+=("$1"); shift ;;
    esac
done

submit() {
    if [[ -n "$DRY_RUN" ]]; then echo "$*"; else "$@"; fi
}

mkdir -p slurm "$out"
for L in "${Ls[@]}"; do
    for U in "${Us[@]}"; do
        if [[ -z "$NO_REF" ]]; then
            submit sbatch -J "ref_L${L}_U${U}" "${sbatch_opts[@]}" --export=ALL,TARGET=dmrg_reference.py \
                run_mps_sweep.sh --L "$L" --U "$U" --chi "${REF_CHI:-200}" --out "$out/reference.jsonl"
        fi
        for w in "${chi_ws[@]}"; do
            submit sbatch -J "L${L}_U${U}_w${w}" "${sbatch_opts[@]}" run_mps_sweep.sh \
                --L "$L" --U "$U" --chi-w "$w" "${rest[@]}"
        done
    done
done
