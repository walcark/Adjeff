#!/bin/bash
#SBATCH --job-name=simu_smartg
#SBATCH --export=none
#SBATCH --output=sbatch_reports/simu_smartg-%j.out
#SBATCH --error=sbatch_reports/simu_smartg-%j.err
#SBATCH -N 1                              
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_std 
#SBATCH --qos=gpu_all
#SBATCH -n 16
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=68:00:00
#SBATCH --account=cesbio

export PATH="/work/scratch/data/walcark/pixi/bin:$PATH"
cd "/work/scratch/env/walcark/Adjeff"

export PYTHONPATH=$(
    echo "$PYTHONPATH" | \
    tr ':' '\n' | \
    grep -v '/home/uz/walcark/PARA/projects/adjeffcorr_simplecase_evaluation/adjeff/src' | \
    awk '!seen[$0]++' | \
    tr '\n' ':' | \
    sed 's/:$//'
)
export PYTHONPATH="/work/scratch/env/walcark/Adjeff/src:$PYTHONPATH"

export SMARTG_DIR_AUXDATA=/work/scratch/data/walcark/smartg/auxdata

pixi run --tls-no-verify -e dev-gpu python scripts/article1/generate_psf_param_lut.py \
    --species dust \
    --cache-dir /work/scratch/data/walcark/psf_lut_compute_cache \
    --output-dir /work/scratch/data/walcark/psf_lut_store

pixi run --tls-no-verify -e dev-gpu python scripts/article1/generate_psf_param_lut.py \
    --species nitrate \
    --cache-dir /work/scratch/data/walcark/psf_lut_compute_cache \
    --output-dir /work/scratch/data/walcark/psf_lut_store

pixi run --tls-no-verify -e dev-gpu python scripts/article1/generate_psf_param_lut.py \
    --species organicm \
    --cache-dir /work/scratch/data/walcark/psf_lut_compute_cache \
    --output-dir /work/scratch/data/walcark/psf_lut_store
