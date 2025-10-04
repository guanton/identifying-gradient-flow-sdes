#!/usr/bin/env bash
# run_main_unif.sh
set -euo pipefail

ROOT="main_experiments"
P0="unif"

pot_names=(oakley_ohagan quadratic styblinski_tang wavy_plateau bohachevsky)
pot_flags=(oakley_ohagan poly styblinski_tang wavy_plateau bohachevsky)

seeds=(1000 1001 1002 1003 1004 1005 1006 1007 1008 1009)
betas=(0.1)           # β (σ² = 2β)
N=2000
dt=0.01

for idx in "${!pot_names[@]}"; do
  prefix="${pot_names[$idx]}"
  pot_flag="${pot_flags[$idx]}"

  echo "================ POTENTIAL: ${prefix} ================"

  for seed in "${seeds[@]}"; do
    echo "----- SEED: ${seed} -----"
    for beta in "${betas[@]}"; do
      diff=$(awk "BEGIN{printf \"%.1f\", $beta * 2}")   # σ² = 2β
      dataset="3_margs_langevin_${prefix}_diff-${diff}_seed-${seed}"

      echo "=== Generating data: ${dataset}  (β=${beta} ⇒ σ²=${diff}) ==="
      python data_generator.py \
        --root "${ROOT}" \
        --p0 "${P0}" \
        --potential "${pot_flag}" \
        --dataset-name "${dataset}" \
        --beta "${beta}" \
        --internal wiener \
        --seed "${seed}" \
        --dt "${dt}" \
        --initial_length 4 \
        --n-timesteps 2 \
        --n-particles "${N}"

      echo ">>> Training on ${dataset}"

      # 1) JKONet*
      python train.py \
        --root "${ROOT}" \
        --p0 "${P0}" \
        --solver jkonet-star-potential-internal \
        --seed "${seed}" \
        --dataset "${dataset}" \
        --epochs 100 \
        --potential "${pot_flag}" \
        --sb-iters 0 \
        --diffusivity "${diff}" \
        --dt "${dt}" \
        --method-tag "jkonet_star"

      # 2) WOT
      python train.py \
        --root "${ROOT}" \
        --p0 "${P0}" \
        --solver jkonet-star-potential-internal \
        --seed "${seed}" \
        --dataset "${dataset}" \
        --epochs 0 \
        --potential "${pot_flag}" \
        --sb-iters 1 \
        --diffusivity "${diff}" \
        --activation "silu" \
        --dt "${dt}" \
        --method-tag "wot"

      # 3) APPEX
      python train.py \
        --root "${ROOT}" \
        --p0 "${P0}" \
        --solver jkonet-star-potential-internal \
        --dataset "${dataset}" \
        --epochs 0 \
        --seed "${seed}" \
        --potential "${pot_flag}" \
        --sb-iters 30 \
        --diffusivity "${diff}" \
        --activation "silu" \
        --dt "${dt}" \
        --method-tag "nn_appex"

      # 4) SBIRR (MIT)
      python train.py \
        --root "${ROOT}" \
        --p0 "${P0}" \
        --solver jkonet-star-potential-internal \
        --seed "${seed}" \
        --dataset "${dataset}" \
        --epochs 0 \
        --potential "${pot_flag}" \
        --fix_diffusion \
        --sb-iters 30 \
        --diffusivity "${diff}" \
        --activation "silu" \
        --dt "${dt}" \
        --method-tag "sbirr"

      echo
    done
  done
done