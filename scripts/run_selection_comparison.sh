#!/usr/bin/env bash
# Run from the repository root with the server environment activated.
set -euo pipefail

for seed in 11 22 33; do
  for model in erm fragment; do
    run_name="${model}_policy_s${seed}"
    destination="outputs/assay/policy_comparison/drugood_lbap_core_ec50_assay/group_graph/${run_name}"
    if [ -e "$destination" ]; then
      echo "Refusing to overwrite existing run: $destination" >&2
      exit 1
    fi
  done
done

for seed in 11 22 33; do
  for model in erm fragment; do
    python main.py train \
      --config "configs/${model}_selection_comparison.yaml" \
      --device cuda --epochs 100 --selection-ma-window 5 \
      --seed "$seed" --sampler-seed "$seed" \
      --run-name "${model}_policy_s${seed}"
  done
done
