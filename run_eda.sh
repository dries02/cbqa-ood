#!/bin/bash

# Run EDA script for all dataset/OOD combinations
echo "Running EDA checks..."

datasets=("webquestions" "triviaqa" "nq")
ood_types=("far" "near")

for dataset in "${datasets[@]}"; do
    for ood in "${ood_types[@]}"; do
        echo "\$ uv run src/eda.py --dataset $dataset --ood_type $ood"
        uv run src/eda.py --dataset "$dataset" --ood_type "$ood"
    done
done
