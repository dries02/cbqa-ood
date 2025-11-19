#!/bin/bash

# Run EDA script for all dataset/OOD combinations
echo "Running EDA checks..."

datasets=("webquestions" "triviaqa" "nq")

for dataset in "${datasets[@]}"; do
    echo "\$ uv run src/eda.py --dataset $dataset"
    uv run src/eda.py --dataset "$dataset"
done
