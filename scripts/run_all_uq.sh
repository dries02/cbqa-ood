set -euo pipefail

DATASETS=("webquestions" "nq")
METHODS=("mcdropout" "flipout")
FLAGS=("use_soft" "no-use_soft")
MODEL=t5-large-ssm
UQ=("f1" "bertscore")

for dataset in "${DATASETS[@]}"; do
    for flag in "${FLAGS[@]}"; do
        echo "\$ uv run -m scripts.baselines --dataset $dataset --model $MODEL --$flag"

        uv run -m scripts.baselines --dataset $dataset --model $MODEL --$flag

        for method in "${METHODS[@]}"; do
            echo "\$ uv run -m scripts.forward --dataset $dataset --model $MODEL --method $method --$flag"

            uv run -m scripts.forward --dataset $dataset --model $MODEL --method $method --$flag

            echo "\$ uv run -m scripts.forwardtoken --dataset $dataset --model $MODEL --method $method --$flag"
            uv run -m scripts.forwardtoken --dataset $dataset --model $MODEL --method $method --$flag

            for uq in "${UQ[@]}"; do
                echo "\$ uv run -m scripts.compute_uncertainty --dataset $dataset --method $method --uq_method $uq --$flag"
                uv run -m scripts.compute_uncertainty --dataset $dataset --method $method --uq_method $uq --$flag
            done
        done
    done
done


