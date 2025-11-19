# Reference: https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py

datasets=("webquestions" "triviaqa" "nq")

for dataset in "${datasets[@]}"; do
  mkdir -p "data/$dataset"
  suffixes=("test.qa.csv" "annotations.jsonl")

  for suffix in "${suffixes[@]}"; do
    dest="data/$dataset/$dataset-$suffix"
    url="https://dl.fbaipublicfiles.com/qaoverlap/data/$dataset-$suffix"
    wget -q --show-progress -O $dest $url
  done
done
