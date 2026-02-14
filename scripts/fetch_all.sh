# one script to fetch all necessary train/test data/annotations

./scripts/data/fetch_all_test.sh             # fetch all test+annotations and merge
./scripts/data/fetch_wq_data.sh              # fetch WebQuestions
uv run src/datautils/fetchdata.py       # fetch Natural Questions


# clean unicode stuff
datasets=("webquestions" "nq")
splits=(train dev test)
for dataset in "${datasets[@]}"; do
  for split in "${splits[@]}"; do
    echo "\$ uv run scripts/data/clean_json.py --dataset $dataset --split $split"
    uv run scripts/data/clean_json.py --dataset $dataset --split $split
  done
done
