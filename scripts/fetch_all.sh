# one script to fetch all necessary train/test data/annotations

./scripts/data/fetch_all_test.sh             # fetch all test+annotations and merge
./scripts/data/fetch_wq_data.sh              # fetch WebQuestions
uv run src/datautils/fetchdata.py       # fetch Natural Questions and TriviaQA
