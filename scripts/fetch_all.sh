# one script to fetch all necessary train/test data/annotations

./scripts/fetch_all_test.sh             # fetch all test+annotations and merge
./scripts/fetch_wq_data.sh              # fetch WebQuestions
uv run src/datautils/fetchdata.py       # fetch Natural Questions and TriviaQA
