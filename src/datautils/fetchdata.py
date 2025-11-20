""""Load TriviaQA and Natural Questions datasets from Huggingface."""

from pathlib import Path

from datasets import load_dataset


def fetch_data(base: str, ds_name: str) -> None:
    """Fetch data from Huggingface FlashRAG."""
    basepath = Path(base) / ds_name
    splits = ("train", "dev")
    for split in splits:
        dataset = load_dataset("RUC-NLPIR/FlashRAG_datasets", ds_name, split=split)
        dataset = dataset.remove_columns(["id"]).rename_column("golden_answers", "answers")
        filename = f"{ds_name}-{split}.jsonl"
        dataset.to_json(basepath / filename, orient="records", lines=True)


def main() -> None:
    """Fetch training and development sets for TriviaQA and NaturalQuestions."""
    ds = ("triviaqa", "nq")
    base = "data"
    for dataset in ds:
        fetch_data(base, dataset)


if __name__ == "__main__":
    main()
