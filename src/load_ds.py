import pandas as pd


def load_webq(path: str) -> None:
    splits = {
        "train": "data/train-00000-of-00001.parquet",
        "test": "data/test-00000-of-00001.parquet",         # not accessed
    }

    train_df = pd.read_parquet("hf://datasets/Stanford/web_questions/" + splits["train"]).drop("url", axis=1)
    print(train_df.head())

    train_df.to_parquet(path, index=False)
    print(pd.read_parquet(path).head())


def main():
    load_webq("data/webquestions/webq-train.parquet")


if __name__ == "__main__":
    main()
