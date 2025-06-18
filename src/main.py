import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer


def compute_entropy(model: BartForConditionalGeneration, tokenizer: BartTokenizer, question: str) -> float:
    """Compute entropy of ..."""
    import random
    return random.random()

def main() -> None:
    train_df = pd.read_parquet("data/webquestions/webq-train.parquet").head()

    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    train_df["question"].apply(lambda q: compute_entropy(model, tokenizer, q))


if __name__ == "__main__":
    main()
