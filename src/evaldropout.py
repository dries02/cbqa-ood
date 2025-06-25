import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer

from sbertdemo import frob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_predictions_batched(
        model: BartForConditionalGeneration,
        tokenizer: BartTokenizer,
        questions: list[str],
        n_reps: int,
        batch_size: int,
    ) -> list[list[str]]:
    """Generate predictions in a batch.

    Args:
        batch_size (int): how many questions to process per time
    """
    all_predictions = []
    for idx in tqdm(range(0, len(questions), batch_size)):
        chunk_qs = questions[idx:idx + batch_size]
        rep_qs = [q for q in chunk_qs for _ in range(n_reps)]
        tok_qs = tokenizer(
            rep_qs,
            max_length=32,                  # should be enough
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out_ids = model.generate(
                input_ids=tok_qs["input_ids"],
                attention_mask=tok_qs["attention_mask"],
                max_new_tokens=32,
            )

        # torch.cuda.empty_cache()

        preds = tokenizer.batch_decode(out_ids, skip_special_tokens=True)

        grouped = [preds[idx2 * n_reps:(idx2 + 1) * n_reps] for idx2 in range(len(chunk_qs))]
        all_predictions.extend(grouped)

    return all_predictions


def fetch_predictions(test_df: pd.DataFrame) -> list[list[str]]:
    model = BartForConditionalGeneration.from_pretrained("models/nq-large").train()          # enable dropout
    model.to(device)
    tokenizer = BartTokenizer.from_pretrained("models/nq-large")

    n_reps = 30
    batch_size = 4

    return generate_predictions_batched(model, tokenizer, test_df["question"].tolist(), n_reps, batch_size)


def analyze():
    test_df = pd.read_parquet("data/webquestions/webquestions-merged-seqs.parquet")


    test_df["frob"] = test_df["predictions"].apply(frob)
    test_df.to_parquet("data/webquestions/webquestions-frob.parquet")
    # for row in test_df.itertuples():
    #     print(f"Question: {row.question}\nGround truth: {row.answers}\nPredicted: {row.predictions}\n"
    #           f"Uncertainty: {row.frob:.3f} (frob)\nLabel: {row.labels}\n")


def main() -> None:
    test_df = pd.read_parquet("data/webquestions/webquestions-merged.parquet")

    test_df["predictions"] = fetch_predictions(test_df)

    test_df.to_parquet("data/webquestions/webquestions-merged-seqs.parquet")


if __name__ == "__main__":
    analyze()
