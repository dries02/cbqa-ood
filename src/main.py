from flipoutbart import FlipoutBart

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sbertdemo import frob
from transformers import BartForConditionalGeneration, BartTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_predictions(model: BartForConditionalGeneration, tokenizer: BartTokenizer, question: str, batch_size: int) -> list[str]:
    model.eval()
    tok_q = tokenizer(
        question,
        max_length=64,                  # should be enough
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    batch = {k: v.expand(batch_size, -1).contiguous() for k, v in tok_q.items()}
    batch = {k: v.to(device) for k, v in batch.items()}

    pred_ids = model.generate(**batch, max_length=32)
    predictions = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    return predictions



def compute_uncertainty(model: BartForConditionalGeneration, tokenizer: BartTokenizer, question: str) -> float:
    """Compute entropy of ..."""
    # predictions = 
    return frob(generate_predictions(model, tokenizer, question))


def main() -> None:
    train_df = pd.read_parquet("data/webquestions/webq-train.parquet").head(100)

    model = FlipoutBart.from_pretrained("models/nq-bnn").eval()
    model.to(device)
    tokenizer = BartTokenizer.from_pretrained("models/nq-bnn")

    batch_size = 10
    train_df["frob"] = train_df["question"].apply(lambda q: frob(generate_predictions(model, tokenizer, q, batch_size)))
    print(train_df["frob"].describe())

    # questions = [
    #     "what is padua italy?",
    #     "what is calabria italy?",
    # ]

    # batch_size = 20

    # # for question in train_df["question"]:
    # for question in questions:
    #     answers = generate_predictions(model, tokenizer, question, batch_size)
    #     uncertainty = frob(answers)
    #     print(f"question: {question}\nanswers: {answers}\nnorm: {uncertainty}\n")


if __name__ == "__main__":
    main()
