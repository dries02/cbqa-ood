import pandas as pd
import torch
from qadataset import QADataset
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, BartTokenizer

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def interact(model: BartForConditionalGeneration, tokenizer: BartTokenizer) -> None:
    """Interact with the model."""
    model.eval()

    with torch.no_grad():
        while (question := input("? ")) != "q":                         # prompt, 'q' to quit
            tok_q = tokenizer(question, max_length=64, truncation=True, padding="max_length", return_tensors="pt")
            pred_ids = model.generate(**tok_q, max_length=64)
            prediction = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)[0]
            print(f"Answer: {prediction}")



def evaluate(model: BartForConditionalGeneration, tokenizer: BartTokenizer, dataloader: DataLoader) -> None:
    """Evaluate the model on a data set."""
    model.eval()
    em_count = 0

    with torch.no_grad():
        for batch in dataloader:
            # Move inputs to the right device
            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Generate predictions greedily
            pred_ids = model.generate(**batch, max_length=32)

            # Decode everything
            questions = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            predictions = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            references = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Print line by line
            for q, p, g in zip(questions, predictions, references, strict=True):
                print(f"Q: {q}")
                print(f"P: {p}")
                print(f"G: {g}")
                print("-" * 80)
                em_count += p == g

    print(f"EM = {em_count}")
    # perhaps also add F1 and other metrics... maybe separate it from the inference

def main() -> None:
    model = BartForConditionalGeneration.from_pretrained("models/nq-bnn")
    tokenizer = BartTokenizer.from_pretrained("models/nq-bnn")
    train_df = pd.read_parquet("data/webquestions/webq-train.parquet").head(100)
    dataset = QADataset(train_df, tokenizer)
    dataloader = DataLoader(dataset, batch_size=1)

    # evaluate(model, tokenizer, dataloader)
    interact(model, tokenizer)


if __name__ == "__main__":
    main()
