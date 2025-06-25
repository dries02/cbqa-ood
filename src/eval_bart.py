import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, BartTokenizer

from qadataset import QADataset

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



def evaluate(model: BartForConditionalGeneration, tokenizer: BartTokenizer, dataloader: DataLoader, *, verbose=True) -> None:
    """Evaluate the model on a data set."""
    if dataloader.dataset.is_train:                 # from QADataset.is_train
        msg = "Dataset should be in evaluation mode so all answers are returned."
        raise ValueError(msg)

    tmp_flag = model.training                       # maybe restore
    model.eval()
    em_count = 0

    with torch.no_grad():
        for batch in dataloader:
            # only put inputs and attention mask to device
            batch_gpu = {k: v if k == "labels" else v.to(device) for k, v in batch.items()}

            pred_ids = model.generate(**batch_gpu, max_length=32)               # greedy decoding

            questions = tokenizer.batch_decode(batch_gpu["input_ids"], skip_special_tokens=True)
            predictions = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

            for q, p, g in zip(questions, predictions, batch_gpu["labels"], strict=True):
                if verbose:
                    print(f"Q: {q}")
                    print(f"P: {p}")
                    print(f"G: {g}")
                    print("-" * 80)
                em_count += p in g              # check if in any ground truth answers
    if verbose:
        print(f"EM = {em_count}")

    model.training = tmp_flag                   # restore
    return em_count


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
