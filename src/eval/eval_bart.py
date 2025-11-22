import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, BartTokenizer

from src.train.qadataset import QADataset, eval_collate_fn

import ast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import string
import re

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


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

                g_list = ast.literal_eval(g)
                p_2 = normalize_answer(p)
                if verbose:
                    print(f"Q: {q}")
                    print(f"P: {p}")
                    print(f"G: {g}")
                    if not any(p_2 == normalize_answer(gt) for gt in g_list):
                        print("REJECTED")
                    else:
                        print("ACCEPTED")
                    print("-" * 80)

                # em_count += p in g
                


                # if p in g_list and not any(p_2 == normalize_answer(gt) for gt in g_list):
                    # print(f"Type of g: {type(g_list)}")
                    # print(f"G value: {g_list}")
                    # raise ValueError
                # print(p_2)
                # print()
                em_count += any(p_2 == normalize_answer(gt) for gt in g_list)              # check if in any ground truth answers
    if verbose:
        print(f"EM = {em_count}")

    model.training = tmp_flag                   # restore
    return em_count


def main() -> None:
    model = BartForConditionalGeneration.from_pretrained("models/nq-large").to(device)

    tokenizer = BartTokenizer.from_pretrained("models/nq-large")
    test_df = pd.read_parquet("data/nq/nq-merged.parquet")
    test_df = test_df[test_df["labels"] == "far-ood"]
    print(len(test_df))
    # test_df = pd.read_json("nq-dev.jsonl", lines=True)

    dev_dataset = QADataset(test_df, tokenizer, is_train=False)
    dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=64, collate_fn=eval_collate_fn)

    em = evaluate(model, tokenizer, dataloader, verbose=True)
    print(em)
    # interact(model, tokenizer)


if __name__ == "__main__":
    main()
