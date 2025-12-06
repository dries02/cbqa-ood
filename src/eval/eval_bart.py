import re
import string

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, BartTokenizer

from src.train.qadataset import QADatasetEval

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# source: https://github.com/facebookresearch/QA-Overlap/blob/main/evaluate.py
articles_pattern = re.compile(r"\b(a|an|the)\b")
def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text: str) -> str:
        return re.sub(articles_pattern, " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(str.lower(s))))


def evaluate(
        model: BartForConditionalGeneration, tokenizer: BartTokenizer, dataloader: DataLoader) -> None:
    """Evaluate the model on a data set."""
    if not isinstance(dataloader.dataset, QADatasetEval):
        msg = "Dataset should be in evaluation mode so all answers are returned."
        raise TypeError(msg)

    was_training = model.training                       # maybe restore
    model.eval()                                        # no dropout during inference
    em_count = 0

    with torch.no_grad():
        for batch in dataloader:
                # only put inputs and attention mask to device
            batch_gpu = {k: v.to(device) for k, v in batch.items() if k != "labels"}

            pred_ids = model.generate(**batch_gpu, max_length=32, num_beams=1, early_stopping=False)   # greedy decoding

            predictions = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

            for p, g in zip(predictions, batch["labels"], strict=True):
                p_2 = normalize_answer(p)
                em_count += any(p_2 == normalize_answer(gt) for gt in g)          # check if in any ground truth answers

    model.train(was_training)                                                     # revert if it was in train mode before
    return em_count


def main() -> None:
    model = BartForConditionalGeneration.from_pretrained("models/webquestions-mcdropout-large").to(device)

    tokenizer = BartTokenizer.from_pretrained("models/webquestions-mcdropout-large")
    test_df = pd.read_json("data/webquestions/webquestions-test.jsonl", lines=True)

    dev_dataset = QADatasetEval(test_df, tokenizer)
    dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=128, collate_fn=QADatasetEval.collate_fn)

    em = evaluate(model, tokenizer, dataloader)
    print(f"{em} out of {len(test_df)}, {em / len(test_df):.3f}")


if __name__ == "__main__":
    main()
