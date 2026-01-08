import re
import string

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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
        model: AutoModelForSeq2SeqLM, tokenizer: AutoTokenizer, dataloader: DataLoader) -> None:
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
                # max_new_tokens is equivalent to max_length for encoder-decoder; anyway max_length is deprecated
            pred_ids = model.generate(**batch_gpu)
            predictions = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

            for p, g in zip(predictions, batch["labels"], strict=True):
                p_2 = normalize_answer(p)
                em_count += any(p_2 == normalize_answer(gt) for gt in g)          # check if in any ground truth answers

    model.train(was_training)                                                     # revert if it was in train mode
    return em_count
