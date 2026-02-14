import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.eval.utils import exact_match
from src.train.qadataset import QADatasetEval


def evaluate(
        model: AutoModelForSeq2SeqLM, tokenizer: AutoTokenizer, dataloader: DataLoader, device: torch.device) -> None:
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
            pred_ids = model.generate(**batch_gpu)
            predictions = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

            for p, g in zip(predictions, batch["labels"], strict=True):
                em_count += exact_match(p, g)

    model.train(was_training)                                                     # revert if it was in train mode
    return em_count
