import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.eval.utils import exact_match
from src.train.qadataset import QADatasetEval


def evaluate(
        model: AutoModelForSeq2SeqLM, tokenizer: AutoTokenizer, dataloader: DataLoader, device: torch.device) -> dict:
    """Evaluate the model on a data set."""
    if not isinstance(dataloader.dataset, QADatasetEval):
        msg = "Dataset should be in evaluation mode so all answers are returned."
        raise TypeError(msg)

    # was_training = model.training                       # maybe restore
    model.eval()                                        # no dropout during inference
    em_count = 0
    # losses = []

    # em2 = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            # labels = batch["labels"].to(device)
            all_labels = batch["all_labels"]

            pred_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask)

            predictions = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

            for p, g in zip(predictions, all_labels, strict=True):
                em_count += exact_match(p, g)

            # zippy = zip(input_ids, attention_mask, labels, all_labels, strict=True)

            # for input_id, attention, label, all_label in zippy:
                # if len(all_label) != 1:
                    # continue

                # lossy = model(input_ids=input_id.unsqueeze(0), attention_mask=attention.unsqueeze(0), labels=label).loss.item()
                # losses.append(lossy)

                # pred_ids = model.generate(input_ids=input_id.unsqueeze(0), attention_mask=attention.unsqueeze(0))

                # predictions = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)[0]

                # em2 += exact_match(predictions, all_label)

                # question = tokenizer.batch_decode(input_id.unsqueeze(0), skip_special_tokens=True)

                # print(question, predictions, all_label, f"{lossy:.3f}", exact_match(predictions, all_label))

                # print(predictions, all_label)


    model.train()
    # model.train(was_training)                           # revert if it was in train mode
    # import numpy as np
    # losses = np.array(losses)

    # print("mean=", losses.mean(), "sd=", losses.std() / np.sqrt(len(losses)))

    # print(f"{em2=}", f"{mean_loss=}")

    return {
        "em": em_count,
        # "loss": losses.mean(),
    }
