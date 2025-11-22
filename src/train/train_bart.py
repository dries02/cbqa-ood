from pathlib import Path

import pandas as pd
import torch
from torch.optim import Adam, AdamW, Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer

from src.eval.eval_bart import evaluate
from src.train.flipoutbart import FlipoutBart
from src.train.qadataset import QADataset, eval_collate_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_EPOCHS = 15
BATCH_SIZE = 32


def train(
        model: BartForConditionalGeneration,
        tokenizer: BartTokenizer,
        train_data: DataLoader,
        dev_data: DataLoader,
        optimizer: Optimizer,
    ) -> None:
    """Fit the model on the training set."""
    for epoch in tqdm(range(1, N_EPOCHS+1), desc="Epochs", position=0):
        running_loss = 0

        loop = tqdm(train_data, desc=f"Epoch {epoch}/{N_EPOCHS}", position=1)
        for idx, batch in enumerate(loop, start=1):
            batch_gpu = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch_gpu)

            loss = outputs.loss

            optimizer.zero_grad()                                   # clear out old gradients
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            avg_so_far = running_loss / (loop.n + 1)                # loop.n is batches done so far

            if idx == len(train_data):
                em_count = evaluate(model, tokenizer, dev_data, verbose=False)
                # em_count = "-"
                loop.set_postfix(train_loss=f"{avg_so_far:.4f}", EM=str(em_count))
            else:
                loop.set_postfix(train_loss=f"{avg_so_far:.4f}", EM="-")

def experiment(
        model: BartForConditionalGeneration,
        tokenizer: BartTokenizer,
        train_df: pd.DataFrame,
        dev_df: pd.DataFrame,
        optimizer: Optimizer,
        output_dir: str | None = None,
    ) -> None:
    model.to(device)

    train_dataset = QADataset(train_df, tokenizer, is_train=True)
    train_data = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)

    dev_dataset = QADataset(dev_df, tokenizer, is_train=False)
    dev_data = DataLoader(dev_dataset, shuffle=False, batch_size=BATCH_SIZE, collate_fn=eval_collate_fn)

    train(model, tokenizer, train_data, dev_data, optimizer)
    if output_dir is not None:                  # time to save
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    # evaluate(model, tokenizer, dev_data, verbose=True)


def train_bnn():
    train_df = pd.read_json("data/webquestions/webquestions-train.json")
    dev_df = pd.read_json("data/webquestions/webquestions-dev.json")

    model = FlipoutBart.from_pretrained("facebook/bart-base").eval()                  # disable dropout
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    for p in model.model.parameters():                                                # freeze
        p.requires_grad = False
    # optimizer = Adam(model.lm_head.parameters(), lr=2e-3)
    optimizer = Adam([
        {"params": model.lm_head.mu_weight,  "lr": 5e-3, "weight_decay": 0.0},
        {"params": model.lm_head.rho_weight, "lr": 1e-3, "weight_decay": 0.0},
    ])

    experiment(model, tokenizer, train_df, dev_df, optimizer, "models/nq-bnn")


def main() -> None:
    ds = "nq"

    train_df = pd.read_json(Path("data") / ds / (ds + "-train.jsonl"), lines=True)
    dev_df = pd.read_json(Path("data") / ds / (ds + "-dev.jsonl"), lines=True)

    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large").train()  # enable dropout
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

    optimizer = AdamW(model.parameters(), lr=1e-5)

    experiment(model, tokenizer, train_df, dev_df, optimizer, output_dir=Path("models") / (ds + "-large"))


if __name__ == "__main__":
    # train_bnn()
    main()
