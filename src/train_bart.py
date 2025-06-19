import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer

from eval_bart import evaluate
from flipoutbart import FlipoutBart
from qadataset import QADataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_EPOCHS = 20
BATCH_SIZE = 32


def train(model: BartForConditionalGeneration, dataloader: DataLoader, optimizer) -> None:
    """Fit the model on the training set."""
    for epoch in tqdm(range(1, N_EPOCHS+1), desc="Epochs"):
        running_loss = 0

        loop = tqdm(dataloader, desc=f"Epoch {epoch}/{N_EPOCHS}")
        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            loss = outputs.loss

            optimizer.zero_grad()                     # clear out old gradients
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            avg_so_far = running_loss / (loop.n + 1)  # loop.n is batches done so far
            loop.set_postfix(train_loss=f"{avg_so_far:.4f}")


def experiment(
        model: BartForConditionalGeneration,
        tokenizer: BartTokenizer,
        train_df: pd.DataFrame,
        optimizer,
        output_dir: str | None = None,
    ) -> None:
    model.to(device)

    dataset = QADataset(train_df, tokenizer)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE)

    train(model, dataloader, optimizer)
    if output_dir is not None:                  # time to save
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    evaluate(model, tokenizer, dataloader)


def train_bnn():
    train_df = pd.read_parquet("data/webquestions/webq-train.parquet")

    model = FlipoutBart.from_pretrained("facebook/bart-base").eval()                    # disable dropout
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    for p in model.model.parameters():
        p.requires_grad = False

    optimizer = Adam([
        {"params": model.lm_head.mu_weight,  "lr": 7e-3, "weight_decay": 0.0},
        {"params": model.lm_head.rho_weight, "lr": 2e-3, "weight_decay": 0.0},
    ])

    experiment(model, tokenizer, train_df, optimizer, "models/nq-bnn")


def main() -> None:
    train_df = pd.read_parquet("data/webquestions/webq-train.parquet")

    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large").train()  # enable dropout
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

    optimizer = Adam(model.parameters(), lr=1e-5)

    experiment(model, tokenizer, train_df, optimizer, "models/nq-large")


if __name__ == "__main__":
    # train_bnn()
    main()
