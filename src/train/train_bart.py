import pandas as pd
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer

from src.data.qadataset import QADataset
from src.eval.eval_bart import evaluate

N_EPOCHS = 30
BATCH_SIZE = 8


def train(model: BartForConditionalGeneration, dataloader: DataLoader) -> None:
    """Fit the model on the training set."""
    optimizer = AdamW(
        model.parameters(),
        lr=1e-5,
        weight_decay=0.01,
        eps=1e-8,
    )

    for epoch in tqdm(range(1, N_EPOCHS+1), desc="Epochs"):
        model.train()
        running_loss = 0

        loop = tqdm(dataloader, desc=f"Epoch {epoch}/{N_EPOCHS}")
        for batch in loop:
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
        output_dir: str | None = None,
    ) -> None:

    train_df = pd.read_parquet("data/webquestions/webq-train.parquet").head()
    dataset = QADataset(train_df, tokenizer)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE)

    train(model, dataloader)
    if output_dir is not None:                  # time to save
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    evaluate(model, tokenizer, dataloader)


def main() -> None:
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    experiment(model, tokenizer, "models/nq")


if __name__ == "__main__":
    main()
