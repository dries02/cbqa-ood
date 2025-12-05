import pandas as pd
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer

from src.eval.eval_bart import evaluate
from src.train.qadataset import QADatasetEval, QADatasetTrain
from src.train.trainconfig import TrainConfig


class Trainer:
    """"Handles model training."""

    def __init__(
        self,
        model: BartForConditionalGeneration,
        tokenizer: BartTokenizer,
        optimizer: Optimizer,
        train_df: pd.DataFrame,
        dev_df: pd.DataFrame,
        config: TrainConfig,
    ) -> None:
        """Create a trainer."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.config = config

        train_dataset = QADatasetTrain(train_df, tokenizer, use_stochastic_labels=True)
        self.train_data = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size)

        dev_dataset = QADatasetEval(dev_df, tokenizer)
        self.dev_data = DataLoader(
            dev_dataset, shuffle=False, batch_size=128, collate_fn=QADatasetEval.collate_fn)

    def train(self) -> None:
        """Train the model."""
        best_em = 0
        epochs_no_improvement = 0

        for epoch in tqdm(range(1, self.config.n_epochs+1), desc="Epochs", position=0):
            running_loss = 0

            loop = tqdm(self.train_data, desc=f"Epoch {epoch}/{self.config.n_epochs}", position=1)
            for idx, batch in enumerate(loop, start=1):
                batch_gpu = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch_gpu)

                loss = outputs.loss

                self.optimizer.zero_grad()                                      # clear out old gradients
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                avg_so_far = running_loss / idx                                 # batches done so far

                if idx == len(self.train_data):
                    em_count = evaluate(self.model, self.tokenizer, self.dev_data)
                    loop.set_postfix(train_loss=f"{avg_so_far:.4f}", EM=str(em_count))

                    if em_count > best_em:                                      # found better, save immediately
                        best_em = em_count
                        epochs_no_improvement = 0
                        self.save()
                    elif epochs_no_improvement + 1 == self.config.patience:     # patience ran out, stop early
                        print(f"\nEarly stopping at epoch {epoch}."
                              f"Best EM: {best_em} at epoch {epoch - self.config.patience}.")
                        return
                    else:
                        epochs_no_improvement += 1
                else:
                    loop.set_postfix(train_loss=f"{avg_so_far:.4f}", EM="-")

    def save(self) -> None:
        """Save the model."""
        if self.config.output_dir is not None:
            self.model.save_pretrained(self.config.output_dir)
            self.tokenizer.save_pretrained(self.config.output_dir)
