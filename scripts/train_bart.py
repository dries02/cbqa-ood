from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
from torch.optim import Adam, AdamW, Optimizer
from transformers import BartForConditionalGeneration, BartTokenizer, GenerationConfig

from src.train.flipoutbart import FlipoutBart
from src.train.trainconfig import TrainConfig
from src.train.trainer import Trainer


def parse_args() -> Namespace:
    """Create a parser."""
    parser = ArgumentParser(description="Train a model.")
    parser.add_argument(
        "--dataset", type=str, choices=["nq", "webquestions", "triviaqa"], required=True)
    parser.add_argument("--n_epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    return parser.parse_args()


def train_bart(config: TrainConfig) -> tuple[BartForConditionalGeneration, BartTokenizer, Optimizer]:
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large").train()  # enable dropout
    model.generation_config = GenerationConfig.from_model_config(model.config)           # modern stuff
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    optimizer = AdamW(model.parameters(), lr=config.lr)

    return model, tokenizer, optimizer


def train_bnn() -> tuple[BartForConditionalGeneration, BartTokenizer, Optimizer]:
    model = FlipoutBart.from_pretrained("facebook/bart-base").eval()                  # disable dropout
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    for p in model.model.parameters():                                                # freeze
        p.requires_grad = False

    optimizer = Adam([
        {"params": model.lm_head.mu_weight,  "lr": 5e-3, "weight_decay": 0.0},
        {"params": model.lm_head.rho_weight, "lr": 1e-3, "weight_decay": 0.0},
    ])
    return model, tokenizer, optimizer


def main() -> None:
    """Entry point for training CBQA model. Also asks for command line arguments."""
    config = TrainConfig(**vars(parse_args()))                  # fetch and unpack the __dict__
    ds = config.dataset

    train_df = pd.read_json(Path("data") / ds / (ds + "-train.jsonl"), lines=True)
    dev_df = pd.read_json(Path("data") / ds / (ds + "-dev.jsonl"), lines=True)

    model, tokenizer, optimizer = train_bart(config)

    trainer = Trainer(model, tokenizer, optimizer, train_df, dev_df, config)
    trainer.train()
    trainer.save()


if __name__ == "__main__":
    main()
