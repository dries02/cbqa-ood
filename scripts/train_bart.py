from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
from torch.optim import AdamW, Optimizer
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer, GenerationConfig

from src.train.flipoutbart import FlipoutBart
from src.train.trainconfig import TrainConfig
from src.train.trainer import Trainer


def parse_args() -> Namespace:
    """Create a parser."""
    parser = ArgumentParser(description="Train a model.")
    parser.add_argument("--dataset", type=str, choices=["nq", "webquestions", "triviaqa"], required=True)
    parser.add_argument("--method", type=str, choices=["mcdropout", "flipout"], required=True)
    parser.add_argument("--n_epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def make_bart(config: TrainConfig) -> tuple[BartForConditionalGeneration, BartTokenizer, Optimizer]:
    bartconfig: BartConfig = BartConfig.from_pretrained("facebook/bart-large")
    bartconfig.dropout = config.dropout

    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", config=bartconfig).train()  # enable dropout
    model.generation_config = GenerationConfig.from_model_config(model.config)           # modern stuff
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    optimizer = AdamW(model.parameters(), lr=config.lr)

    return model, tokenizer, optimizer


def make_flipout(config: TrainConfig) -> tuple[BartForConditionalGeneration, BartTokenizer, Optimizer]:
    train_size = sum(1 for _ in Path.open(config.train_path))

    model = FlipoutBart.from_pretrained("facebook/bart-large", train_size).train()
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

    optimizer = AdamW([
    {
        "params": model.model.parameters(),     # Encoder + Decoder
        "lr": 1e-5,
    },
    {
        "params": model.lm_head.parameters(),   # mu_weight + rho_weight
        "lr": 1e-4,                             # Higher LR for new Bayesian layer
        "weight_decay": 0.0,                    # No weight decay on Bayesian params
    }])

    return model, tokenizer, optimizer


def main() -> None:
    """Entry point for training CBQA model. Also asks for command line arguments."""
    config = TrainConfig(**vars(parse_args()))                  # fetch and unpack the __dict__
    train_df = pd.read_json(config.train_path, lines=True)
    dev_df = pd.read_json(config.dev_path, lines=True)

    methods = {"mcdropout": make_bart, "flipout": make_flipout}

    model, tokenizer, optimizer = methods[config.method](config)

    trainer = Trainer(model, tokenizer, optimizer, train_df, dev_df, config)
    trainer.train()


if __name__ == "__main__":
    main()
