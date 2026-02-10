from pathlib import Path

import pandas as pd

from scripts.train_model import make_flipout, make_optimizer, make_vanilla, parse_args
from src.train.trainconfig import TrainConfig
from src.train.trainer import Trainer


def main() -> None:
    """Entry point for training CBQA model. Also asks for command line arguments."""
    for n in range(1, 5):
        config = TrainConfig(**vars(parse_args()))                  # fetch and unpack the __dict__
        suffix = "soft" if config.use_soft_labels else "hard"

        config.output_dir = Path("models") / f"{config.dataset}-{config.model}-{config.method}-{suffix}-{n}"
        train_df = pd.read_json(config.train_path, lines=True)
        dev_df = pd.read_json(config.dev_path, lines=True)

        methods = {"mcdropout": make_vanilla, "flipout": make_flipout}

        model, tokenizer = methods[config.method](config)
        optimizer = make_optimizer(model, config)

        trainer = Trainer(model, tokenizer, optimizer, train_df, dev_df, config)
        trainer.train()

if __name__ == "__main__":
    main()
