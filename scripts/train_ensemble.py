from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from pathlib import Path

import pandas as pd
from torch.optim import AdamW, Optimizer
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from src.train.trainconfig import TrainConfig
from src.train.trainer import Trainer


def parse_args() -> Namespace:
    """Create a parser."""
    parser = ArgumentParser(description="Train a model.")
    parser.add_argument("--dataset", type=str, choices=["webquestions", "nq"], required=True)
    parser.add_argument("--model", type=str, choices=["bart-large", "t5-large-ssm"], required=True)
    parser.add_argument("--method", type=str, choices=["mcdropout", "flipout"], required=True)
    parser.add_argument("--use_soft_labels", action=BooleanOptionalAction, required=True)
    parser.add_argument("--use_stochastic_labels", type=bool, default=False)
    parser.add_argument("--n_epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--rho", type=float, default=-2.5)          # for flipout
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def update_gen_config(model: AutoModelForSeq2SeqLM) -> None:
    """Update default BART settings for sequence generation. Handling answer length, greedy decoding, and `<bos>`."""
    gen_kwargs = {
        "min_new_tokens": 1,            # prevent empty sequences
        "max_new_tokens": 32,           # max_length also ok with encoder-decoder
        "num_beams": 1,                 # greedy decoding
        "do_sample": False,             # greedy decoding
        "early_stopping": False,        # greedy decoding
        "no_repeat_ngram_size": 0,      # not necessary after changed <bos> handling. see github
    }

    if hasattr(model.config, "forced_bos_token_id"):    # for changed <bos> handling
        gen_kwargs["forced_bos_token_id"] = None

    for key, value in gen_kwargs.items():       # make sure everything gets saved properly in config.json files
        setattr(model.config, key, value)
        setattr(model.generation_config, key, value)


def make_vanilla(config: TrainConfig) -> tuple[AutoModelForSeq2SeqLM, AutoTokenizer, Optimizer]:
    model_config: AutoConfig = AutoConfig.from_pretrained(config.hf_name)
    if hasattr(model_config, "dropout_rate"):       # T5
        model_config.dropout_rate = config.dropout
    if hasattr(model_config, "dropout"):            # BART
        model_config.dropout = config.dropout

    model = AutoModelForSeq2SeqLM.from_pretrained(config.hf_name, config=model_config).train()
    update_gen_config(model)
    tokenizer = AutoTokenizer.from_pretrained(config.hf_name)
    optimizer = AdamW(model.parameters(), lr=config.lr)

    return model, tokenizer, optimizer


def make_flipout(config: TrainConfig) -> tuple[AutoModelForSeq2SeqLM, AutoTokenizer, Optimizer]:
    model = config.flipout_model.from_base_pretrained(config.hf_name, rho=config.rho).train()

    update_gen_config(model)
    tokenizer = AutoTokenizer.from_pretrained(config.hf_name)

    yes_decay = []
    no_decay = []
    forbidden = ["layer_norm", "rho_weight", "mu_weight"]

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if any(sub in name for sub in forbidden):
            no_decay.append(p)
        else:
            yes_decay.append(p)

    optimizer = AdamW([
    {
        "params": yes_decay,
        "lr": 1e-4,
    },
    {
        "params": no_decay,
        "lr": 1e-4,                             # Higher LR for new Bayesian layer
        "weight_decay": 0.0,                    # No weight decay on Bayesian params
    }])

    return model, tokenizer, optimizer


def main() -> None:
    """Entry point for training CBQA model. Also asks for command line arguments."""
    for n in range(1, 5):
        config = TrainConfig(**vars(parse_args()))                  # fetch and unpack the __dict__
        suffix = "soft" if config.use_soft_labels else "hard"

        config.output_dir = Path("models") / f"{config.dataset}-{config.model}-{config.method}-{suffix}-{n}"
        train_df = pd.read_json(config.train_path, lines=True)
        dev_df = pd.read_json(config.dev_path, lines=True)

        methods = {"mcdropout": make_vanilla, "flipout": make_flipout}

        model, tokenizer, optimizer = methods[config.method](config)

        trainer = Trainer(model, tokenizer, optimizer, train_df, dev_df, config)
        trainer.train()

if __name__ == "__main__":
    main()
