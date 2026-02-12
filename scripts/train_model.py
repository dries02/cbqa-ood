from argparse import ArgumentParser, BooleanOptionalAction, Namespace

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
    parser.add_argument("--n_epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--rho", type=float, default=-2.0)          # for flipout
    # parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def make_optimizer(model: AutoModelForSeq2SeqLM, config: AutoConfig) -> Optimizer:
    """Create an optimizer while carefully choosing where weight decay should be applied."""
    yes_decay = []
    no_decay = []
    forbidden = [
        "layer_norm",                   # layer normalization
        "relative_attention_bias",      # positional embeddings
        "shared",                       # embedding
        "mu_weight",                    # Flipout mean
        "rho_weight",                   # Flipout sigma
        ]

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if any(sub in name for sub in forbidden):
            no_decay.append(p)
        else:
            yes_decay.append(p)

    return AdamW(
        [
            {"params": yes_decay, "lr": config.lr},
            {"params": no_decay, "lr": config.lr, "weight_decay": 0.0},
        ])


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


def make_vanilla(config: TrainConfig) -> tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    model_config: AutoConfig = AutoConfig.from_pretrained(config.hf_name)
    if hasattr(model_config, "dropout_rate"):       # T5
        model_config.dropout_rate = config.dropout
    if hasattr(model_config, "dropout"):            # BART
        model_config.dropout = config.dropout

    model = AutoModelForSeq2SeqLM.from_pretrained(config.hf_name, config=model_config).train()
    update_gen_config(model)
    tokenizer = AutoTokenizer.from_pretrained(config.hf_name)

    return model, tokenizer


def make_flipout(config: TrainConfig) -> tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    model_config: AutoConfig = AutoConfig.from_pretrained(config.hf_name)
    if hasattr(model_config, "dropout_rate"):       # T5
        model_config.dropout_rate = config.dropout
    if hasattr(model_config, "dropout"):            # BART
        model_config.dropout = config.dropout

    model = config.flipout_model.from_base_pretrained(config.hf_name, config=model_config, rho=config.rho).train()
    update_gen_config(model)
    tokenizer = AutoTokenizer.from_pretrained(config.hf_name)

    return model, tokenizer


def main() -> None:
    """Entry point for training CBQA model. Also asks for command line arguments."""
    config = TrainConfig(**vars(parse_args()))                  # fetch and unpack the __dict__
    train_df = pd.read_json(config.train_path, lines=True)
    dev_df = pd.read_json(config.dev_path, lines=True)

    methods = {"mcdropout": make_vanilla, "flipout": make_flipout}

    model, tokenizer = methods[config.method](config)
    optimizer = make_optimizer(model, config)

    trainer = Trainer(model, tokenizer, optimizer, train_df, dev_df, config)
    trainer.train()


if __name__ == "__main__":
    main()
