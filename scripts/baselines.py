from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.train.trainconfig import MODEL_CONFIGS


@dataclass
class GenConfig:
    """Configuration for generating answers based on parsed arguments."""

    dataset: str
    model: str
    use_soft: bool
    model_path: Path = field(init=False)
    test_df_path: Path = field(init=False)
    answers_dest_path: Path = field(init=False)

    def __post_init__(self) -> None:
        """Set some directories."""
        suffix = "soft" if self.use_soft else "hard"
        self.model_path = Path("models") / f"{self.dataset}-{self.model}-mcdropout-{suffix}-0"
        self.test_df_path = Path("data") / self.dataset / f"{self.dataset}-test.jsonl"
        self.answers_dest_path = Path("results") / self.dataset


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["webquestions", "nq"], required=True)
    parser.add_argument("--model", type=str, choices=["bart-large", "t5-large-ssm"], required=True)
    parser.add_argument("--use_soft", action=BooleanOptionalAction, required=True)
    return parser.parse_args()


def validate_sequences(probs: torch.Tensor, pred_ids: torch.Tensor,
                       model: AutoModelForSeq2SeqLM, tokenizer: AutoTokenizer) -> None:
    """Validate some preconditions. This guarantees <eos><answer><eos><pad> without strange tokens."""
    if not (pred_ids[:, 0] == model.generation_config.decoder_start_token_id).all():
        msg = "The first token must be the decoder start token."
        raise ValueError(msg)                       # first token must be decoder start token

    seq = pred_ids[:, 1:]                           # skip decoder start token
    probs = probs.transpose(0, 1)
    if probs.shape[:2] != seq.shape:
        msg = f"Shape mismatch: probs {probs.shape[:2]} vs pred_ids {seq.shape}"
        raise ValueError(msg)                       # dimensions must align

    eos_mask = (seq == tokenizer.eos_token_id)
    if not (eos_mask.sum(dim=-1) == 1).all():
        return

    eos_pos = eos_mask.int().argmax(dim=-1)
    pos = torch.arange(seq.shape[1], device=seq.device)
    after_eos = pos > eos_pos.unsqueeze(1)
    if after_eos.any() and not (seq[after_eos] == tokenizer.pad_token_id).all():
        msg = "Non-PAD tokens found after EOS."
        raise ValueError(msg)                       # <pad> only after <eos>


def compute_uncertainties(probs: torch.Tensor, pred_ids: torch.Tensor,
                          model: AutoModelForSeq2SeqLM, tokenizer: AutoTokenizer) -> dict[str, list[float]]:
    validate_sequences(probs, pred_ids, model, tokenizer)
    pred_ids = pred_ids[:, 1:]                      # (B, L + 1) -> (B, L)
    probs = probs.transpose(0, 1)                   # (L, B, V)  -> (B, L, V)

    mask: torch.Tensor = pred_ids != tokenizer.pad_token_id

    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    mean_entropy = (entropy * mask).sum(dim=-1) / mask.sum(dim=-1)
    entropy[~mask] = float("-inf")
    max_entropy = entropy.max(dim=-1).values  # noqa: PD011

    token_probs = probs.gather(-1, pred_ids.unsqueeze(-1)).squeeze(-1)
    token_nll = -torch.log(token_probs + 1e-9)
    token_nll[~mask] = float("-inf")
    max_token_nll = token_nll.max(dim=-1).values  # noqa: PD011

    token_nll[~mask] = 0
    mean_nll = (token_nll * mask).sum(dim=-1) / mask.sum(dim=-1)
    perplexity = torch.exp(mean_nll)

    return {
        "mean_entropy": mean_entropy.tolist(),
        "max_entropy": max_entropy.tolist(),
        "max_token_nll": max_token_nll.tolist(),
        "perplexity": perplexity.tolist(),
    }


@torch.no_grad()
def compute_baselines(
    model: AutoModelForSeq2SeqLM, tokenizer: AutoTokenizer, device: torch.device,
    questions: list[str], batch_size: int = 128) -> pd.DataFrame:
    """Compute elementary baselines with deterministic model."""
    uncertainties = defaultdict(list)

    for idx in tqdm(range(0, len(questions), batch_size)):
        chunk_qs = questions[idx:idx + batch_size]
        tok_qs = tokenizer(                                 # tokenize each question only once
            chunk_qs, max_length=32, truncation=True, padding="max_length", return_tensors="pt").to(device)

                                                            # not logits but *scores* which are post-processed
        outputs = model.generate(**tok_qs, return_dict_in_generate=True, output_scores=True)

        probs = torch.softmax(torch.stack(outputs.scores), dim=-1)  # make it a probability distribution

        for k, v in compute_uncertainties(probs, outputs.sequences, model, tokenizer).items():
            uncertainties[k].extend(v)

        decoded = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        uncertainties["prediction"].extend(decoded)

    return pd.DataFrame(uncertainties)


def main() -> None:
    config = GenConfig(**vars(parse_args()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_path).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    test_df = pd.read_json(config.test_df_path, lines=True)
    prefix = MODEL_CONFIGS[config.model]["prefix"]
    questions = test_df["question"].apply(lambda q: prefix + q).tolist()
    results = compute_baselines(model, tokenizer, device, questions)
    Path.mkdir(config.answers_dest_path, parents=True, exist_ok=True)
    suffix = "soft" if config.use_soft else "hard"
    pd.concat([test_df, results], axis=1).to_json(
        config.answers_dest_path / f"baselines-{suffix}.jsonl", orient="records", lines=True, force_ascii=False)


if __name__ == "__main__":
    main()
