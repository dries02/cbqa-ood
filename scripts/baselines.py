from argparse import ArgumentParser, Namespace
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer


@dataclass
class GenConfig:
    """Configuration for generating answers based on parsed arguments."""

    dataset: str
    model_path: Path = field(init=False)
    test_df_path: Path = field(init=False)
    answers_dest_path: Path = field(init=False)

    def __post_init__(self) -> None:
        """Set some directories."""
        self.model_path = Path("models") / f"{self.dataset}-mcdropout-large"
        self.test_df_path = Path("data") / self.dataset / f"{self.dataset}-test.jsonl"
        self.answers_dest_path = Path("results") / self.dataset


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["nq", "webquestions", "triviaqa"], required=True)
    return parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 0 = <bos>, 1 = <pad>, 2 = <eos>, 3 = <unk> 4 = '.', 50118 = '\n', 50264 = <mask>
BAD_IDS = torch.tensor([0, 1, 2, 3, 4, 50118, 50264], device=DEVICE)

def compute_uncertainties(probs_per_token: torch.Tensor) -> dict[str, list[float]]:
    mask = ~torch.isin(probs_per_token.indices.T, BAD_IDS)
    masked_probs = probs_per_token.values.T * mask
    masked_probs_for_min = masked_probs.clone()
    masked_probs_for_min[~mask] = 1.0                   # so they don't affect min

    mask_counts = mask.sum(dim=1)

    mean_probs = torch.where(
        mask_counts > 0,
        masked_probs.sum(dim=1) / mask_counts,
        torch.zeros_like(mask_counts, dtype=masked_probs.dtype),  # results in uncertainty = 1.0
    )

    min_probs = torch.where(
        mask_counts > 0,
        masked_probs_for_min.min(dim=1).values,
        torch.zeros_like(mask_counts, dtype=masked_probs.dtype),  # results in uncertainty = 1.0
    )

    return {                # take complement, so higher probability means lower uncertainty
        "mean_sp": (1 - mean_probs).tolist(),
        "min_sp": (1 - min_probs).tolist(),
    }

@torch.no_grad()
def compute_baselines(
    model: BartForConditionalGeneration, tokenizer: BartTokenizer, device: torch.device,
    questions: list[str], batch_size: int = 128) -> pd.DataFrame:
    """Compute elementary baselines with deterministic model."""
    uncertainties = defaultdict(list)

    for idx in tqdm(range(0, len(questions), batch_size)):
        chunk_qs = questions[idx:idx + batch_size]
        tok_qs = tokenizer(                                 # tokenize each question only once
            chunk_qs, max_length=32, truncation=True, padding="max_length", return_tensors="pt").to(device)
        outputs = model.generate(
            **tok_qs, max_length=32, num_beams=1, do_sample=False, early_stopping=False,         # greedy decoding
            return_dict_in_generate=True, output_scores=True,
            # output_logits=True,
        )

        probs_per_token = torch.softmax(torch.stack(outputs.scores), dim=-1).max(dim=-1)

        for k, v in compute_uncertainties(probs_per_token).items():
            uncertainties[k].extend(v)

        decoded = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        uncertainties["prediction"].extend(decoded)
    return pd.DataFrame(uncertainties)


def main() -> None:
    config = GenConfig(**vars(parse_args()))

    model = BartForConditionalGeneration.from_pretrained(config.model_path).to(DEVICE).eval()
    tokenizer = BartTokenizer.from_pretrained(config.model_path)

    test_df = pd.read_json(config.test_df_path, lines=True)

    results = compute_baselines(model, tokenizer, DEVICE, test_df["question"].tolist())
    Path.mkdir(config.answers_dest_path, parents=True, exist_ok=True)
    pd.concat([test_df, results], axis=1).to_json(
        config.answers_dest_path / "baselines.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    main()
