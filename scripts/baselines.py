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


def compute_uncertainties(
        probs: torch.Tensor, pred_indices: torch.Tensor, tokenizer: BartTokenizer) -> dict[str, list[float]]:
    # compute mean and max entropy
    # compute max token nll
    # compute perplexity
    pred_indices = pred_indices[:, 1:]              # skip decoder start token
    eos_indices = (pred_indices == tokenizer.eos_token_id)
    if not all(eos_indices.sum(dim=-1) == 1):
        msg = "There must be exactly 1 <eos> token per predicted answer."
        raise ValueError(msg)

    eos_indices = eos_indices.int().argmax(dim=-1)  # maximizer is unique
    decoded = tokenizer.batch_decode(pred_indices)
    # print(decoded)
    print(pred_indices)

    return {}


@torch.no_grad()
def compute_baselines(
    model: BartForConditionalGeneration, tokenizer: BartTokenizer, device: torch.device,
    bad_ids: torch.Tensor, questions: list[str], batch_size: int = 128) -> pd.DataFrame:
    """Compute elementary baselines with deterministic model."""
    uncertainties = defaultdict(list)

    for idx in tqdm(range(0, len(questions), batch_size)):
        chunk_qs = questions[idx:idx + batch_size]
        tok_qs = tokenizer(                                 # tokenize each question only once
            chunk_qs, max_length=32, truncation=True, padding="max_length", return_tensors="pt").to(device)

                                                            # greedy decoding
        outputs = model.generate(                           # not logits but *scores* which are post-processed
            **tok_qs, max_new_tokens=32, num_beams=1, do_sample=False, early_stopping=False,
            return_dict_in_generate=True, output_scores=True,
        )

        probs = torch.softmax(torch.stack(outputs.scores), dim=-1)

        for k, v in compute_uncertainties(probs, outputs.sequences, tokenizer).items():
            uncertainties[k].extend(v)

        decoded = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        uncertainties["prediction"].extend(decoded)

    return pd.DataFrame(uncertainties)


def main() -> None:
    config = GenConfig(**vars(parse_args()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BartForConditionalGeneration.from_pretrained(config.model_path).to(device).eval()
    tokenizer = BartTokenizer.from_pretrained(config.model_path)

    # 0 = <bos>, 1 = <pad>, 3 = <unk>, 50264 = <mask>.
    # no more 2 = <eos>, 4 = '.', 50118 = '\n'
    # could use tokenizer.all_special_ids and tokenizer.eos_token_id for this
    bad_ids = torch.tensor([0, 1, 3, 4, 50264], device=device)

    test_df = pd.read_json(config.test_df_path, lines=True)

    results = compute_baselines(model, tokenizer, device, bad_ids, test_df["question"].tolist())
    Path.mkdir(config.answers_dest_path, parents=True, exist_ok=True)
    # pd.concat([test_df, results], axis=1).to_json(
    #     config.answers_dest_path / "baselines.jsonl", orient="records", lines=True, force_ascii=False)


if __name__ == "__main__":
    main()
