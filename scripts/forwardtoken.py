from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.eval.loadmodel import load_stochastic_model, load_tokenizer
from src.train.trainconfig import MODEL_CONFIGS

MAX_Q_LEN = 32
MAX_ANS_LEN = 32


@dataclass
class ResultsTracker:
    prediction: str = ""
    totals: list[float] = field(default_factory=list)       # predictive entropy
    aleatorics: list[float] = field(default_factory=list)   # expected entropy
    epistemics: list[float] = field(default_factory=list)   # MI = total - aleatoric

    def add_token_entropies(self, total: float, aleatoric: float) -> None:
        self.totals.append(total)
        self.aleatorics.append(aleatoric)
        self.epistemics.append(total - aleatoric)

    def summary(self) -> dict[str, float | str | list[float]]:
        return {
            "prediction": self.prediction,
            "mean_mi": float(np.mean(self.epistemics)),
            "max_mi": max(self.epistemics),
            "all_mi": self.epistemics,

            "mean_au": float(np.mean(self.aleatorics)),
            "max_au": max(self.aleatorics),
            "all_au": self.aleatorics,

            "mean_pred_entropy": float(np.mean(self.totals)),
            "max_pred_entropy": max(self.totals),
        }


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["webquestions", "nq"], required=True)
    parser.add_argument("--model", type=str, choices=["bart-large", "t5-large-ssm"], required=True)
    parser.add_argument("--method", type=str, choices=["mcdropout", "flipout"], required=True)
    parser.add_argument("--use_soft", action=BooleanOptionalAction, required=True)
    parser.add_argument("--n_reps", type=int, default=30)
    return parser.parse_args()


@torch.no_grad()
def generate_answer(
        model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, question: str, device: torch.device,
        n_reps: int) -> dict[str, str | float]:
    tok_q = tokenizer(
        question, max_length=MAX_Q_LEN, truncation=True, padding="max_length", return_tensors="pt").to(device)
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]], device=device)

    tok_q_batched = {k: v.repeat(n_reps, 1) for k, v in tok_q.items()}

    tracker = ResultsTracker()

    for _ in range(MAX_ANS_LEN):
        decoder_batched = decoder_input_ids.repeat(n_reps, 1)  # (n_reps, seq_len)
        all_logits = model(**tok_q_batched, decoder_input_ids=decoder_batched).logits[:, -1, :]  # (n_reps, vocab)

        all_probs = torch.softmax(all_logits, dim=-1)
        mean_probs = all_probs.mean(dim=0)

        mean_probs = all_probs.mean(dim=0)
        pred_entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum()          # TU := H[E[p]]

        aleatoric = -(all_probs * torch.log(all_probs + 1e-10)).sum(dim=-1).mean()  # AU := E[H[p]]

        tracker.add_token_entropies(pred_entropy.item(), aleatoric.item())

        next_token = mean_probs.argmax()          # Next token from mean probs
        decoder_input_ids = torch.cat([decoder_input_ids, next_token.view(1, 1)], dim=1)

        if next_token == tokenizer.eos_token_id:
            break

    tracker.prediction = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)

    return tracker.summary()


def get_results(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
                questions: list[str], prefix: str, device: torch.device, n_reps: int) -> pd.DataFrame:
    all_results = defaultdict(list)
    for question in tqdm(questions):
        row = generate_answer(model, tokenizer, prefix + question, device, n_reps)
        for method, result in row.items():
            all_results[method].append(result)

    return pd.DataFrame(all_results)


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suffix = "soft" if args.use_soft else "hard"
    model_path = Path("models") / f"{args.dataset}-{args.model}-{args.method}-{suffix}-0"
    model = load_stochastic_model(args.method, args.model, model_path, device)

    tokenizer = load_tokenizer(args.dataset, args.model, suffix)
    test_df = pd.read_json(f"data/{args.dataset}/{args.dataset}-test.jsonl", lines=True)
    prefix = MODEL_CONFIGS[args.model]["prefix"]

    results = get_results(model, tokenizer, test_df["question"].to_list(), prefix, device, args.n_reps)
    test_df = test_df.join(results)

    results_path = Path("results") / args.dataset / f"{args.method}-{suffix}-token.jsonl"
    test_df.to_json(results_path, orient="records", lines=True)


if __name__ == "__main__":
    main()
