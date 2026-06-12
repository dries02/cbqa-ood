from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import BatchEncoding, PreTrainedModel, PreTrainedTokenizerBase

from src.eval.loadmodel import load_ensemble, load_tokenizer
from src.eval.utils import majority_vote, vote_entropy
from src.train.trainconfig import MODEL_CONFIGS


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["webquestions", "nq"], required=True)
    parser.add_argument("--model_type", type=str, choices=["t5-large-ssm"], default="t5-large-ssm")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n_ensemble", type=int, default=5)
    parser.add_argument("--use_soft", action=BooleanOptionalAction, required=True)
    parser.add_argument("--n_passes", type=int, default=50)
    parser.add_argument("--fraction", type=float, default=1.0)
    return parser.parse_args()


def generate_answers(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
                     tok_q: BatchEncoding, *, temperature: float, n_passes: int) -> list[str]:
    """Use multinomial sampling to generate various answers to a question."""
    outputs = model.generate(**tok_q, do_sample=True,
                             num_return_sequences=n_passes, num_beams=1, temperature=temperature)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def process_question(models: list[PreTrainedModel], tokenizer: PreTrainedTokenizerBase, question: str,
                     device: torch.device, args: Namespace) -> dict[str, str | float]:
    """Compute TU, AU, EU using sampling from ensemble."""
    tok_q = tokenizer(question, max_length=32, truncation=True, padding="max_length", return_tensors="pt").to(device)
    sum_entropy_per_model = 0
    all_answers = []
    hard_predictions = []

    for model in models:
        answers = generate_answers(model, tokenizer, tok_q, temperature=args.temperature, n_passes=args.n_passes)
        sum_entropy_per_model += vote_entropy(answers)
        all_answers.extend(answers)
        hard_predictions.append(majority_vote(answers))             # each model makes a prediction

    aleatoric = sum_entropy_per_model / len(models)                 # E[H[p]]
    total = vote_entropy(all_answers)                               # H[E[p]]
    epistemic = total - aleatoric                                   # E = T - A
    prediction = majority_vote(hard_predictions)                    # combine predictions from each model

    return {
        "prediction": prediction,
        "tu": total,
        "au": aleatoric,
        "eu": epistemic,
    }


def get_results(models: list[PreTrainedModel], tokenizer: PreTrainedTokenizerBase,
                questions: list[str], prefix: str, device: torch.device, args: Namespace) -> pd.DataFrame:
    """Generate all results to be added to the dataframe."""
    all_results = defaultdict(list)
    for question in tqdm(questions):
        row = process_question(models, tokenizer, prefix + question, device, args)
        for method, result in row.items():
            all_results[method].append(result)

    return pd.DataFrame(all_results)


def main() -> None:
    args = parse_args()
    test_df_path = Path("data") / args.dataset / f"{args.dataset}-test.jsonl"
    test_df = pd.read_json(test_df_path, lines=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suffix = "soft" if args.use_soft else "hard"

    models = load_ensemble(args.dataset, args.model_type, suffix, args.n_ensemble, args.fraction, device)
    tokenizer = load_tokenizer(args.dataset, args.model_type, args.fraction, suffix)
    prefix = MODEL_CONFIGS[args.model_type]["prefix"]

    results = get_results(models, tokenizer, test_df["question"].to_list(), prefix, device, args)

    test_df = test_df.join(results)

    results_path = Path("results") / args.dataset / f"ensemble{args.n_ensemble}-{suffix}-{args.fraction}-sampling.jsonl"
    test_df.to_json(results_path, orient="records", lines=True)


if __name__ == "__main__":
    main()
