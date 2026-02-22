from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from scripts.forwardtoken import ResultsTracker
from src.train.trainconfig import MODEL_CONFIGS

MAX_Q_LEN = 32
MAX_ANS_LEN = 32


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["webquestions", "nq"], required=True)
    parser.add_argument("--model", type=str, choices=["t5-large-ssm"], required=True)
    parser.add_argument("--use_soft", action=BooleanOptionalAction, required=True)
    parser.add_argument("--n_ensemble", type=int, default=5)
    return parser.parse_args()


@torch.no_grad()
def generate_answer_ensemble(models: list[PreTrainedModel], tokenizer: PreTrainedTokenizerBase,
                             question: str, device: torch.device) -> dict[str, float | str]:
    tok_q = tokenizer(
        question, max_length=MAX_Q_LEN, truncation=True, padding="max_length", return_tensors="pt").to(device)

    decoder_input_ids = torch.tensor([[models[0].config.decoder_start_token_id]], device=device)

    tracker = ResultsTracker()

    for _ in range(MAX_ANS_LEN):
        all_probs = []

        for model in models:
            logits = model(**tok_q, decoder_input_ids=decoder_input_ids).logits[:, -1, :]   # (1, vocab)
            probs = torch.softmax(logits, dim=-1).squeeze(0)    # (vocab,)
            all_probs.append(probs)

        all_probs = torch.stack(all_probs)                      # (n_ensemble, vocab)

        mean_probs = all_probs.mean(dim=0)
        pred_entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum()          # TU := H[E[p]]

        aleatoric = -(all_probs * torch.log(all_probs + 1e-10)).sum(dim=-1).mean()  # AU := E[H[p]]

        tracker.add_token_entropies(pred_entropy.item(), aleatoric.item())

        next_token = mean_probs.argmax()          # Next token from mean probs
        decoder_input_ids = torch.cat([decoder_input_ids, next_token.view(1, 1)], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    tracker.prediction = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)
    return tracker.summary()


def get_results(models: list[PreTrainedModel], tokenizer: PreTrainedTokenizerBase,
                questions: list[str], prefix: str, device: torch.device) -> pd.DataFrame:
    all_results = defaultdict(list)
    for question in tqdm(questions):
        row = generate_answer_ensemble(models, tokenizer, prefix + question, device)
        for method, result in row.items():
            all_results[method].append(result)

    return pd.DataFrame(all_results)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suffix = "soft" if args.use_soft else "hard"

    model_paths = [
        Path("models") / f"{args.dataset}-{args.model}-mcdropout-{suffix}-0.1-{i}"
        for i in range(args.n_ensemble)
    ]

    for path in model_paths:
        if not path.exists():
            msg = f"Model not found: {path}"
            raise FileNotFoundError(msg)

    print(f"Loading {args.n_ensemble} models...")
    models = [
        AutoModelForSeq2SeqLM.from_pretrained(path).eval().to(device)
        for path in model_paths
    ]
    print(f"Models loaded. GPU memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    tokenizer = AutoTokenizer.from_pretrained(model_paths[0])
    test_df = pd.read_json(f"data/{args.dataset}/{args.dataset}-test.jsonl", lines=True)
    prefix = MODEL_CONFIGS[args.model]["prefix"]

    results = get_results(models, tokenizer, test_df["question"].to_list(), prefix, device)
    test_df = test_df.join(results)

    results_path = Path("results") / args.dataset / f"ensemble{args.n_ensemble}-{suffix}-0.1-token.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_json(results_path, orient="records", lines=True)


if __name__ == "__main__":
    main()
