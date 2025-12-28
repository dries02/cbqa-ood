from argparse import ArgumentParser, Namespace
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from transformers import BartForConditionalGeneration, BartTokenizer

MAX_Q_LEN = 32
MAX_ANS_LEN = 32


def parse_args() -> Namespace:      # maybe add wrapper class again like elsewhere
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, choices=["mcdropout", "flipout"], required=True)
    parser.add_argument("--dataset", type=str, choices=["nq", "webquestions", "triviaqa"], required=True)
    parser.add_argument("--n_reps", type=int, default=10)
    return parser.parse_args()


@torch.no_grad()
def generate_answer(
        model: BartForConditionalGeneration, tokenizer: BartTokenizer, question: str, device: torch.device,
        n_reps: int) -> dict:
    """Generate an answer token by token while preventing repeated trigrams."""
    tok_q = tokenizer(
        question, max_length=MAX_Q_LEN, truncation=True, padding="max_length", return_tensors="pt").to(device)

    decoder_input_ids = torch.tensor([[tokenizer.eos_token_id]], device=device)
    # ngram_map = defaultdict(set)                    # maps 2-gram prefix to set of tokens that followed
    token_mis = []

    for _ in range(MAX_ANS_LEN - 2):                # excluding <BOS>, <EOS> ... <EOS>
        all_logits = torch.stack([
            model(**tok_q, decoder_input_ids=decoder_input_ids).logits[0, -1, :]
            for _ in range(n_reps)
        ])
        probs = torch.softmax(all_logits, dim=-1)
        mean_probs = probs.mean(dim=0)

        # TODO should I compute entropy before or after masking blocked tokens?

        entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum()          # entropy of the mean

        per_pass_entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)     # entropy of distributions
        mutual_info = entropy - per_pass_entropy.mean()                        # MI = entropy of mean - mean of entropy
        token_mis.append(mutual_info.item())

        prefix = tuple(decoder_input_ids[0, -2:].tolist())
        # blocked = ngram_map[prefix]
        # if blocked:
            # mean_probs[list(blocked)] = 0           # prevent repeated trigrams

        next_token = mean_probs.argmax()
        if next_token == tokenizer.eos_token_id:
            break

        # ngram_map[prefix].add(next_token.item())
        decoder_input_ids = torch.cat([decoder_input_ids, next_token.view(1, 1)], dim=1)

    return {
        "answer": tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True),
        "mean_mi": np.mean(token_mis),
        "max_mi": np.max(token_mis),
    }


def main() -> None:
    args = parse_args()

    model = BartForConditionalGeneration.from_pretrained(f"models/{args.dataset}-{args.model}-large").train()
    tokenizer = BartTokenizer.from_pretrained(f"models/{args.dataset}-{args.model}-large")
    test_df = pd.read_json(f"data/{args.dataset}/{args.dataset}-test.jsonl", lines=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    test_df[["answer", "mean_mi", "max_mi"]] = test_df["question"].apply(
        lambda q: pd.Series(generate_answer(model, tokenizer, q, device, args.n_reps)))
    test_df.to_json(f"results/{args.dataset}/{args.model}-large-token.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    main()
