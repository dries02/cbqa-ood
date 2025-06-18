from pathlib import Path

import torch
from sbertdemo import frob
from torch.nn.functional import softmax
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer


# maybe wrap this in a OOD detector clasS?
def entropy(model, tokenizer, question: str):
    model.eval()
    with torch.no_grad():
            tok_q = tokenizer(question, max_length=64, truncation=True, padding="max_length", return_tensors="pt")

            gen = model.generate(**tok_q, max_length=32, output_scores=True, return_dict_in_generate=True)

            scores = torch.stack(gen.scores, dim=1).squeeze(0)  # remove batch dim
            eps = 1e-12
            probs = softmax(scores, dim=-1).clamp(min=eps)      # no log(0)
            ent_per_tok = -(probs * probs.log()).sum(dim=-1)
            ent = ent_per_tok.mean().item()
            decoded = tokenizer.batch_decode(gen.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            print(ent)
            print("Generated answer:", decoded[0])


def mc_dropout(model: BartForConditionalGeneration, tokenizer: BartTokenizer, question: str, n_samples: int = 100):
    """Implement Monte Carlo dropout."""
    # the BART model already has dropout implemented, we just need to enable at inference time
    model.train()                      # turn on all F.dropout
    for p in model.parameters():
        p.requires_grad = False

    enc = tokenizer(question, return_tensors="pt", max_length=64, truncation=True, padding="max_length")
    # all_logits = []
    answers = []
    with torch.no_grad():
         for _ in tqdm(range(n_samples)):
            # gen = model.generate(**enc, max_length=32)
            gen = model.generate(**enc, max_length=32, output_scores=True, return_dict_in_generate=True)
            # logits = torch.stack(gen.scores, dim=1).squeeze(0)
            # all_logits.append(logits)
            decoded = tokenizer.batch_decode(gen.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            # print(decoded[0])
            answers.append(decoded[0])

    return answers

def store(question: str, answers: list[str], score: float) -> None:
    outname = Path("mc-answers") / question.replace(" ", "").removesuffix("?")
    with Path.open(outname, "w") as f:
        f.write(f"Question: {question}\n")
        for answer in answers:
            f.write(answer + "\n")

        f.write(f"\nFrobenius: {score:.3f}")

def main() -> None:
    model = BartForConditionalGeneration.from_pretrained("models/nq")
    tokenizer = BartTokenizer.from_pretrained("models/nq")

    question = "who is kobe bryant?"
    answers = mc_dropout(model, tokenizer, question)
    score = frob(answers)


    store(question, answers, score)


if __name__ == "__main__":
    main()
