import torch
from torch.nn.functional import softmax
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
    all_logits = []
    with torch.no_grad():
         for _ in range(n_samples):
            gen = model.generate(**enc, max_length=32, output_scores=True, return_dict_in_generate=True)
            logits = torch.stack(gen.scores, dim=1).squeeze(0)
            all_logits.append(logits)
            decoded = tokenizer.batch_decode(gen.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            print(decoded[0])


def main() -> None:
    model = BartForConditionalGeneration.from_pretrained("models/nq")
    tokenizer = BartTokenizer.from_pretrained("models/nq")

    mc_dropout(model, tokenizer, input())
    # while (question := input("? ")) != "q":
        # entropy(model, tokenizer, question)
    # mc_dropout(model, tokenizer, question)


if __name__ == "__main__":
    main()
