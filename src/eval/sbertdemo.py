import math

import torch
from bert_score import BERTScorer

device = "cuda" if torch.cuda.is_available() else "cpu"
scorer = BERTScorer(model_type="bert-base-uncased", device=device)


def frob(answers: list[str]) -> float:
    r"""Compute the Frobenius norm of a distance matrix.

    Implementing Frobenius norm: ```sqrt(2\sum_{i<j} D_{ij}^2)```.
    """
    answers = list(filter(None, answers))           # filter empty strings (Falsy)

    T = len(answers)
    lhs, rhs = [], []
    for i in range(T):
        for j in range(i+1, T):                     # actual distance so symmetry
            lhs.append(answers[i])
            rhs.append(answers[j])

    _, _, f1 = scorer.score(lhs, rhs, batch_size=128)

    d = 1.0 - f1                                    # actual distance so h(x,x) = 0
    sq = (d * d).sum().item() * 2
    return math.sqrt(sq) / math.sqrt(T * (T - 1))   # normalized


