import math
from functools import cache
from itertools import combinations

from bert_score import BERTScorer
from tqdm import tqdm


def frob(answers: list[str]) -> float:
    """Compute the Frobenius norm of a similarity matrix."""
    answers = sorted(answers)           # so (x,y) and (y,x) hit the same cache entry
    scorer = BERTScorer(model_type="bert-base-uncased")

    @cache                              # actual distance so symmetry
    def bert_distance(a: str, b: str) -> float:
        _, _, f1 = scorer.score([a], [b])
        f1 = f1[0].item()
        return 1.0 - f1

    total_distance = 0
    T = len(answers)
    total_pairs = T * (T - 1) // 2      # symmetric matrix, zero diagonal

    # implementing Frobenius norm: sqrt(2\sum_{i<j} s_{ij}^2)
    for i, j in tqdm(combinations(range(T), 2), total=total_pairs):
        d = bert_distance(answers[i], answers[j])
        total_distance += d * d

    total_distance *= 2                 # symmetry
    total_distance = math.sqrt(total_distance)
    max_score = math.sqrt(T * (T - 1))
    return total_distance / max_score   # normalized
