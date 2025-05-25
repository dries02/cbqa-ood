import math
from functools import cache
from itertools import combinations
from pathlib import Path

from bert_score import BERTScorer
from tqdm import tqdm

with Path.open("answerscalabria") as f:
    answers = sorted(f.read().splitlines())

scorer = BERTScorer(model_type="bert-base-uncased")

@cache                          # actual distance so symmetry
def bert_distance(a: str, b: str) -> float:
    # # sort the pair so (x,y) and (y,x) hit the same cache entry
    # x, y = (a, b) if a <= b else (b, a)
    _, _, f1 = scorer.score([a], [b])
    f1 = f1[0].item()
    return 1.0 - f1


total_distance = 0
N = len(answers)
total_pairs = N*(N-1) // 2      # symmetric matrix

# implementing Frobenius norm: sqrt(2\sum_{i<j} D_{ij}^2)
for i, j in tqdm(combinations(range(N), 2), total=total_pairs):
    d = bert_distance(answers[i], answers[j])
    total_distance += d * d

# for a_i, a_j in tqdm(combinations(answers, 2)):
#     # _, _, f1 = scorer.score([a_i], [a_j])
#     # d_ij = 1 - f1
#     d_ij = bert_distance(a_i, a_j)
#     total_distance += d_ij ** 2

total_distance *= 2
total_distance = math.sqrt(total_distance)
print(f"{total_distance:.3f}")

