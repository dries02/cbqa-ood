import math
import re
import string
from collections import Counter

import numpy as np

# source: https://github.com/facebookresearch/QA-Overlap/blob/main/evaluate.py
articles_pattern = re.compile(r"\b(a|an|the)\b")
def _normalize_answer(answer: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text: str) -> str:
        return re.sub(articles_pattern, " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)               # TODO check is this sufficient?
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(answer.lower())))


def _f1_pair(ans1: str, ans2: str) -> float:
    """Compute the F1 score between two answers. Symmetric and in [0,1]."""
    ans1_toks = ans1.split()
    ans2_toks = ans2.split()

    if not ans1_toks and not ans2_toks:       # both empty, match
        return 1.0
    if not ans1_toks or not ans2_toks:        # one empty, no match
        return 0.0

    overlap = sum((Counter(ans1_toks) & Counter(ans2_toks)).values())     # & takes min
    precision = overlap / len(ans1_toks)
    recall = overlap / len(ans2_toks)
    denom = precision + recall
    return (2 * precision * recall) / denom if denom != 0 else 0.0


def majority_vote(preds: list[str]) -> str:
    """Pick the most common answer by majority vote."""
    normalized = map(_normalize_answer, preds)
    return Counter(normalized).most_common(1)[0][0]


def exact_match(pred: str, gold_answers: list[str]) -> bool:
    """Check whether prediction matches any gold answer (after normalization)."""
    pred_clean = _normalize_answer(pred)
    return any(pred_clean == _normalize_answer(gt) for gt in gold_answers)


def f1(pred: str, gold_answers: list[str]) -> float:
    """Compute F1 score as maximum F1 over all gold answers."""
    pred = _normalize_answer(pred)
    gold_answers = map(_normalize_answer, gold_answers)
    return max(_f1_pair(pred, gt) for gt in gold_answers)


def variation_ratio(answers: list[str]) -> float:
    """Compute the variation ratio."""
    ans_counter = Counter(map(_normalize_answer, answers))
    mode = ans_counter.most_common(1)[0][1]
    return 1 - mode / len(answers)


def vote_entropy(answers: list[str]) -> float:
    """Compute the vote entropy."""
    ans_counter = Counter(map(_normalize_answer, answers))
    probs = (p / len(answers) for p in ans_counter.values())
    return -sum(p * math.log(p) for p in probs)


def risk_coverage_curve(uncertainty: list[float], correctness: list[float]) -> tuple[list[float], list[float]]:
    """Compute risk at each coverage level."""
    order = np.argsort(uncertainty, kind="mergesort")  # stable for ties
    sorted_correct = correctness[order]

    n = sorted_correct.size
    k = np.arange(1, n + 1, dtype=np.float32)

    coverages = k / n
    risks = 1.0 - (np.cumsum(sorted_correct) / k)       # risk_k = 1 - mean(correct among top-k)
    return coverages, risks


def compute_aurc(uncertainty: list[float], correctness: list[float]) -> float:
    """Compute the Area Under the Risk-Coverage curve."""
    coverages, risks = risk_coverage_curve(uncertainty, correctness)
    return np.trapezoid(risks, coverages)               # integration over coverage in [0,1]


def compute_e_aurc(uncertainty: list[float], correctness: list[float]) -> float:
    """Compute the Excess Area Under the Risk-Coverage curve."""
    oracle_uncertainty = 1 - correctness                # correct -> 0, wrong -> 1
    return compute_aurc(uncertainty, correctness) - compute_aurc(oracle_uncertainty, correctness)


if __name__ == "__main__":
    sample_answers = ["foo", "bar", "baz", "foo"]
    print("variation ratio:", variation_ratio(sample_answers))
    print("vote entropy:", vote_entropy(sample_answers))

    pred = "foo foo bar"
    gold_answers = ["foo foo foo bar", "foo bar bar bar"]
    print("f1:", f1(pred, gold_answers))

    np.random.seed(42)
    correctness = np.random.random(size=1000)
    uncertainty = np.random.random(size=1000)
    print(compute_aurc(correctness, uncertainty))
