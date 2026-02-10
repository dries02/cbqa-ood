import math
from collections import Counter

from src.eval.eval_model import normalize_answer


def variation_ratio(answers: list[str]) -> float:
    """Compute the variation ratio."""
    ans_counter = Counter(map(normalize_answer, answers))
    mode = ans_counter.most_common(1)[0][1]
    return 1 - mode / len(answers)


def vote_entropy(answers: list[str]) -> float:
    """Compute the normalized vote entropy."""
    ans_counter = Counter(map(normalize_answer, answers))
    if len(ans_counter) == 1:
        return 0.0                                      # single answer, very certain

    probs = (p / len(answers) for p in ans_counter.values())
    entropy = -sum(p * math.log(p) for p in probs)
    return entropy / math.log(len(ans_counter))         # normalize in [0,1]


if __name__ == "__main__":
    sample_answers = ["foo", "bar", "baz", "foo"]
    print("variation ratio:", variation_ratio(sample_answers))
    print("vote entropy:", vote_entropy(sample_answers))
