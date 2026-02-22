import math
import re
import string
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import cache, reduce

import numpy as np
import torch
from nltk.stem import PorterStemmer
from transformers import PreTrainedModel, PreTrainedTokenizerBase

articles_pattern = re.compile(r"\b(a|an|the)\b")
possessive_pattern = re.compile(r"'s\b")
dash_pattern = re.compile(r"\s*[-]+\s*")
initials_pattern = re.compile(r"\b([a-z])\.\s*")
preposition_pattern = re.compile(r"^(from|in|at|on|by|with|of|to|for|as|into|through|about)\s+")
stemmer = PorterStemmer()

# inspired by https://github.com/facebookresearch/QA-Overlap/blob/main/evaluate.py (with additions)
def _normalize_answer(answer: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    transformations = [
        str.lower,                                                  # remove casing
        lambda t: re.sub(preposition_pattern, "", t),               # leading prepositions
        lambda t: t.replace("&", "and"),
        lambda t: re.sub(dash_pattern, " ", t),                     # added, fixes issues with a -- b == a-b
        lambda t: re.sub(possessive_pattern, "", t),                # added, fixes issues with one 's == one's
        lambda t: re.sub(initials_pattern, r"\1 ", t),              # added, fixes issues with U.S. == U. S.
        lambda t: unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii"),   # added unicode
        lambda t: "".join(ch for ch in t if ch not in string.punctuation),  # remove punctuation
        lambda t: re.sub(articles_pattern, " ", t),                         # remove articles
        lambda t: " ".join(t.split()),                                      # remove whitespace
        lambda t: " ".join(stemmer.stem(w) for w in t.split()),     # added, stemming
    ]

    return reduce(lambda text, fn: fn(text), transformations, answer)


def _normalize_answer_safe(answer: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace. No stemming."""
    transformations = [
        str.lower,
        lambda t: re.sub(preposition_pattern, "", t),               # leading prepositions
        lambda t: t.replace("&", "and"),
        lambda t: re.sub(dash_pattern, " ", t),                     # added, fixes issues with a -- b == a-b
        lambda t: re.sub(possessive_pattern, "", t),                # added, fixes issues with one 's == one's
        lambda t: re.sub(initials_pattern, r"\1 ", t),              # added, fixes issues with U.S. == U. S.
        lambda t: unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii"),   # added unicode
        lambda t: "".join(ch for ch in t if ch not in string.punctuation),
        lambda t: re.sub(articles_pattern, " ", t),
        lambda t: " ".join(t.split()),
    ]

    return reduce(lambda text, fn: fn(text), transformations, answer)

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
    normalized = [_normalize_answer_safe(p) for p in preds]             # find the most common normalized form
    most_common_norm = Counter(normalized).most_common(1)[0][0]
    for orig, norm in zip(preds, normalized, strict=True):              # return the corresponding ORIGINAL prediction
        if norm == most_common_norm:
            return orig

    raise ValueError


def exact_match(pred: str, gold_answers: list[str]) -> bool:
    """Check whether prediction matches any gold answer (after normalization)."""
    pred_clean = _normalize_answer(pred)
    return any(pred_clean == _normalize_answer(gt) for gt in gold_answers)


def f1(pred: str, gold_answers: list[str]) -> float:
    """Compute F1 score as maximum F1 over all gold answers."""
    pred = _normalize_answer(pred)
    gold_answers = map(_normalize_answer, gold_answers)
    return max(_f1_pair(pred, gt) for gt in gold_answers)


def variation_ratio(preds: list[str]) -> float:
    """Compute the variation ratio."""
    ans_counter = Counter(map(_normalize_answer, preds))
    mode = ans_counter.most_common(1)[0][1]
    return 1 - mode / len(preds)


def vote_entropy(preds: list[str]) -> float:
    """Compute the vote entropy."""
    ans_counter = Counter(map(_normalize_answer, preds))
    probs = (p / len(preds) for p in ans_counter.values())
    return -sum(p * math.log(p) for p in probs)


@dataclass(frozen=True)
class NLIResources:
    """Contains a model, tokenizer and device for NLI inference."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: torch.device
    contradict_idx: int         # == 0
    neutral_idx: int            # == 1
    entail_idx: int             # == 2

    def entails(self, lhs: str, rhs: str) -> bool:
        """Check whether lhs entails rhs."""
        if lhs == rhs:
            return True

        inputs = self.tokenizer(lhs, rhs, return_tensors="pt").to(self.device)
        with torch.no_grad():
            pred_logits = self.model(**inputs).logits

        pred = pred_logits.argmax().item()      # might prefer thresholded?
        return pred == self.entail_idx          # neutral seems too lenient


def semantic_entropy(question: str, answers: list[str], nli: NLIResources) -> float:
    """Compute semantic entropy based on Kuhn et al. (https://arxiv.org/pdf/2302.09664)."""
    @cache                                  # cache of inner function is cleared when semantic_entropy returns
    def entails(lhs: str, rhs: str) -> bool:
        return nli.entails(lhs, rhs)

    def is_entailment(ans1: str, ans2: str) -> bool:
        lhs = question + " " + ans1
        rhs = question + " " + ans2
        return entails(lhs, rhs) and entails(rhs, lhs)              # bidirectional entailment

    def entailment_clustering(answers: list[str]) -> list[list[str]]:       # see Algorithm 1 on p.15
        clusters = [[answers[0]]]
        for answer in answers[1:]:
            for cluster in clusters:                        # compare with existing clusters
                if is_entailment(answer, cluster[0]):       # use first sequence as reference... maybe all?
                    cluster.append(answer)
                    break                                   # caveat: assume transitivity...
            else:                                           # semantically distinct
                clusters.append([answer])                   # new semantic class

        return clusters

    clusters = entailment_clustering(answers)
    cluster_sizes = list(map(len, clusters))
    n_clusters = sum(cluster_sizes)
    probs = (c / n_clusters for c in cluster_sizes)
    return -sum(p * math.log(p) for p in probs)


def risk_coverage_curve(uncertainty: list[float], correctness: list[float]) -> tuple[list[float], list[float]]:
    """Compute risk at each coverage level."""
    order = np.argsort(uncertainty, kind="mergesort")
    sorted_correct = correctness[order]
    sorted_uncertainty = uncertainty[order]
    n = len(sorted_correct)
    k = np.arange(1, n + 1, dtype=np.float32)
    coverages = k / n
    risks = 1.0 - (np.cumsum(sorted_correct) / k)

    # average risks within tied uncertainty groups
    _, inverse = np.unique(sorted_uncertainty, return_inverse=True)
    avg_risks = np.zeros_like(risks)
    for i in np.unique(inverse):
        mask = inverse == i
        avg_risks[mask] = risks[mask].mean()

    return coverages, avg_risks

# def risk_coverage_curve(uncertainty: list[float], correctness: list[float]) -> tuple[list[float], list[float]]:
#     """Compute risk at each coverage level."""
#     order = np.argsort(uncertainty, kind="mergesort")  # stable for ties
#     sorted_correct = correctness[order]

#     n = sorted_correct.size
#     k = np.arange(1, n + 1, dtype=np.float32)

#     coverages = k / n
#     risks = 1.0 - (np.cumsum(sorted_correct) / k)       # risk_k = 1 - mean(correct among top-k)
#     return coverages, risks


def compute_aurc(uncertainty: list[float], correctness: list[float]) -> float:
    """Compute the Area Under the Risk-Coverage curve."""
    coverages, risks = risk_coverage_curve(uncertainty, correctness)
    return np.trapezoid(risks, coverages)               # integration over coverage in [0,1]


def compute_e_aurc(uncertainty: list[float], correctness: list[float]) -> float:
    """Compute the Excess Area Under the Risk-Coverage curve."""
    oracle_uncertainty = 1 - correctness                # correct -> 0, wrong -> 1
    return compute_aurc(uncertainty, correctness) - compute_aurc(oracle_uncertainty, correctness)


def _main():
    pass
    # import pandas as pd

    # df = pd.read_json("results/webquestions/mcdropout-hard.jsonl", lines=True).head(200)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").to(device)
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
    # contradict_idx = model.config.label2id["CONTRADICTION"]
    # neutral_idx = model.config.label2id["NEUTRAL"]
    # entail_idx = model.config.label2id["ENTAILMENT"]

    # nli = NLIResources(model, tokenizer, device, contradict_idx, neutral_idx, entail_idx)

    # for _, row in df.iterrows():
    #     question = row["question"]
    #     preds = row["predictions"]
    #     maj_pred = majority_vote(preds)
    #     print("correct" if exact_match(maj_pred, row["answers"]) else "incorrect")

    #     # print("variation ratio:", variation_ratio(preds))
    #     print("vote entropy:", vote_entropy(preds))
    #     se = semantic_entropy(question, preds, nli)
    #     print("semantic entropy: ", se)
    #     print()


if __name__ == "__main__":
    _main()
