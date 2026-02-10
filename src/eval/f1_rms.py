import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

# ------------------------------------------------------------------
# 1.  Build sparse bag-of-words matrix  (token pattern: \w+)
# ------------------------------------------------------------------
_tok_pat = r"\b\w+\b"            # SQuAD-style: lowercase letters & digits

def _bow_matrix(texts):
    vec = CountVectorizer(lowercase=True, token_pattern=_tok_pat)
    X = vec.fit_transform(texts).tocsr()          # shape (T , V), very sparse
    lengths = X.sum(axis=1).A1                    # token count per row
    return X, lengths

# ------------------------------------------------------------------
# 2.  Pair-wise overlap counts  (min(bow_i , bow_j))
# ------------------------------------------------------------------
def _overlap_matrix(X):
    T = X.shape[0]
    overlaps = np.zeros((T, T), dtype=np.uint16)  # uint16 is plenty for token counts
    for i in range(T):
        Xi = sparse.vstack([X[i]] * T)            # broadcast row i
        overlaps[:, i] = Xi.minimum(X).sum(axis=1).A1
    return overlaps

# ------------------------------------------------------------------
# 3.  Convert to F1 distance matrix
# ------------------------------------------------------------------
def _f1_distance_matrix(texts):
    X, lengths = _bow_matrix(texts)
    ov = _overlap_matrix(X)

    precision = np.divide(
        ov,                             # numerator
        lengths[None, :],               # denominator (broadcast as columns)
        out=np.zeros_like(ov, dtype=float),
        where=lengths[None, :] != 0
    )

    # safe recall = ov / len(gold)
    recall = np.divide(
        ov,
        lengths[:, None],               # broadcast as rows
        out=np.zeros_like(ov, dtype=float),
        where=lengths[:, None] != 0
    )

    den = precision + recall                      # denominator
    f1  = np.divide(
            2 * precision * recall,               # numerator
            den,
            out=np.zeros_like(den),               # what to write where den==0
            where=den > 0                         # only divide where safe
        )
    np.fill_diagonal(f1, 1.0)
    return 1.0 - f1

# ------------------------------------------------------------------
# 4.  Collapse D -> u_RMS
# ------------------------------------------------------------------
def f1_rms_uncertainty(strings: list[str]) -> float:
    if all(not s for s in strings):
        return 1.0

    D = _f1_distance_matrix(strings)
    T = D.shape[0]
    off_diag = D[~np.eye(T, dtype=bool)]
    return np.sqrt((off_diag ** 2).mean())        # your formula
