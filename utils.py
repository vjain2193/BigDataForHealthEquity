# utils.py
import numpy as np
import pandas as pd
import torch
from torch import nn

# -----------------------------------------------------------------------------
# Relation vocabulary (single source of truth)
# -----------------------------------------------------------------------------
RELATION_TO_TYPE = {
    "mentioned": 0,
    "replied_to": 1,
    "co_mention": 2,
    "co_reply": 3,
    "same_conversation": 4,
}
TYPE_TO_RELATION = {v: k for k, v in RELATION_TO_TYPE.items()}


def normalize_relation_series(s: pd.Series) -> pd.Series:
    """
    Normalize a pandas Series of relation names -> integer type ids.
    Unseen/invalid values become <NA>; caller can drop them.
    """
    return (
        s.astype(str)
         .str.strip()
         .str.lower()
         .map(RELATION_TO_TYPE)
         .astype("Int64")
    )


def norm_user_id(x) -> str:
    """
    Normalize a user id string to match the dataset conventions:
    - strip leading 'u'
    - strip trailing '.0'
    - trim spaces
    """
    s = str(x).strip()
    if s.startswith("u"):
        s = s[1:]
    if s.endswith(".0"):
        s = s[:-2]
    return s


def infer_num_relations(edge_type_tensor: torch.Tensor) -> int:
    """
    Derive num_relations from an edge_type tensor (0..R-1).
    Returns 1 if tensor is empty.
    """
    if edge_type_tensor is None or edge_type_tensor.numel() == 0:
        return 1
    return int(edge_type_tensor.max().item()) + 1


# -----------------------------------------------------------------------------
# Core metrics/helpers you already had
# -----------------------------------------------------------------------------
def accuracy(output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    output: logits [N, C]
    labels: int64 [N]
    """
    preds = output.argmax(dim=-1)
    return (preds == labels).float().mean()


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# -----------------------------------------------------------------------------
# Threshold selection helpers (optional but handy)
# -----------------------------------------------------------------------------
def best_threshold_f1(y_true, y_score, grid=None) -> float:
    """
    Pick the threshold (0..1) that maximizes F1 on a validation set.
    """
    from sklearn.metrics import f1_score

    if grid is None:
        grid = np.linspace(0.05, 0.95, 181)  # step=0.005
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score).astype(float).ravel()

    f1s = [f1_score(y_true, (y_score >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def threshold_for_precision(y_true, y_score, prec_target=0.90) -> float:
    """
    Small utility to pick the smallest threshold that achieves a desired precision.
    Falls back to 0.5 if none meet the target.
    """
    from sklearn.metrics import precision_recall_curve

    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score).astype(float).ravel()

    p, r, th = precision_recall_curve(y_true, y_score)  # p len = len(th)+1
    candidates = [T for P, T in zip(p[1:], th) if P >= float(prec_target)]
    if not candidates:
        return 0.5
    return float(min(candidates))
