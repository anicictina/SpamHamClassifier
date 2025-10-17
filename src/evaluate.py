import os
import json
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay,
)
from .utils import ensure_dir


def compute_metrics(y_true, y_pred, y_proba=None) -> Dict:
    """
    Računa osnovne metrike performansi za binarnu klasifikaciju (spam vs ham).
    Pozitivna klasa je 'spam'.
    """
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label="spam", zero_division=0
    )

    out = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }

    # ROC-AUC ako postoje skorovi/probabilnosti
    if y_proba is not None:
        try:
            y_bin = (np.array(y_true) == "spam").astype(int)
            out["roc_auc"] = roc_auc_score(y_bin, y_proba)
        except Exception:
            out["roc_auc"] = None
    else:
        out["roc_auc"] = None

    return out


def save_confusion_matrix(y_true, y_pred, outdir: str, name: str):
    """
    Čuva konfuzionu matricu kao PNG sliku.
    """
    ensure_dir(outdir)
    cm = confusion_matrix(y_true, y_pred, labels=["ham", "spam"])

    fig = plt.figure(figsize=(5, 4), dpi=120)
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["ham", "spam"])
    plt.yticks(tick_marks, ["ham", "spam"])

    # upis brojeva unutar polja
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    path = os.path.join(outdir, f"{name}_confusion.png")
    plt.savefig(path)
    plt.close(fig)
    return path


def save_roc_curve(y_true, y_proba, outdir: str, name: str):
    """
    Crta i čuva ROC krivu za model ako postoje predikcije verovatnoća ili skorovi.
    """
    ensure_dir(outdir)
    try:
        y_bin = (np.array(y_true) == "spam").astype(int)
        fig = plt.figure(figsize=(5, 4), dpi=120)
        RocCurveDisplay.from_predictions(y_bin, y_proba)
        plt.title(f"ROC Curve - {name}")
        plt.tight_layout()

        path = os.path.join(outdir, f"{name}_roc.png")
        plt.savefig(path)
        plt.close(fig)
        return path
    except Exception:
        return None
