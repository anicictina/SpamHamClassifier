# scripts/plot_summary.py
import json
import os
import matplotlib.pyplot as plt
import numpy as np

OUTDIR = "outputs"

def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _extract_transformer_metrics(mtr):
    """
    Podrži 2 oblika:
      A) dict sa ključevima eval_accuracy/...
      B) list [ {eval_*...}, "outputs/best_transformer" ]
    Vrati dict sa accuracy/precision/recall/f1/roc_auc.
    """
    if isinstance(mtr, list):
        # očekujemo [metrics_dict, something]
        if not mtr:
            return {}
        m = mtr[0] if isinstance(mtr[0], dict) else {}
    elif isinstance(mtr, dict):
        m = mtr
    else:
        m = {}

    return {
        "accuracy": float(m.get("eval_accuracy", 0) or 0),
        "precision": float(m.get("eval_precision", 0) or 0),
        "recall": float(m.get("eval_recall", 0) or 0),
        "f1": float(m.get("eval_f1", 0) or 0),
        "roc_auc": float(m.get("eval_roc_auc", 0) or 0),
    }

def load_metrics():
    m_classical = _load_json(os.path.join(OUTDIR, "metrics_classical.json"))
    m_transformer_raw = _load_json(os.path.join(OUTDIR, "metrics_transformer.json"))
    m_transformer = _extract_transformer_metrics(m_transformer_raw)

    models = []
    rows = []

    # classical
    for name, d in m_classical["results"].items():
        models.append(name)
        rows.append([
            float(d.get("accuracy", 0) or 0),
            float(d.get("precision", 0) or 0),
            float(d.get("recall", 0) or 0),
            float(d.get("f1", 0) or 0),
            float(d.get("roc_auc", 0) or 0),
        ])

    # transformer
    models.append("Transformer")
    rows.append([
        m_transformer["accuracy"],
        m_transformer["precision"],
        m_transformer["recall"],
        m_transformer["f1"],
        m_transformer["roc_auc"],
    ])

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    return models, metrics, np.array(rows, dtype=float)

def plot_bars(models, metrics, values):
    n_models, n_metrics = values.shape
    x = np.arange(n_metrics)
    width = 0.8 / n_models

    plt.figure(figsize=(10, 5))
    for i, model in enumerate(models):
        plt.bar(x + i * width, values[i], width, label=model)

    plt.xticks(x + width * (n_models - 1) / 2, metrics)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Model comparison (higher is better)")
    plt.legend()
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, "models_summary_bars.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path

def save_table(models, metrics, values):
    lines = []
    header = ["model"] + metrics
    lines.append("\t".join(header))
    for i, m in enumerate(models):
        row = [m] + [f"{v:.4f}" for v in values[i]]
        lines.append("\t".join(row))
    path = os.path.join(OUTDIR, "models_summary_table.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path

if __name__ == "__main__":
    models, metrics, values = load_metrics()
    bar_path = plot_bars(models, metrics, values)
    table_path = save_table(models, metrics, values)
    print(f"Saved: {bar_path}")
    print(f"Saved: {table_path}")
