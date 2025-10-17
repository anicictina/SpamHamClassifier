# scripts/eval_deep_plots.py
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from scipy.special import softmax
from pathlib import Path
from src.datasets import load_spamassassin, load_enron, combine_datasets
from src.evaluate import save_confusion_matrix, save_roc_curve

OUTDIR = "outputs"
MODEL_DIR = f"{OUTDIR}/best_transformer"
MAX_LEN = 256


def load_dataset(name="spamassassin"):
    """Jednostavna zamena za resolve_dataset iz main.py, za evaluaciju."""
    if name == "spamassassin":
        return load_spamassassin("data/spamassassin")
    elif name == "enron":
        return load_enron("data/enron")
    elif name == "both":
        sa = load_spamassassin("data/spamassassin")
        en = load_enron("data/enron")
        return combine_datasets([sa, en])
    else:
        raise ValueError(f"Unknown dataset: {name}")


def main():
    # 🔹 Izmeni ime skupa po potrebi: "spamassassin", "enron" ili "both"
    df = load_dataset("spamassassin")

    y_str = df["label"].astype(str).str.lower()
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(df))
    _, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y_str)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    y_bin = (df_test["label"].str.lower() == "spam").astype(int).values
    texts = df_test["text"].astype(str).tolist()
    ds_test = Dataset.from_dict({"text": texts, "label": y_bin})

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    def tokenize(examples):
        return tok(examples["text"], truncation=True, max_length=MAX_LEN)

    ds_test_tok = ds_test.map(tokenize, batched=True, remove_columns=["text"])

    trainer = Trainer(model=model, tokenizer=tok)
    pred = trainer.predict(ds_test_tok)

    logits = pred.predictions
    y_pred = np.argmax(logits, axis=-1)
    y_score = softmax(logits, axis=-1)[:, 1]

    y_true_str = np.where(y_bin == 1, "spam", "ham")
    y_pred_str = np.where(y_pred == 1, "spam", "ham")

    Path(f"{OUTDIR}/confusion_matrices").mkdir(parents=True, exist_ok=True)
    Path(f"{OUTDIR}/roc_curves").mkdir(parents=True, exist_ok=True)

    save_confusion_matrix(y_true_str, y_pred_str, f"{OUTDIR}/confusion_matrices", "Transformer")
    save_roc_curve(y_true_str, y_score, f"{OUTDIR}/roc_curves", "Transformer")

    print("✅ Saved deep model plots to outputs/")


if __name__ == "__main__":
    main()
