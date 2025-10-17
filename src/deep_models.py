from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from scipy.special import softmax
from datasets import Dataset
from pathlib import Path
import json


def train_transformer(
    texts,
    labels,
    model_name: str = "distilbert-base-uncased",
    outdir: str = "outputs",
    epochs: int = 2,
    batch_size: int = 16,
    seed: int = 42,
) -> Tuple[Dict, str]:
    """
    Trenira transformer (DistilBERT ili sličan) za spam/ham.
    Kompatibilno sa main.py: prima texts i labels kao pandas Series ili listu.
    Pozitivna klasa = 1 (spam).
    """
    # 1) DataFrame + stratifikovan split po string labelama
    df = pd.DataFrame({"text": texts, "label": labels})
    y_str = df["label"].astype(str).str.lower()

    from sklearn.model_selection import train_test_split

    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=seed, stratify=y_str
    )
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # 2) Pretvori labelu u 0/1 (ham=0, spam=1) i napravi HF Dataset-e
    def _to_hf_dataset(df_):
        y = (df_["label"].astype(str).str.lower() == "spam").astype(int)
        X = df_["text"].astype(str).tolist()
        return [{"text": t, "label": int(l)} for t, l in zip(X, y)]

    ds_train = Dataset.from_list(_to_hf_dataset(df_train))
    ds_test = Dataset.from_list(_to_hf_dataset(df_test))

    # 3) Tokenizer i model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    label2id = {"ham": 0, "spam": 1}
    id2label = {0: "ham", 1: "spam"}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, id2label=id2label, label2id=label2id
    )

    # 4) Tokenizuj preko Dataset.map (bez custom collate_fn)
    def tokenize_batch(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,
        )

    # remove_columns uklanja "text" posle tokenizacije; "label" ostaje
    ds_train = ds_train.map(tokenize_batch, batched=True, remove_columns=["text"])
    ds_test = ds_test.map(tokenize_batch, batched=True, remove_columns=["text"])

    # 5) Data collator koji samo radi pad prema tokenizeru
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 6) Metrics (pozitivna klasa = 1 → spam)
    def compute_metrics(eval_pred):
        logits, labels_np = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels_np, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels_np, preds, average="binary", pos_label=1, zero_division=0
        )
        try:
            y_score = softmax(logits, axis=-1)[:, 1]
            auc = roc_auc_score(labels_np, y_score)
        except Exception:
            auc = None
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}

    # 7) TrainingArguments — kompatibilno i sa starijim i sa novijim transformers
    def _build_args():
        base_kwargs = dict(
            output_dir=f"{outdir}/best_transformer",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=5e-5,
            weight_decay=0.01,
            seed=seed,
            logging_steps=50,
        )
        try:
            return TrainingArguments(
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                greater_is_better=True,
                **base_kwargs,
            )
        except TypeError:
            # starije verzije (bez evaluation/save strategija)
            return TrainingArguments(**base_kwargs)

    args = _build_args()

    # 8) Trainer (bez custom collate_fn — koristimo DataCollatorWithPadding)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_test,
        tokenizer=tokenizer,  # ok i ako izbaci FutureWarning
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 9) Treniraj i evaluiraj
    trainer.train()
    metrics = trainer.evaluate()

    # 10) Sačuvaj metrike
    Path(outdir).mkdir(parents=True, exist_ok=True)
    with open(f"{outdir}/metrics_transformer.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics, f"{outdir}/best_transformer"
