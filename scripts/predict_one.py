# scripts/predict_one.py
import argparse
import json
import os
import sys
from typing import Optional

def read_text(args) -> str:
    if args.text is not None:
        return args.text
    if args.file is not None:
        with open(args.file, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    # stdin
    return sys.stdin.read()

def predict_classical(text: str, model_path: str):
    import joblib
    import numpy as np
    from src.preprocess import clean_text

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Classical model not found at {model_path}")

    model = joblib.load(model_path)  # Pipeline(vec, clf)
    x = [clean_text(text)]

    # label
    y_pred = model.predict(x)[0]

    # probability/score (best effort)
    p_spam = None
    clf = model.named_steps.get("clf", None)
    try:
        if hasattr(model, "predict_proba"):
            p_spam = float(model.predict_proba(x)[:, 1][0])
        elif clf is not None and hasattr(clf, "predict_proba"):
            p_spam = float(clf.predict_proba(model.named_steps["vec"].transform(x))[:, 1][0])
        elif hasattr(model, "decision_function"):
            score = float(model.decision_function(x)[0])
            # pseudo-prob via sigmoid for models bez prob (npr. LinearSVC)
            p_spam = 1.0 / (1.0 + np.exp(-score))
        elif clf is not None and hasattr(clf, "decision_function"):
            score = float(clf.decision_function(model.named_steps["vec"].transform(x))[0])
            p_spam = 1.0 / (1.0 + np.exp(-score))
    except Exception:
        p_spam = None

    if p_spam is None:
        # fallback: 1.0 if predicted spam; 0.0 otherwise
        p_spam = 1.0 if str(y_pred).lower() == "spam" else 0.0

    p_ham = 1.0 - p_spam
    return {
        "model_type": "classical",
        "model_path": model_path,
        "pred_label": str(y_pred),
        "prob_spam": p_spam,
        "prob_ham": p_ham,
    }

def predict_transformer(text: str, model_dir: str, max_len: int = 256):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import numpy as np

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Transformer directory not found at {model_dir}")

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    device = torch.device("cpu")
    model.to(device)

    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len)
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits.cpu().numpy()[0]

    # locate spam index using config if possible
    spam_idx: Optional[int] = None
    label2id = getattr(model.config, "label2id", None)
    if isinstance(label2id, dict) and label2id:
        # try case-insensitive map
        for k, v in label2id.items():
            if str(k).lower() == "spam":
                spam_idx = int(v)
                break
    if spam_idx is None:
        # default assumption: class 1 is spam
        spam_idx = 1

    # softmax
    exps = np.exp(logits - logits.max())
    probs = exps / exps.sum()
    p_spam = float(probs[spam_idx])
    p_ham = float(1.0 - p_spam)
    pred_idx = int(probs.argmax())

    # try to recover label name
    id2label = getattr(model.config, "id2label", None)
    if isinstance(id2label, dict) and id2label:
        pred_label = str(id2label.get(pred_idx, pred_idx))
    else:
        pred_label = "spam" if pred_idx == spam_idx else "ham"

    return {
        "model_type": "transformer",
        "model_path": model_dir,
        "pred_label": pred_label,
        "prob_spam": p_spam,
        "prob_ham": p_ham,
    }

def main():
    p = argparse.ArgumentParser(
        description="Predict spam/ham for a single text using classical or transformer model."
    )
    p.add_argument("--type", choices=["auto", "classical", "transformer"], default="auto",
                   help="Which model to use. 'auto' prefers transformer if present.")
    p.add_argument("--model-path", default="outputs/best_model.pkl",
                   help="Path to classical joblib model (Pipeline).")
    p.add_argument("--transformer-dir", default="outputs/best_transformer",
                   help="Directory with saved HF model (config.json, tokenizer.json, etc.).")
    p.add_argument("--text", default=None, help="Input text to classify.")
    p.add_argument("--file", default=None, help="Path to a text file to classify.")
    p.add_argument("--max-len", type=int, default=256, help="Max length for transformer tokenization.")
    p.add_argument("--json", action="store_true", help="Print raw JSON only.")
    args = p.parse_args()

    txt = read_text(args).strip()
    if not txt:
        print("No input text provided. Use --text, --file, or pipe text via stdin.", file=sys.stderr)
        sys.exit(1)

    use_type = args.type
    if use_type == "auto":
        if os.path.isdir(args.transformer_dir) and os.path.exists(os.path.join(args.transformer_dir, "config.json")):
            use_type = "transformer"
        elif os.path.exists(args.model_path):
            use_type = "classical"
        else:
            print("No model found. Provide --model-path or --transformer-dir.", file=sys.stderr)
            sys.exit(2)

    if use_type == "classical":
        result = predict_classical(txt, args.model_path)
    else:
        result = predict_transformer(txt, args.transformer_dir, max_len=args.max_len)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"[{result['model_type']}] {result['model_path']}")
        print(f"Predicted: {result['pred_label']}")
        print(f"P(spam) = {result['prob_spam']:.4f} | P(ham) = {result['prob_ham']:.4f}")

if __name__ == "__main__":
    main()
