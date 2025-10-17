import os
import json
import argparse
import pandas as pd

from src.utils import set_seed, ensure_dir
from src.datasets import (
    load_sample_csv,
    load_spamassassin,
    load_enron,
    combine_datasets,
)
from src.classical_models import train_classical
from src.deep_models import train_transformer
from src.outlier import anomaly_report


def resolve_dataset(name: str, data_dir: str) -> pd.DataFrame:

    name = (name or "").lower()

    if name == "sample":
        path = os.path.join(data_dir, "sample_dataset.csv")
        if os.path.isfile(path):
            return load_sample_csv(path)
        else:
            print("[warn] sample_dataset.csv not found; falling back to SpamAssassin")
            name = "spamassassin"

    if name == "spamassassin":
        return load_spamassassin(os.path.join(data_dir, "spamassassin"))

    if name == "enron":
        return load_enron(os.path.join(data_dir, "enron"))

    if name in ("combined", "both"):
        dfs = []
        sa = load_spamassassin(os.path.join(data_dir, "spamassassin"))
        if len(sa) > 0:
            dfs.append(sa)
        en = load_enron(os.path.join(data_dir, "enron"))
        if len(en) > 0:
            dfs.append(en)
        sample_path = os.path.join(data_dir, "sample_dataset.csv")
        if os.path.isfile(sample_path):
            dfs.append(load_sample_csv(sample_path))
        return combine_datasets(dfs)

    raise ValueError(f"Unknown dataset: {name!r}")


def cmd_train_classical(args):
    set_seed(args.seed)
    df = resolve_dataset(args.dataset, args.datadir)
    outdir = args.outdir
    ensure_dir(outdir)

    report, best_model = train_classical(
        df["text"],
        df["label"],
        features=args.features,
        test_size=args.test_size,
        random_state=args.seed,
        outdir=outdir,
    )

    # Save metrics
    with open(os.path.join(outdir, "metrics_classical.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Persist best model
    try:
        import joblib
        joblib.dump(best_model, os.path.join(outdir, "best_model.pkl"))
    except Exception as e:
        print(f"[warn] could not persist best classical model: {e}")

    print("Classical training finished. Best model:", report["best_model_name"])


def cmd_train_deep(args):
    set_seed(args.seed)
    df = resolve_dataset(args.dataset, args.datadir)
    outdir = args.outdir
    ensure_dir(outdir)

    metrics = train_transformer(
        df["text"],
        df["label"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        outdir=outdir,
        model_name=args.model_name,  # mapirano iz --model
    )

    with open(os.path.join(outdir, "metrics_transformer.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("Transformer training finished. Metrics saved.")


def cmd_outlier(args):
    set_seed(args.seed)
    df = resolve_dataset(args.dataset, args.datadir)
    ensure_dir(args.outdir)

    scores = anomaly_report(
        df["text"],
        df["label"],
        features=args.features,
        random_state=args.seed,
        contamination=args.contamination,
    )

    out_path = os.path.join(args.outdir, "outlier_scores.csv")
    out_df = pd.DataFrame({"text": df["text"], "label": df["label"], "anomaly_score": scores})
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Outlier report saved to {out_path}")


def build_argparser():
    p = argparse.ArgumentParser(description="Spam/Ham classifier - classical + transformer")
    sub = p.add_subparsers(dest="command", required=True)

    # ---- classical ----
    pc = sub.add_parser("train_classical", help="Train classical ML models")
    pc.add_argument("--features", choices=["bow", "tfidf", "w2v"], default="tfidf")
    pc.add_argument("--dataset", choices=["sample", "spamassassin", "enron", "combined", "both"], default="spamassassin")
    pc.add_argument("--datadir", default="data")
    pc.add_argument("--test-size", type=float, default=0.2)
    pc.add_argument("--outdir", default="outputs")
    pc.add_argument("--seed", type=int, default=42)
    pc.set_defaults(func=cmd_train_classical)

    # ---- deep ----
    pdp = sub.add_parser("train_deep", help="Fine-tune transformer (HuggingFace)")
    pdp.add_argument("--dataset", choices=["sample", "spamassassin", "enron", "combined", "both"], default="spamassassin")
    pdp.add_argument("--datadir", default="data")
    pdp.add_argument("--epochs", type=int, default=2)
    pdp.add_argument("--batch-size", type=int, default=16)
    # user-friendly flag; i dalje se prosleđuje kao model_name
    pdp.add_argument("--model", dest="model_name", default="distilbert-base-uncased")
    pdp.add_argument("--outdir", default="outputs")
    pdp.add_argument("--seed", type=int, default=42)
    pdp.set_defaults(func=cmd_train_deep)

    # ---- outlier ----
    po = sub.add_parser("outlier_check", help="Outlier / novelty detection report")
    po.add_argument("--features", choices=["bow", "tfidf", "w2v"], default="tfidf")
    po.add_argument("--dataset", choices=["sample", "spamassassin", "enron", "combined", "both"], default="spamassassin")
    po.add_argument("--datadir", default="data")
    po.add_argument("--contamination", type=float, default=0.05)
    po.add_argument("--outdir", default="outputs")
    po.add_argument("--seed", type=int, default=42)
    po.set_defaults(func=cmd_outlier)

    return p


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    args.func(args)
