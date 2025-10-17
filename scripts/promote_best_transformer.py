from transformers import AutoModelForSequenceClassification, AutoTokenizer
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, help="Path to checkpoint-X dir")
ap.add_argument("--out",  default="outputs/best_transformer", help="Where to save merged model")
args = ap.parse_args()

model = AutoModelForSequenceClassification.from_pretrained(args.ckpt)
model.save_pretrained(args.out)

tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tok.save_pretrained(args.out)

print("Saved merged model to", args.out)
