# scripts/download_enron_csv.py
import os
import pandas as pd
from pathlib import Path

RAW_CSV = "data/enron/raw/emails.csv"
OUT_DIR = Path("data/enron/ham")

def main():
    if not os.path.exists(RAW_CSV):
        print(" Nije pronađen data/enron/raw/emails.csv")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW_CSV)

    # koristi kolonu "message" (ili "content") ako postoji
    text_col = "message" if "message" in df.columns else "content" if "content" in df.columns else None
    if text_col is None:
        print(" CSV nema kolonu 'message' ili 'content'. Pogledaj imena kolona u fajlu.")
        print("Kolone:", list(df.columns))
        return

    print(f" Učitano {len(df)} emailova. Kreiram .txt fajlove u {OUT_DIR}/")

    for i, msg in enumerate(df[text_col].astype(str).fillna("")):
        with open(OUT_DIR / f"email_{i:06d}.txt", "w", encoding="utf-8") as f:
            f.write(msg)

    print(f" Zapisano {len(df)} fajlova u {OUT_DIR}/")

if __name__ == "__main__":
    main()
