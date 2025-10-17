# scripts/quick_data_check_fast.py
import os
from pathlib import Path

def count_files(root):
    total = 0
    for _, _, files in os.walk(root):
        total += len(files)
    return total

def main():
    base = Path("data")
    print(f"🔎 Looking in: {base.resolve()}")
    sa_spam = count_files(base / "spamassassin" / "spam")
    sa_ham  = count_files(base / "spamassassin" / "ham")
    en_ham  = count_files(base / "enron" / "ham")

    print("\n✅ Fast counts (no file reading):")
    print(f"  SpamAssassin spam : {sa_spam}")
    print(f"  SpamAssassin ham  : {sa_ham}")
    print(f"  Enron ham         : {en_ham}")
    print(f"\n  Combined total    : {sa_spam + sa_ham + en_ham}")

if __name__ == "__main__":
    main()
