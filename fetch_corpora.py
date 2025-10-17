#!/usr/bin/env python3
"""
fetch_corpora.py
----------------
Robusno preuzimanje i raspakivanje korpusa za projekat "Spam vs Ham".

Podržano:
- SpamAssassin Public Corpus: easy_ham, easy_ham_2, hard_ham, spam, spam_2
- Enron Email Dataset: enron_mail_20110402.tgz

Primeri:
  python fetch_corpora.py --spamassassin
  python fetch_corpora.py --enron
  python fetch_corpora.py --spamassassin --enron
  python fetch_corpora.py --data-dir data --spamassassin
"""
import argparse, sys, time, os, tarfile, shutil
from pathlib import Path
from urllib.error import URLError, HTTPError

SA_BASE = "https://spamassassin.apache.org/old/publiccorpus"
SA_ARCHIVES = {
    "easy_ham":   "20030228_easy_ham.tar.bz2",
    "easy_ham_2": "20030228_easy_ham_2.tar.bz2",
    "hard_ham":   "20030228_hard_ham.tar.bz2",
    "spam":       "20030228_spam.tar.bz2",
    "spam_2":     "20030228_spam_2.tar.bz2",
}
ENRON_URL = "https://www.cs.cmu.edu/~enron/enron_mail_20110402.tgz"

def progress_bar(downloaded:int, total:int):
    total = total or 1
    pct = int(downloaded * 100 / total)
    pct = max(0, min(100, pct))
    w = 30
    bar = "#" * int(w * pct / 100) + "-" * (w - int(w * pct / 100))
    print(f"\r[{bar}] {pct:3d}%", end="", flush=True)

def magic_ok(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            head = fh.read(8)
        return (
            head.startswith(b"BZh") or        # bz2
            head.startswith(b"\x1f\x8b") or   # gzip
            head.startswith(b"\xfd7zXZ") or   # xz
            head.startswith(b"ustar")         # tar
        )
    except Exception:
        return False

def download_with_retries(url: str, dest: Path, retries=4, backoff=2.0, min_size=100_000):
    import urllib.request
    last_err = None
    for attempt in range(1, retries+1):
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            print(f"Preuzimam (pokusaj {attempt}/{retries}): {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=90) as r, open(dest, "wb") as f:
                total = int(r.headers.get("Content-Length","0"))
                downloaded = 0
                chunk = 64 * 1024
                while True:
                    b = r.read(chunk)
                    if not b: break
                    f.write(b)
                    downloaded += len(b)
                    progress_bar(downloaded, total)
            print("")  # newline posle progressa
            if dest.exists() and dest.stat().st_size >= min_size and magic_ok(dest):
                print("OK:", dest.name); return dest
            else:
                print("Upozorenje: fajl izgleda korumpirano ili prekratko, pokušaću ponovo…")
        except (HTTPError, URLError, Exception) as e:
            last_err = e
            print(f"Greška pri preuzimanju: {e}. Novi pokušaj za {backoff}s…")
        time.sleep(backoff)
    raise SystemExit(f"Neuspešno preuzimanje posle {retries} pokušaja: {url} (poslednja greška: {last_err})")

def tar_mode_for(path: Path) -> str:
    name = path.name.lower()
    if name.endswith((".tar.bz2",".tbz2",".tbz")): return "r:bz2"
    if name.endswith((".tar.gz",".tgz")):          return "r:gz"
    if name.endswith((".tar.xz",".txz")):          return "r:xz"
    return "r:*"

def extract_tar(archive_path: Path, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Raspakujem: {archive_path.name} -> {dest_dir}")
    if not magic_ok(archive_path):
        raise SystemExit(f"Arhiva izgleda korumpirano ili nije tar.*: {archive_path}")
    with tarfile.open(archive_path, tar_mode_for(archive_path)) as tf:
        # filter="data" je kompatibilno sa Python 3.14 default bezbednosnim filterom
        tf.extractall(dest_dir, filter="data")
    print("Raspakovano.")

def normalize_spamassassin(extracted_root: Path, target_root: Path):
    target_ham  = target_root / "ham"
    target_spam = target_root / "spam"
    target_ham.mkdir(parents=True, exist_ok=True)
    target_spam.mkdir(parents=True, exist_ok=True)
    moved_ham = moved_spam = 0

    for p in extracted_root.rglob("*"):
        if p.is_file():
            low = p.name.lower()
            # preskoči eventualne arhive unutar ekstrakcije
            if low.endswith((".bz2",".gz",".tar",".tgz",".zip")):
                continue
            parts = [x.name.lower() for x in p.parents]
            dst_folder = None
            if any("spam" in x for x in parts):
                dst_folder = target_spam; moved_spam += 1
            elif any(("ham" in x) or ("easy_ham" in x) or ("hard_ham" in x) for x in parts):
                dst_folder = target_ham; moved_ham += 1
            if dst_folder is not None:
                dst = dst_folder / p.name
                i = 1
                # unikatan naziv u slučaju kolizije
                while dst.exists():
                    stem = p.stem; ext = "".join(p.suffixes)
                    dst = dst_folder / f"{stem}_{i}{ext}"; i += 1
                shutil.move(str(p), str(dst))

    # očisti privremeni ekstrakt
    if extracted_root.exists():
        shutil.rmtree(extracted_root, ignore_errors=True)

    return moved_ham, moved_spam

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data", help="Root folder za podatke (default: data)")
    ap.add_argument("--spamassassin", action="store_true", help="Preuzmi i pripremi SpamAssassin")
    ap.add_argument("--enron", action="store_true", help="Preuzmi i raspakuj Enron (veliko)")
    ap.add_argument("--skip-normalize", action="store_true", help="Ne premeštaj u ham/spam (ostavi raw)")
    args = ap.parse_args()

    if not (args.spamassassin or args.enron):
        ap.error("Dodaj --spamassassin i/ili --enron.")

    data_root = Path(args.data_dir)

    if args.spamassassin:
        sa_root = data_root / "spamassassin"
        sa_tmp  = sa_root / "_tmp_extract"
        sa_tmp.mkdir(parents=True, exist_ok=True)
        # 1) preuzmi sve arhive
        for key, fname in SA_ARCHIVES.items():
            url = f"{SA_BASE}/{fname}"
            archive_path = sa_root / fname
            download_with_retries(url, archive_path)
            extract_tar(archive_path, sa_tmp / key)
        # 2) normalizuj u ham/spam
        if not args.skip_normalize:
            moved_ham, moved_spam = normalize_spamassassin(sa_tmp, sa_root)
            ham_cnt  = len(list((sa_root/"ham").glob("*")))
            spam_cnt = len(list((sa_root/"spam").glob("*")))
            print(f"\nSpamAssassin spreman u: {sa_root}")
            print(f"  Ham fajlova:  {ham_cnt} (+{moved_ham} premešteno)")
            print(f"  Spam fajlova: {spam_cnt} (+{moved_spam} premešteno)")

    if args.enron:
        enron_root = data_root / "enron"
        enron_root.mkdir(parents=True, exist_ok=True)
        archive_path = enron_root / "enron_mail_20110402.tgz"
        # Enron je veliki -> očekuj bar 10 MB
        download_with_retries(ENRON_URL, archive_path, min_size=10_000_000)
        extract_tar(archive_path, enron_root)
        print("Enron spreman u:", enron_root)

    print("\nGotovo.")

if __name__ == "__main__":
    main()
