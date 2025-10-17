"""
Preuzimanje i priprema Enron Email Dataset-a.
- Ako proslediš --url (ka .zip/.tar.gz), biće preuzet i raspakovan u data/enron/raw
- Ako ne, skripta samo ispiše instrukcije i očekuješ da ručno staviš fajlove u data/enron/raw
Sav sadržaj tretiramo kao 'ham' (realni korpus poslovnih mejlova).
"""

import argparse, tarfile, zipfile, shutil
from urllib.request import urlretrieve
from pathlib import Path

def _extract(archive: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(dest)
    elif archive.suffixes[-2:] == [".tar", ".gz"] or archive.suffixes[-1:] == [".tgz"] \
         or archive.suffixes[-2:] == [".tar", ".bz2"]:
        with tarfile.open(archive, "r:*") as tf:
            tf.extractall(dest)
    else:
        raise ValueError(f"Unknown archive type: {archive.name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", type=str, default=None, help="URL Enron arhive (.zip/.tar.gz). Opcionalno.")
    args = ap.parse_args()

    base = Path("data/enron")
    raw = base / "raw"
    ham_dir = base / "ham"
    raw.mkdir(parents=True, exist_ok=True)

    if args.url:
        print(f"Downloading: {args.url}")
        out = raw / ("enron" + (".zip" if args.url.endswith(".zip") else ".tar.gz"))
        urlretrieve(args.url, out)
        print("Extracting...")
        _extract(out, raw)
        out.unlink(missing_ok=True)
    else:
        print("No URL provided. Place Enron raw folders under data/enron/raw manually.")
        print("Example: data/enron/raw/maildir/...")

    # Svu strukturu iz raw tretiramo kao ham → kopiramo tekst fajlove
    if ham_dir.exists():
        shutil.rmtree(ham_dir)
    ham_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for p in raw.rglob("*"):
        if p.is_file():
            # kopiramo sve kao ham (Enron je legit korpus)
            rel = p.relative_to(raw)
            dst = ham_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(p, dst)
                count += 1
            except Exception:
                pass

    print(f"Enron ready at: {base}")
    print(f"  ham files: {count}")

if __name__ == "__main__":
    main()
