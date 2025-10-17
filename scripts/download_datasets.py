import os
import sys
import tarfile
import time
from urllib.request import urlretrieve
from pathlib import Path


SPAMASSASSIN_URLS = {
    "easy_ham": "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2",
    "easy_ham_2": "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2",
    "hard_ham": "https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2",
    "spam": "https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2",
    "spam_2": "https://spamassassin.apache.org/old/publiccorpus/20030228_spam_2.tar.bz2",
}


def extract_tar(archive_path: Path, dest_dir: Path):
    """
    Bezbedno raspakivanje .tar.bz2 arhive.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(dest_dir)
    except tarfile.ReadError as e:
        print(f"Greška pri otvaranju {archive_path.name}: {e}")


def download_file(url: str, out_path: Path, retries: int = 4) -> bool:

    for attempt in range(1, retries + 1):
        try:
            print(f"Preuzimam (pokusaj {attempt}/{retries}): {url}")
            urlretrieve(url, out_path)
            size = out_path.stat().st_size
            if size < 1024:  # ako je previše mali
                raise IOError("file too small")
            print(f"OK: {out_path.name}")
            return True
        except Exception as e:
            print(f"Upozorenje: fajl izgleda korumpirano ili prekratko, pokusavam ponovo... ({e})")
            time.sleep(1)
    print(f"Neuspesno preuzimanje posle {retries} pokusaja: {url}")
    return False


def download_spamassassin():

    base_dir = Path("data/spamassassin")
    tmp_extract = base_dir / "_tmp_extract"
    tmp_extract.mkdir(parents=True, exist_ok=True)

    total_spam, total_ham = 0, 0

    for name, url in SPAMASSASSIN_URLS.items():
        dest_archive = tmp_extract / f"{name}.tar.bz2"
        if not download_file(url, dest_archive):
            continue

        print(f"Raspakujem: {dest_archive.name} -> {tmp_extract / name}")
        extract_tar(dest_archive, tmp_extract / name)
        dest_archive.unlink(missing_ok=True)

    # Provera broja fajlova po tipu
    for sub in tmp_extract.iterdir():
        files = list(sub.rglob("*"))
        if "spam" in sub.name:
            total_spam += len(files)
        else:
            total_ham += len(files)

    print(f"\nSpamAssassin spreman u: {base_dir}")
    print(f"  Ham fajlova:  {total_ham}")
    print(f"  Spam fajlova: {total_spam}")
    print("Gotovo.")


if __name__ == "__main__":
    download_spamassassin()
