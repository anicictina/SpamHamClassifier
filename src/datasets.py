import os
from typing import List, Tuple, Optional
from pathlib import Path
import pandas as pd


def load_sample_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert 'text' in df.columns and 'label' in df.columns
    return df


def _gather_text_files(root: str, label_hint: Optional[str] = None) -> List[Tuple[str, str]]:

    pairs: List[Tuple[str, str]] = []
    if not os.path.isdir(root):
        return pairs

    for dirpath, _, filenames in os.walk(root):
        parts = [p.lower() for p in Path(dirpath).parts]
        for fn in filenames:
            low = fn.lower()
            # preskoči arhive i kompresovane fajlove
            if low.endswith(('.tar', '.tar.bz2', '.tar.gz', '.tgz', '.bz2', '.gz', '.zip')):
                continue

            fpath = os.path.join(dirpath, fn)
            label = label_hint  # ako je prosleđen hint, koristim ga

            if label is None:
                # prvo proveri 'ham' varijante
                if any(x in ('ham', 'ham_subset', 'easy_ham', 'hard_ham', 'legit') for x in parts):
                    label = 'ham'
                # zatim 'spam' varijante (ne uključuj 'spamassassin'!)
                elif any(x in ('spam', 'spam_2') for x in parts):
                    label = 'spam'

            if label is None:
                # ne znamo labelu → preskoči
                continue

            pairs.append((fpath, label))

    return pairs


def _read_text(path: str) -> str:
    # probaj nekoliko enkodiranja; ignorisi greške
    for enc in ('utf-8', 'latin-1'):
        try:
            with open(path, 'r', encoding=enc, errors='ignore') as f:
                return f.read()
        except Exception:
            continue
    return ""


def _pairs_to_df(pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    rows = [{'text': _read_text(p), 'label': lab} for p, lab in pairs]
    df = pd.DataFrame(rows)
    # odbaci prazne tekstove i normalizuj labele
    if not df.empty:
        df['text'] = df['text'].fillna('').astype(str)
        df = df[df['text'].str.len() > 0]
        df['label'] = df['label'].astype(str).str.lower().map({'spam': 'spam', 'ham': 'ham'})
        df = df[df['label'].isin(['spam', 'ham'])]
        df = df.drop_duplicates(subset=['text']).reset_index(drop=True)
    return df


def load_spamassassin(root: str) -> pd.DataFrame:

    pairs = _gather_text_files(root)
    return _pairs_to_df(pairs)


def load_enron(root: str) -> pd.DataFrame:

    hs = Path(root) / "ham_subset"
    if hs.exists():
        pairs = _gather_text_files(str(hs), label_hint='ham')
        return _pairs_to_df(pairs)

    # prvo probaj automatsku detekciju (ham/spam ako ih ima)
    pairs = _gather_text_files(root)
    if pairs:
        return _pairs_to_df(pairs)

    # fallback: sve kao ham
    pairs = _gather_text_files(root, label_hint='ham')
    return _pairs_to_df(pairs)


def combine_datasets(dfs: List[pd.DataFrame]) -> pd.DataFrame:

    kept = []
    for df in dfs:
        if df is None or df.empty:
            continue
        df2 = df[['text', 'label']].dropna()
        if not df2.empty:
            kept.append(df2)
    if not kept:
        return pd.DataFrame(columns=['text', 'label'])

    out = pd.concat(kept, axis=0, ignore_index=True)
    out['text'] = out['text'].fillna('').astype(str)
    out = out[out['text'].str.len() > 0]
    out['label'] = out['label'].astype(str).str.lower().map({'spam': 'spam', 'ham': 'ham'})
    out = out[out['label'].isin(['spam', 'ham'])]
    out = out.drop_duplicates(subset=['text']).reset_index(drop=True)
    return out


def load_dataset(name: str) -> pd.DataFrame:

    n = (name or '').lower()
    if n == 'spamassassin':
        return load_spamassassin('data/spamassassin')
    if n == 'enron':
        return load_enron('data/enron')
    if n in ('both', 'combined'):
        sa = load_spamassassin('data/spamassassin')
        en = load_enron('data/enron')
        return combine_datasets([sa, en])
    raise ValueError(f"Unknown dataset name: {name!r} (use 'spamassassin' | 'enron' | 'both').")
