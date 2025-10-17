import os
import csv
from typing import List, Tuple
import pandas as pd
from pathlib import Path


def load_sample_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert 'text' in df.columns and 'label' in df.columns
    return df

def _gather_text_files(root: str, label_hint: str = None) -> List[Tuple[str, str]]:
    pairs = []
    if not os.path.isdir(root):
        return pairs
    for dirpath, _, filenames in os.walk(root):
        parts = [p.lower() for p in Path(dirpath).parts]  # <<< ključno: delovi putanje
        for fn in filenames:
            low = fn.lower()
            if low.endswith(('.tar', '.tar.bz2', '.tar.gz', '.tgz', '.bz2', '.gz', '.zip')):
                continue

            fpath = os.path.join(dirpath, fn)
            label = label_hint

            if label is None:
                # prvo proveri ham varijante
                if any(x in ('ham', 'easy_ham', 'hard_ham', 'legit') for x in parts):
                    label = 'ham'
                # zatim spam varijante ('spamassassin' NE SME da se računa kao 'spam')
                elif any(x in ('spam', 'spam_2') for x in parts):
                    label = 'spam'

            if label is None:
                continue  # ne znam labelu => preskoči

            pairs.append((fpath, label))
    return pairs

def _read_text(path: str) -> str:
    for enc in ('utf-8', 'latin-1'):
        try:
            with open(path, 'r', encoding=enc, errors='ignore') as f:
                return f.read()
        except Exception:
            continue
    return ""


def load_spamassassin(root: str) -> pd.DataFrame:
    pairs = _gather_text_files(root)
    rows = []
    for p, lab in pairs:
        rows.append({'text': _read_text(p), 'label': lab})
    return pd.DataFrame(rows)

def load_enron(root: str) -> pd.DataFrame:
    pairs = _gather_text_files(root)
    rows = []
    for p, lab in pairs:
        rows.append({'text': _read_text(p), 'label': lab})
    return pd.DataFrame(rows)

def combine_datasets(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    dfs = [df[['text', 'label']].dropna().copy() for df in dfs if df is not None and len(df) > 0]
    if not dfs:
        return pd.DataFrame(columns=['text','label'])
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df = df.drop_duplicates(subset=['text'])
    df['label'] = df['label'].str.lower().map({'spam':'spam','ham':'ham'})
    df = df[df['label'].isin(['spam','ham'])]
    return df.reset_index(drop=True)
