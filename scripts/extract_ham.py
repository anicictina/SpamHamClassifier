import tarfile, shutil
from pathlib import Path

SA   = Path("data/spamassassin")
HAM  = SA / "ham"
TMP  = SA / "_ham_tmp"

HAM.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)

archives = [
    "20030228_easy_ham.tar.bz2",
    "20030228_easy_ham_2.tar.bz2",
    "20030228_hard_ham.tar.bz2",
]

moved = 0
for arc in archives:
    arc_path = SA / arc
    if not arc_path.exists():
        print(f"[skip] {arc} not found")
        continue
    out = TMP / arc.replace(".tar.bz2", "")
    out.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {arc} -> {out}")
    with tarfile.open(arc_path, mode="r:bz2") as tf:
        tf.extractall(out)
    for p in out.rglob("*"):
        if p.is_file():
            # preskoči arhive ako se nađu
            low = p.name.lower()
            if low.endswith((".bz2",".gz",".tar",".tgz",".zip")):
                continue
            dst = HAM / p.name
            i = 1
            while dst.exists():
                dst = HAM / f"{p.stem}_{i}{''.join(p.suffixes)}"
                i += 1
            shutil.move(str(p), str(dst))
            moved += 1
    shutil.rmtree(out, ignore_errors=True)

shutil.rmtree(TMP, ignore_errors=True)
print("Moved to ham:", moved)
print("Final counts:")
print("  ham :", len(list(HAM.glob('*'))))
