#!/usr/bin/env python
"""Fetch the EnterpriseRAG-Bench corpus and split the answer key out of it.

The dataset is not redistributed in this repository; this script pulls it from
the upstream release. It also enforces the leakage protocol described in the
README: gold answers are written OUTSIDE the repository, and what stays behind
carries only question ids and question text.

    python scripts/fetch_corpus.py [--gold-out PATH]

Note that the corpus archive itself contains a questions.jsonl with the answer
key inside it. That file is moved out too - a detail worth knowing if you ever
re-extract the archive by hand.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CORPUS = ROOT / "data" / "corpus"
BENCH = ROOT / "data" / "bench"

BASE = "https://github.com/onyx-dot-app/EnterpriseRAG-Bench/releases/download/v1.0.0"
ASSETS = {
    "all_documents.zip": f"{BASE}/all_documents.zip",
    "questions.jsonl": f"{BASE}/questions.jsonl",
    "extra_questions.jsonl": f"{BASE}/extra_questions.jsonl",
}

# Fields that constitute the answer key. question_type and source_types are
# included deliberately: "info_not_found" tells a system to abstain before doing
# any work, and source_types tells it which sources to search. Both are hints.
ANSWER_FIELDS = {"gold_answer", "answer_facts", "expected_doc_ids", "question_type", "source_types"}


def download(name: str, url: str) -> Path:
    dest = RAW / name
    if dest.exists():
        print(f"  {name:26s} already present ({dest.stat().st_size/1e6:.1f} MB)")
        return dest
    RAW.mkdir(parents=True, exist_ok=True)
    print(f"  {name:26s} downloading...", flush=True)
    urllib.request.urlretrieve(url, dest)
    print(f"  {name:26s} {dest.stat().st_size/1e6:.1f} MB")
    return dest


def split_questions(gold_out: Path) -> None:
    """Write the blind question set into the repo, the answer key outside it."""
    BENCH.mkdir(parents=True, exist_ok=True)
    gold_out.parent.mkdir(parents=True, exist_ok=True)

    blind, gold = [], []
    for name in ("questions.jsonl", "extra_questions.jsonl"):
        path = RAW / name
        if not path.exists():
            continue
        split = "extra" if "extra" in name else "main"
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            blind.append(
                {"question_id": rec["question_id"], "question": rec["question"], "split": split}
            )
            gold.append(rec | {"split": split})

    (BENCH / "questions_blind.jsonl").write_text(
        "".join(json.dumps(b) + "\n" for b in blind)
    )
    gold_out.write_text("".join(json.dumps(g) + "\n" for g in gold))

    # the originals carry answers; they must not linger in the repo
    for name in ("questions.jsonl", "extra_questions.jsonl"):
        (RAW / name).unlink(missing_ok=True)

    print(f"\n  blind set : data/bench/questions_blind.jsonl  ({len(blind)} questions)")
    print(f"  gold key  : {gold_out}  ({len(gold)} records, outside the repo)")


def extract(archive: Path) -> None:
    if CORPUS.exists() and any(CORPUS.iterdir()):
        print(f"  corpus already extracted at {CORPUS}")
        return
    CORPUS.mkdir(parents=True, exist_ok=True)
    print("  extracting ~512k files, this takes a minute...", flush=True)
    with zipfile.ZipFile(archive) as z:
        z.extractall(CORPUS)


def scrub_corpus(gold_out: Path) -> None:
    """The archive hides a copy of the answer key inside it. Move it out."""
    stray = CORPUS / "questions.jsonl"
    if stray.exists():
        shutil.move(str(stray), str(gold_out.with_name(gold_out.stem + "_dupe.jsonl")))
        print(f"  moved stray answer key out of the corpus tree")


def verify() -> bool:
    """Fail loudly if any answer field survived inside the repository."""
    leaks = []
    for path in BENCH.rglob("*.jsonl"):
        for line in path.read_text().splitlines()[:5]:
            if line.strip() and (ANSWER_FIELDS & set(json.loads(line))):
                leaks.append(str(path))
                break
    if (CORPUS / "questions.jsonl").exists():
        leaks.append(str(CORPUS / "questions.jsonl"))
    if leaks:
        print("\n  LEAK DETECTED - answer fields inside the repo:")
        for f in leaks:
            print("   ", f)
        return False
    print("  leak check : clean, no answer fields inside the repo")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gold-out",
        type=Path,
        default=Path(os.environ.get("GOLD_ANSWERS_PATH") or Path.home() / "Downloads" / "erb_gold.jsonl"),
        help="where to write the answer key (must be outside this repo)",
    )
    args = ap.parse_args()

    if ROOT in args.gold_out.resolve().parents:
        sys.exit(f"refusing to write the answer key inside the repo: {args.gold_out}")

    print("Fetching EnterpriseRAG-Bench v1.0.0\n")
    for name, url in ASSETS.items():
        download(name, url)

    print("\nExtracting corpus")
    extract(RAW / "all_documents.zip")

    print("\nSplitting questions")
    split_questions(args.gold_out)
    scrub_corpus(args.gold_out)

    print()
    if not verify():
        sys.exit(1)
    print(f"\nDone. Next: python scripts/intake.py")


if __name__ == "__main__":
    main()
