#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepares the Clean Poker & Game Theory Pretraining Corpus:
Extracts, deduplicates, and normalizes 45 scientific papers and textbooks from:
C:\\projects\\.archive\\hmuriy-lm\\hmuriy-huy-0.1\\docs\\*.md
Outputs:
- data/poker_docs_train.txt (~90%)
- data/poker_docs_val.txt (~10%)
"""

import os
import glob
import re
import hashlib
import random

def clean_paragraph(p: str) -> str:
    # Remove markdown headers, links, image tags, code fences
    p = re.sub(r'#+\s*', '', p)
    p = re.sub(r'```[a-zA-Z]*', '', p)
    p = re.sub(r'```', '', p)
    p = re.sub(r'!\[.*?\]\(.*?\)', '', p)
    p = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', p)
    p = re.sub(r'[*_~`]', '', p)
    p = re.sub(r'[\r\t]+', ' ', p)
    p = re.sub(r'\s+', ' ', p).strip()
    return p

def main():
    random.seed(42)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    docs_dir = r"C:\projects\.archive\hmuriy-lm\hmuriy-huy-0.1\docs"

    files = sorted(glob.glob(os.path.join(docs_dir, "*.md")))
    print(f"Found {len(files)} markdown documents in {docs_dir}")

    seen_hashes = set()
    cleaned_sentences = []

    for f in files:
        with open(f, "r", encoding="utf-8", errors="ignore") as fp:
            raw_text = fp.read()

        # Deduplicate identical papers
        h = hashlib.md5(raw_text[:2000].encode('utf-8')).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        # Split into paragraphs/sections
        paragraphs = raw_text.split("\n\n")
        for p in paragraphs:
            cleaned = clean_paragraph(p)
            # Split paragraph into sentence chunks of reasonable size
            sentences = re.split(r'(?<=[.!?])\s+', cleaned)
            for s in sentences:
                s = s.strip()
                # Keep coherent English/scientific sentences
                if len(s) >= 30 and len(s) <= 400 and not s.startswith("|") and not s.startswith("-"):
                    cleaned_sentences.append(s)

    print(f"Total extracted clean sentences: {len(cleaned_sentences)}")

    # Deduplicate sentences
    unique_sentences = list(dict.fromkeys(cleaned_sentences))
    print(f"Unique sentences: {len(unique_sentences)}")

    # Shuffle deterministically
    random.shuffle(unique_sentences)

    # 90% train, 10% val split
    split_idx = int(len(unique_sentences) * 0.9)
    train_sents = unique_sentences[:split_idx]
    val_sents = unique_sentences[split_idx:]

    train_path = os.path.join(data_dir, "poker_docs_train.txt")
    val_path = os.path.join(data_dir, "poker_docs_val.txt")

    with open(train_path, "w", encoding="utf-8") as fp:
        for s in train_sents:
            fp.write(f"{s} <eos>\n")

    with open(val_path, "w", encoding="utf-8") as fp:
        for s in val_sents:
            fp.write(f"{s} <eos>\n")

    train_bytes = os.path.getsize(train_path)
    val_bytes = os.path.getsize(val_path)
    print(f"Generated '{train_path}': {len(train_sents)} lines ({train_bytes / 1024 / 1024:.2f} MB)")
    print(f"Generated '{val_path}': {len(val_sents)} lines ({val_bytes / 1024 / 1024:.2f} MB)")

if __name__ == "__main__":
    main()
