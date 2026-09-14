#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepares the Long-Horizon Instruction & Russian Folklore Corpus:
1. Russian Folklore, Rhymes, and Proverbs (including user's target "хочешь сей а хочешь куй...").
2. 1024-token long Needle-in-a-Haystack tasks with coherent grammatical Russian text from wiki_train.txt.
3. 1024-token multi-step state tracking tasks.
4. Folklore Typo Evaluation test suite.
"""

import os
import random

def main():
    random.seed(42)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    wiki_path = os.path.join(data_dir, "wiki_train.txt")
    out_path = os.path.join(data_dir, "long_instruct_corpus.txt")
    eval_path = os.path.join(data_dir, "folklore_typo_eval.txt")

    # 1. Load coherent Russian Wikipedia sentences for haystacks
    wiki_sentences = []
    if os.path.exists(wiki_path):
        with open(wiki_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s.endswith("<eos>"):
                    s = s[:-5].strip()
                if len(s) > 20:
                    wiki_sentences.append(s)
    print(f"Loaded {len(wiki_sentences)} sentences for grammatical haystacks.")

    # 2. Russian Folklore, Rhymes, Idioms
    folklore_pairs = [
        ("хочешь сей а хочешь куй все равно получишь", "результат"),
        ("делу время", "потехе час"),
        ("без труда не выловишь", "рыбку из пруда"),
        ("терпенье и труд", "все перетрут"),
        ("семь раз отмерь", "один раз отрежь"),
        ("повторенье", "мать ученья"),
        ("век живи", "век учись"),
        ("тише едешь", "дальше будешь"),
        ("слово не воробей", "вылетит не поймаешь"),
        ("яблоко от яблони", "недалеко падает"),
        ("любишь кататься", "люби и саночки возить"),
        ("ученье свет", "а неученье тьма"),
        ("старый друг", "лучше новых двух"),
        ("не имей сто рублей", "а имей сто друзей"),
        ("один в поле", "не воин"),
    ]

    lines = []

    # Inject folklore directly and dialogically
    for prompt, target in folklore_pairs:
        for _ in range(25):
            lines.append(f"<user> {prompt} <bot> {target} <eos>")
            lines.append(f"{prompt} {target} <eos>")

    # 3. Generate 1024-token Needle-in-a-Haystack instances
    secrets = [
        ("квант", "физика"),
        ("резонанс", "музыка"),
        ("алгоритм", "логика"),
        ("память", "кристалл"),
        ("солитон", "волна"),
        ("матрица", "вектор"),
        ("фотон", "свет"),
        ("электрон", "заряд"),
    ]

    print("Generating 1024-token needle-in-a-haystack sequences...")
    for idx, (secret_key, secret_val) in enumerate(secrets):
        for rep in range(12):
            needle = f"секретный факт: {secret_key} это {secret_val}."
            question = f"вспомни секретный факт что это {secret_key}?"
            bot_reply = f"{secret_val}"

            # Build coherent haystack of ~950 tokens
            haystack_tokens = []
            while len(haystack_tokens) < 950:
                s = random.choice(wiki_sentences)
                words = s.split()
                haystack_tokens.extend(words)

            haystack_text = " ".join(haystack_tokens[:950])
            long_doc = f"<user> {needle} контекст: {haystack_text} вопрос: {question} <bot> {bot_reply} <eos>"
            lines.append(long_doc)

    # 4. Generate 1024-token State Tracking sequences
    places = ["дом", "лес", "поле", "река", "город", "сад", "замок", "остров"]
    subjects = ["кот", "волк", "лис", "медведь", "заяц", "сокол", "щука", "олень"]

    print("Generating 1024-token state tracking sequences...")
    for rep in range(30):
        target_subj = random.choice(subjects)
        curr_places = {s: random.choice(places) for s in subjects}
        story_steps = []

        # Generate ~80 distinct steps to reach ~1024 tokens
        for step in range(1, 85):
            s = random.choice(subjects)
            p = random.choice(places)
            curr_places[s] = p
            story_steps.append(f"шаг {step}: {s} идет в {p}.")

        final_place = curr_places[target_subj]
        story_text = " ".join(story_steps)
        long_state = f"<user> хроника перемещений: {story_text} вопрос: где в итоге находится {target_subj}? <bot> {final_place} <eos>"
        lines.append(long_state)

    random.shuffle(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        for l in lines:
            f.write(l + "\n")

    print(f"Successfully generated {len(lines)} instances in {out_path}.")

    # 5. Create Folklore Typo Evaluation test suite
    eval_pairs = [
        ("<user> хочешь сей а хочешь куй все равно получишь <bot>", "<user> хочишь сей а хочишь кй все равно получишь <bot>", "результат"),
        ("<user> делу время <bot>", "<user> делу времья <bot>", "потехе час"),
        ("<user> без труда не выловишь <bot>", "<user> без труда не выловиш <bot>", "рыбку из пруда"),
        ("<user> терпенье и труд <bot>", "<user> терпенье и трруд <bot>", "все перетрут"),
        ("<user> семь раз отмерь <bot>", "<user> семь раз отммерь <bot>", "один раз отрежь"),
        ("<user> ученье свет <bot>", "<user> ученье свеет <bot>", "а неученье тьма"),
        ("<user> тише едешь <bot>", "<user> тиши едешь <bot>", "дальше будешь"),
        ("<user> один в поле <bot>", "<user> один в поли <bot>", "не воин"),
    ]

    with open(eval_path, "w", encoding="utf-8") as f:
        for orig, typo, target in eval_pairs:
            f.write(f"{orig}\t{typo}\t{target}\n")

    print(f"Generated {len(eval_pairs)} typo evaluation pairs in {eval_path}.")

if __name__ == "__main__":
    main()
