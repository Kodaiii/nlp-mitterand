#!/usr/bin/env python3
"""Construit le dataset d'entraînement QLoRA « Mitterrand 1981 ».

Lit les fichiers d'écriture lisibles dans data/sources/*.md, valide chaque
exemple, contrôle la couverture des 110 propositions et des citations, puis
écrit des fichiers JSONL au format conversationnel `messages` (compatible TRL
SFTTrainer / Unsloth, le chat template de Gemma convertit `assistant` en `model`).
Le prompt système de data/system_prompt.txt est placé en tête de chaque exemple.

Format d'un fichier source :

    @@@
    [PROPS] 21, 34          (optionnel : propositions couvertes)
    [USER]
    Question...
    [ASSISTANT]
    Réponse...
    [USER]                  (tours supplémentaires optionnels)
    ...

Usage : python3 scripts/build_dataset.py [--val-ratio 0.10] [--test-ratio 0.05] [--seed 42] [--no-system-prompt]
"""

import argparse
import csv
import difflib
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "data" / "sources"
OUT = ROOT / "data"
PROPOSITIONS_CSV = ROOT / "110_propositions_mitterrand_1981.csv"
CITATIONS_CSV = ROOT / "citations_francois_mitterrand.csv"
SYSTEM_PROMPT = ROOT / "data" / "system_prompt.txt"

SEPARATOR = "@@@"
TAG_RE = re.compile(r"^\[(USER|ASSISTANT|PROPS)\]\s*(.*)$")
ROLE = {"USER": "user", "ASSISTANT": "assistant"}

# Tics de style à proscrire dans les réponses (markdown, anachronismes lexicaux).
FORBIDDEN_IN_ANSWERS = [
    re.compile(r"^\s*[-*•]\s", re.M),  # listes à puces
    re.compile(r"^\s*#", re.M),  # titres markdown
    re.compile(r"\*\*"),  # gras markdown
    re.compile(r"\b(impactant|clairement parlant|en mode|du coup)\b", re.I),
]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("’", "'").replace("‘", "'").replace("«", '"').replace("»", '"')
    text = text.replace(" ", " ").replace("…", "...")
    text = re.sub(r"\s+", " ", text)
    # Insensible aux accents : le CSV écrit « Etre », les réponses « Être ».
    text = "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))
    return text.strip().lower()


def parse_file(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8")
    blocks = raw.split(f"\n{SEPARATOR}\n")
    examples = []
    for block_index, block in enumerate(blocks[1:], start=1):
        messages, props = [], []
        current_role, buffer = None, []

        def flush():
            if current_role is not None:
                messages.append({"role": ROLE[current_role], "content": "\n".join(buffer).strip()})

        for line in block.splitlines():
            match = TAG_RE.match(line.strip())
            if match and match.group(1) == "PROPS":
                props = [int(n) for n in re.findall(r"\d+", match.group(2))]
                continue
            if match:
                flush()
                current_role, buffer = match.group(1), []
                if match.group(2):
                    buffer.append(match.group(2))
                continue
            if current_role is not None:
                buffer.append(line)
        flush()
        examples.append(
            {
                "id": f"{path.stem}#{block_index}",
                "category": re.sub(r"^\d+[a-z]?_", "", path.stem),
                "props": props,
                "messages": messages,
            }
        )
    return examples


def validate(example: dict) -> list[str]:
    errors = []
    messages = example["messages"]
    if not messages:
        return ["exemple vide"]
    for i, message in enumerate(messages):
        expected = "user" if i % 2 == 0 else "assistant"
        if message["role"] != expected:
            errors.append(f"tour {i}: rôle {message['role']} au lieu de {expected}")
        if not message["content"]:
            errors.append(f"tour {i}: contenu vide")
    if messages[-1]["role"] != "assistant":
        errors.append("le dernier tour doit être une réponse assistant")
    for message in messages:
        if message["role"] != "assistant":
            continue
        for pattern in FORBIDDEN_IN_ANSWERS:
            if pattern.search(message["content"]):
                errors.append(f"motif interdit {pattern.pattern!r}")
    for n in example["props"]:
        if not 1 <= n <= 110:
            errors.append(f"proposition hors bornes: {n}")
    return errors


def load_quotes() -> list[str]:
    with CITATIONS_CSV.open(encoding="utf-8-sig", newline="") as f:
        return [row["Citation"].strip() for row in csv.DictReader(f) if row["Citation"].strip()]


def words(text: str) -> list[str]:
    """Mots normalisés, sans ponctuation : « scalpel du chirurgien : elle » == « scalpel du chirurgien, elle »."""
    return re.findall(r"[a-z0-9œæ]+", normalize(text))


def quote_found(quote: list[str], answer: list[str], answer_set: set[str], threshold: float = 0.85) -> bool:
    """Présence mot pour mot, ou quasi mot pour mot (coquille du CSV, accord, mot ajouté)."""
    n = len(quote)
    if " ".join(quote) in " ".join(answer):
        return True
    if sum(w in answer_set for w in set(quote)) < 0.8 * len(set(quote)):
        return False
    for start in range(0, max(1, len(answer) - n + 1)):
        window = answer[start : start + n + 2]
        if difflib.SequenceMatcher(None, quote, window, autojunk=False).ratio() >= threshold:
            return True
    return False


def stratified_split(examples: list[dict], val_ratio: float, test_ratio: float, seed: int) -> tuple[list, list, list]:
    rng = random.Random(seed)
    by_category = defaultdict(list)
    for example in examples:
        by_category[example["category"]].append(example)
    train, val, test = [], [], []
    for category in sorted(by_category):
        items = by_category[category][:]
        rng.shuffle(items)
        n_val = max(1, round(len(items) * val_ratio)) if len(items) >= 10 else 0
        n_test = max(1, round(len(items) * test_ratio)) if len(items) >= 10 else 0
        if n_val + n_test >= len(items):
            raise ValueError(f"Split impossible pour la catégorie {category!r}")
        test.extend(items[:n_test])
        val.extend(items[n_test : n_test + n_val])
        train.extend(items[n_test + n_val :])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def write_jsonl(path: Path, examples: list[dict], system_prompt: str | None, with_meta: bool = False) -> None:
    system = [{"role": "system", "content": system_prompt}] if system_prompt else []
    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            record = {"messages": system + example["messages"]}
            if with_meta:
                record = {"id": example["id"], "category": example["category"], "props": example["props"], **record}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-system-prompt", action="store_true", help="ne pas ajouter data/system_prompt.txt")
    args = parser.parse_args()
    if not 0 < args.val_ratio < 1 or not 0 < args.test_ratio < 1 or args.val_ratio + args.test_ratio >= 1:
        parser.error("--val-ratio et --test-ratio doivent être positifs et leur somme doit être inférieure à 1")
    system_prompt = None if args.no_system_prompt else SYSTEM_PROMPT.read_text(encoding="utf-8").strip()

    examples = []
    for path in sorted(SOURCES.glob("*.md")):
        examples.extend(parse_file(path))

    failed = False
    for example in examples:
        for error in validate(example):
            print(f"ERREUR {example['id']}: {error}", file=sys.stderr)
            failed = True

    first_questions = Counter(normalize(e["messages"][0]["content"]) for e in examples if e["messages"])
    for question, count in first_questions.items():
        if count > 1:
            print(f"ERREUR question dupliquée ({count}x): {question[:80]}", file=sys.stderr)
            failed = True
    if failed:
        return 1

    covered_props = Counter(n for e in examples for n in e["props"])
    missing_props = [n for n in range(1, 111) if n not in covered_props]

    answers = [words(m["content"]) for e in examples for m in e["messages"] if m["role"] == "assistant"]
    answer_sets = [set(a) for a in answers]
    quotes = load_quotes()
    quote_hits = {}
    for quote in quotes:
        quote_words = words(quote)
        quote_hits[quote] = sum(quote_found(quote_words, a, s) for a, s in zip(answers, answer_sets))
    missing_quotes = [q for q, hits in quote_hits.items() if hits == 0]
    overused_quotes = [(q, hits) for q, hits in quote_hits.items() if hits > 4]

    train, val, test = stratified_split(examples, args.val_ratio, args.test_ratio, args.seed)
    write_jsonl(OUT / "mitterrand_train.jsonl", train, system_prompt)
    write_jsonl(OUT / "mitterrand_val.jsonl", val, system_prompt)
    write_jsonl(OUT / "mitterrand_test.jsonl", test, system_prompt)
    write_jsonl(OUT / "mitterrand_full_with_meta.jsonl", examples, system_prompt, with_meta=True)

    answer_words = [len(m["content"].split()) for e in examples for m in e["messages"] if m["role"] == "assistant"]
    print(f"Exemples : {len(examples)}  (train {len(train)} / val {len(val)} / test {len(test)})")
    print(f"Multi-tours : {sum(len(e['messages']) > 2 for e in examples)}")
    print(f"Prompt système : {f'{len(system_prompt.split())} mots' if system_prompt else 'aucun'}")
    print("Par catégorie :")
    for category, count in sorted(Counter(e["category"] for e in examples).items()):
        print(f"  {category:<28} {count}")
    print(
        f"Mots par réponse : min {min(answer_words)} / moy {sum(answer_words) // len(answer_words)}"
        f" / max {max(answer_words)}"
    )
    print(f"Propositions couvertes : {110 - len(missing_props)}/110" + (f"  manquantes : {missing_props}" if missing_props else ""))
    print(f"Citations reprises mot pour mot : {len(quotes) - len(missing_quotes)}/{len(quotes)}")
    for quote in missing_quotes:
        print(f"  absente : {quote}")
    for quote, hits in overused_quotes:
        print(f"  surreprésentée ({hits}x) : {quote}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
