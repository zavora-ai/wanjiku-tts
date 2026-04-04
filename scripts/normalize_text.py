"""Kikuyu text normalization for TTS training."""
import argparse
import json
import re


# Kikuyu number words
ONES = ["", "ĩmwe", "igĩrĩ", "ithatũ", "inya", "ithano", "ithathatũ", "mũgwanja", "inyanya", "kenda"]
TENS = ["", "ikũmi", "mĩrongo ĩĩrĩ", "mĩrongo ĩtatũ", "mĩrongo ĩna", "mĩrongo ĩtano",
        "mĩrongo ĩtandatũ", "mĩrongo mũgwanja", "mĩrongo ĩnana", "mĩrongo kenda"]


def number_to_kikuyu(n):
    """Convert integer to Kikuyu words (basic, up to 9999)."""
    if n == 0:
        return "hatarĩ"
    parts = []
    if n >= 1000:
        parts.append(f"ngiri {ONES[n // 1000]}" if n // 1000 > 1 else "ngiri ĩmwe")
        n %= 1000
    if n >= 100:
        parts.append(f"magana {ONES[n // 100]}" if n // 100 > 1 else "igana rĩmwe")
        n %= 100
    if n >= 10:
        parts.append(TENS[n // 10])
        n %= 10
    if n > 0:
        parts.append(f"na {ONES[n]}")
    return " ".join(parts).strip()


def expand_numbers(text):
    """Replace digit sequences with Kikuyu words."""
    def _replace(m):
        n = int(m.group())
        if n > 9999:
            return m.group()  # Leave large numbers as-is for now
        return number_to_kikuyu(n)
    return re.sub(r"\b\d{1,4}\b", _replace, text)


def expand_currency(text):
    """Expand Ksh amounts."""
    def _replace(m):
        amount = int(m.group(1))
        return f"ciringĩ {number_to_kikuyu(amount)}"
    return re.sub(r"Ksh\.?\s*(\d+)", _replace, text, flags=re.IGNORECASE)


def normalize_punctuation(text):
    """Standardize punctuation."""
    text = re.sub(r"[""„]", '"', text)
    text = re.sub(r"[''‚]", "'", text)
    text = re.sub(r"\.{2,}", "…", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize(text):
    """Full normalization pipeline."""
    text = expand_currency(text)
    text = expand_numbers(text)
    text = normalize_punctuation(text)
    return text


def main():
    parser = argparse.ArgumentParser(description="Normalize Kikuyu text for TTS")
    parser.add_argument("--input", required=True, help="Input JSONL manifest")
    parser.add_argument("--output", required=True, help="Output normalized JSONL")
    args = parser.parse_args()

    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            entry = json.loads(line)
            if entry.get("text"):
                entry["text"] = normalize(entry["text"])
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Normalized: {args.input} → {args.output}")


if __name__ == "__main__":
    main()
