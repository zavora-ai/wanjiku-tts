"""Kikuyu text normalizer.

Unifies diacritic variants, expands numbers, cleans punctuation.
All Kikuyu text should pass through normalize() before training.
"""

import re
import unicodedata

# Diacritic mapping: all variants → canonical tilde (ĩ, ũ)
_DIACRITIC_MAP = str.maketrans({
    # Macron variants → tilde
    'ī': 'ĩ', 'ū': 'ũ', 'Ī': 'Ĩ', 'Ū': 'Ũ',
    # Grave accent variants
    'ì': 'ĩ', 'ù': 'ũ', 'Ì': 'Ĩ', 'Ù': 'Ũ',
    # Acute accent variants (sometimes used for vowel quality)
    'í': 'ĩ', 'ú': 'ũ', 'Í': 'Ĩ', 'Ú': 'Ũ',
    # Circumflex variants
    'î': 'ĩ', 'û': 'ũ', 'Î': 'Ĩ', 'Û': 'Ũ',
    # Greek/breve variants (found in African Storybook)
    'ῖ': 'ĩ', 'ῦ': 'ũ',
    'ŭ': 'ũ', 'ȋ': 'ĩ',
    'ĭ': 'ĩ',
    # Combining diacritics on i/u that should be tilde
    # These are handled separately via NFD decomposition
})

# Kikuyu number words
_NUMBERS = {
    0: 'hatarĩ', 1: 'ĩmwe', 2: 'igĩrĩ', 3: 'ithatu', 4: 'inya',
    5: 'ithano', 6: 'ithathatu', 7: 'mũgwanja', 8: 'inyanya',
    9: 'kenda', 10: 'ikũmi',
    20: 'mĩrongo ĩĩrĩ', 30: 'mĩrongo ĩtatũ', 40: 'mĩrongo ĩna',
    50: 'mĩrongo ĩtano', 60: 'mĩrongo ĩtandatũ', 70: 'mĩrongo mũgwanja',
    80: 'mĩrongo ĩnana', 90: 'mĩrongo kenda', 100: 'igana',
    1000: 'ngiri',
}


def _normalize_combining(text):
    """Handle combining diacritics via NFD decomposition."""
    # Decompose to base + combining marks
    nfd = unicodedata.normalize('NFD', text)
    result = []
    i = 0
    while i < len(nfd):
        ch = nfd[i]
        # Check if next char is a combining diacritic on i or u
        if i + 1 < len(nfd) and ch.lower() in ('i', 'u'):
            combining = nfd[i + 1]
            cat = unicodedata.category(combining)
            if cat.startswith('M'):  # combining mark
                # Map i+combining → ĩ, u+combining → ũ
                if ch == 'i':
                    result.append('ĩ')
                elif ch == 'I':
                    result.append('Ĩ')
                elif ch == 'u':
                    result.append('ũ')
                elif ch == 'U':
                    result.append('Ũ')
                i += 2
                continue
        result.append(ch)
        i += 1
    return unicodedata.normalize('NFC', ''.join(result))


def _expand_number(n):
    """Expand integer to Kikuyu words (basic, up to 9999)."""
    if n in _NUMBERS:
        return _NUMBERS[n]
    if n < 0:
        return 'thaĩ ' + _expand_number(-n)
    if n < 20:
        return f'ikũmi na {_NUMBERS[n - 10]}'
    if n < 100:
        tens = (n // 10) * 10
        ones = n % 10
        if ones == 0:
            return _NUMBERS[tens]
        return f'{_NUMBERS[tens]} na {_NUMBERS[ones]}'
    if n < 1000:
        h = n // 100
        rest = n % 100
        prefix = f'magana {_NUMBERS[h]}' if h > 1 else 'igana'
        if rest == 0:
            return prefix
        return f'{prefix} na {_expand_number(rest)}'
    if n < 10000:
        t = n // 1000
        rest = n % 1000
        prefix = f'ngiri {_NUMBERS[t]}' if t > 1 else 'ngiri'
        if rest == 0:
            return prefix
        return f'{prefix} na {_expand_number(rest)}'
    return str(n)


def _expand_numbers_in_text(text):
    """Replace digit sequences with Kikuyu number words."""
    def _replace(m):
        try:
            return _expand_number(int(m.group()))
        except (ValueError, KeyError):
            return m.group()
    return re.sub(r'\b\d+\b', _replace, text)


def normalize(text, expand_numbers=True):
    """Normalize Kikuyu text to canonical form.

    - Unifies all diacritic variants to tilde (ĩ, ũ)
    - Optionally expands numbers to Kikuyu words
    - Cleans whitespace and punctuation
    """
    if not text:
        return text

    # Step 1: Handle combining diacritics
    text = _normalize_combining(text)

    # Step 2: Direct character mapping
    text = text.translate(_DIACRITIC_MAP)

    # Step 3: Expand numbers
    if expand_numbers:
        text = _expand_numbers_in_text(text)

    # Step 4: Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Step 5: Normalize quotes
    text = re.sub(r'[""„‟]', '"', text)
    text = re.sub(r'[''‚‛]', "'", text)

    return text


if __name__ == '__main__':
    tests = [
        # Macron → tilde
        ('Ūgíthamia mīūngūrwa', 'Ũgĩthamia mĩũngũrwa'),
        # Already tilde (no change)
        ('Kĩambĩrĩria-inĩ kĩa maũndũ', 'Kĩambĩrĩria-inĩ kĩa maũndũ'),
        # Bare text (no change — can't recover diacritics)
        ('Kiambiriria', 'Kiambiriria'),
        # Greek/breve variants
        ('Mwarῖmῦ Njeri', 'Mwarĩmũ Njeri'),
        # Number expansion
        ('Mĩaka 100 mĩhĩtũku', 'Mĩaka igana mĩhĩtũku'),
    ]
    passed = 0
    for inp, expected in tests:
        result = normalize(inp)
        ok = result == expected
        passed += ok
        status = '✓' if ok else '✗'
        if not ok:
            print(f'  {status} normalize({inp!r})')
            print(f'    expected: {expected!r}')
            print(f'    got:      {result!r}')
        else:
            print(f'  {status} {inp!r} → {result!r}')
    print(f'\n{passed}/{len(tests)} tests passed')
