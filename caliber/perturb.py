"""Numeric perturbation operator pi.

Given a financial passage with numeric content, returns a near-duplicate in
which a single numerical fact is altered while context, length, and syntax are
preserved. Five rule categories: magnitude, polarity, period, unit, currency.
Each call applies one rule to one span under a deterministic per-passage seed,
and returns None if no eligible span or rule fires.
"""
from __future__ import annotations

import re
import random
from dataclasses import dataclass
from typing import Optional, Tuple, List, Callable

NUM_RE = re.compile(
    r"""
    (?<![A-Za-z0-9])
    (?:
        \$?\d{1,3}(?:,\d{3})+(?:\.\d+)?         # 1,234.56
      | \$?\d+\.\d+                              # 12.4
      | \$?\d+%                                  # 12%
      | \d+\s?(?:bps|bp|basis\s+points?)         # 25 bps
      | \d+(?:\.\d+)?\s?(?:million|billion|thousand|M|B|K)\b
      | (?:Q[1-4]|FY)\s?\d{2,4}                  # Q3 2023, FY2022
      | \b(?:19|20)\d{2}\b                       # 2023
      | [+\-\u2212]\s?\d+(?:\.\d+)?%?            # +12% or -12%
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

POLARITY_PAIRS = [
    ("gain", "loss"), ("gains", "losses"),
    ("profit", "loss"), ("profits", "losses"),
    ("grew", "fell"), ("growing", "falling"),
    ("increased", "decreased"), ("increasing", "decreasing"),
    ("rose", "declined"), ("rising", "declining"),
    ("exceeded", "missed"), ("exceeding", "missing"),
    ("above", "below"),
    ("surplus", "deficit"),
    ("positive", "negative"),
    ("up", "down"),
    ("higher", "lower"),
    ("strong", "weak"),
]

CURRENCY_CODE_PAIRS = [("USD", "EUR"), ("USD", "JPY"), ("USD", "GBP"),
                       ("EUR", "JPY"), ("EUR", "GBP")]
CURRENCY_SYMBOL_PAIRS = [("$", "€"), ("$", "¥"), ("$", "£"), ("€", "£")]


def _find_numeric_spans(text: str) -> List[Tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group()) for m in NUM_RE.finditer(text)]


def _replace_span(text: str, start: int, end: int, new: str) -> str:
    return text[:start] + new + text[end:]


def _rule_magnitude(text: str, rng: random.Random) -> Optional[str]:
    """Decimal shift / factor-of-10 swap on a numeric span.
    Skips year-like and Q/FY period spans (handled by _rule_period)."""
    spans = _find_numeric_spans(text)
    rng.shuffle(spans)
    for s, e, surface in spans:
        # Skip period-like spans
        if re.match(r"^(Q[1-4]|FY)", surface, re.IGNORECASE):
            continue
        # Skip standalone 4-digit years (1900..2099)
        if re.fullmatch(r"(?:19|20)\d{2}", surface):
            continue
        # Try decimal shift first
        m = re.search(r"(\d+)\.(\d+)", surface)
        if m and len(m.group(2)) >= 1:
            integer, decimal = m.group(1), m.group(2)
            mode = rng.choice(["left", "right"])
            if mode == "left" and len(integer) >= 1:
                new_int = integer[:-1] if len(integer) > 1 else "0"
                new_dec = integer[-1] + decimal
                new_num = f"{new_int}.{new_dec}"
            else:
                new_int = integer + decimal[0]
                new_dec = decimal[1:] if len(decimal) > 1 else "0"
                new_num = f"{new_int}.{new_dec}"
            new_surface = surface[:m.start()] + new_num + surface[m.end():]
            if new_surface != surface:
                return _replace_span(text, s, e, new_surface)
        # Otherwise factor-of-10 swap on the leading digit run
        m2 = re.search(r"(\d+)", surface)
        if m2:
            digits = m2.group(1)
            if digits.startswith("0"):
                continue
            # Skip if the digit run is itself a year
            if re.fullmatch(r"(?:19|20)\d{2}", digits):
                continue
            mode = rng.choice(["x10", "div10"])
            if mode == "x10":
                new_digits = digits + "0"
            else:
                if len(digits) <= 1:
                    continue
                new_digits = digits[:-1]
            new_surface = surface[:m2.start()] + new_digits + surface[m2.end():]
            if new_surface != surface:
                return _replace_span(text, s, e, new_surface)
    return None


def _rule_polarity(text: str, rng: random.Random) -> Optional[str]:
    """Direction-word swap within +/-50 chars of a numeric span."""
    spans = _find_numeric_spans(text)
    if not spans:
        return None
    rng.shuffle(spans)
    pairs = POLARITY_PAIRS[:]
    rng.shuffle(pairs)
    for s, e, _ in spans:
        ctx_lo = max(0, s - 50); ctx_hi = min(len(text), e + 50)
        ctx = text[ctx_lo:ctx_hi]
        for a, b in pairs:
            for word_a, word_b in [(a, b), (b, a)]:
                pat = re.compile(rf"\b{re.escape(word_a)}\b", re.IGNORECASE)
                m = pat.search(ctx)
                if m:
                    g = m.group()
                    # Preserve case
                    if g.isupper():
                        repl = word_b.upper()
                    elif g[0].isupper():
                        repl = word_b.capitalize()
                    else:
                        repl = word_b
                    abs_start = ctx_lo + m.start()
                    abs_end = ctx_lo + m.end()
                    return _replace_span(text, abs_start, abs_end, repl)
    # Sign flip on +N% / -N%
    m = re.search(r"([+\-\u2212])\s?(\d+(?:\.\d+)?%?)", text)
    if m:
        sign = "+" if m.group(1) in ("-", "\u2212") else "-"
        return text[:m.start()] + sign + m.group(2) + text[m.end():]
    return None


def _rule_period(text: str, rng: random.Random) -> Optional[str]:
    """Q[1-4] YYYY / FYyyyy / standalone year shift."""
    # Try Q-period first
    m = re.search(r"\b(Q[1-4])\s?(\d{2,4})\b", text, re.IGNORECASE)
    if m:
        q = m.group(1)
        y = int(m.group(2))
        delta = rng.choice([-2, -1, 1, 2])
        new_y = y + delta
        # Preserve digit width
        new_y_s = str(new_y).zfill(len(m.group(2)))
        return text[:m.start()] + f"{q} {new_y_s}" + text[m.end():]
    # FY year
    m = re.search(r"\b(FY)\s?(\d{2,4})\b", text, re.IGNORECASE)
    if m:
        fy = m.group(1)
        y = int(m.group(2))
        delta = rng.choice([-2, -1, 1, 2])
        new_y = y + delta
        new_y_s = str(new_y).zfill(len(m.group(2)))
        return text[:m.start()] + f"{fy}{new_y_s}" + text[m.end():]
    # Standalone year
    years = list(re.finditer(r"\b(?:19|20)\d{2}\b", text))
    if years:
        m = rng.choice(years)
        y = int(m.group())
        delta = rng.choice([-2, -1, 1, 2])
        return text[:m.start()] + str(y + delta) + text[m.end():]
    return None


def _rule_unit(text: str, rng: random.Random) -> Optional[str]:
    """million<->billion, thousand<->million, bps<->percent."""
    swaps = [
        (r"\bmillion\b", "billion", "million"),
        (r"\bbillion\b", "million", "billion"),
        (r"\bthousand\b", "million", "thousand"),
        (r"\bbps\b", "percent", "bps"),
        (r"\bpercent\b", "bps", "percent"),
    ]
    rng.shuffle(swaps)
    for pat_str, repl, _orig in swaps:
        pat = re.compile(pat_str, re.IGNORECASE)
        m = pat.search(text)
        if m:
            g = m.group()
            if g.isupper():
                fixed_repl = repl.upper()
            elif g[0].isupper():
                fixed_repl = repl.capitalize()
            else:
                fixed_repl = repl
            return text[:m.start()] + fixed_repl + text[m.end():]
    return None


def _rule_currency(text: str, rng: random.Random) -> Optional[str]:
    """ISO code swap or symbol swap."""
    code_pairs = CURRENCY_CODE_PAIRS[:]; rng.shuffle(code_pairs)
    for a, b in code_pairs:
        for x, y in [(a, b), (b, a)]:
            m = re.search(rf"\b{x}\b", text)
            if m:
                return text[:m.start()] + y + text[m.end():]
    sym_pairs = CURRENCY_SYMBOL_PAIRS[:]; rng.shuffle(sym_pairs)
    for a, b in sym_pairs:
        for x, y in [(a, b), (b, a)]:
            i = text.find(x)
            if i >= 0:
                return text[:i] + y + text[i + len(x):]
    return None


_RULES: List[Tuple[str, Callable[[str, random.Random], Optional[str]]]] = [
    ("magnitude", _rule_magnitude),
    ("polarity",  _rule_polarity),
    ("period",    _rule_period),
    ("unit",      _rule_unit),
    ("currency",  _rule_currency),
]


@dataclass
class Perturbation:
    category: str
    text: str

class Perturber:
    def __init__(self, seed: int = 42, categories: Optional[List[str]] = None):
        self.seed = seed
        self.categories = categories or [c for c, _ in _RULES]

    def is_eligible(self, text: str, min_numeric: int = 2) -> bool:
        return len(NUM_RE.findall(text)) >= min_numeric

    def perturb(self, text: str, key: Optional[str] = None) -> Optional[Perturbation]:
        # Per-text deterministic RNG
        s = hash((self.seed, key or text)) & 0xFFFFFFFF
        rng = random.Random(s)
        # Try categories in shuffled order; stop at first success
        rules = [(c, fn) for c, fn in _RULES if c in self.categories]
        rng.shuffle(rules)
        for cat, fn in rules:
            new_text = fn(text, rng)
            if new_text is not None and new_text != text:
                # Sanity: edit distance not too large
                if abs(len(new_text) - len(text)) > 30:
                    continue
                return Perturbation(category=cat, text=new_text)
        return None


if __name__ == "__main__":
    p = Perturber(seed=42)
    samples = [
        "Apple revenue grew 12.4% year-over-year in Q3 2023, exceeding analyst estimates by $1.2 billion.",
        "The fund reported a net loss of USD 50 million for FY2022, down from a USD 200 million gain.",
        "Net interest margin compressed by 25 bps in the first quarter of 2024.",
    ]
    for s in samples:
        out = p.perturb(s)
        print(f"[{out.category if out else 'NONE'}]")
        print(f"  in : {s}")
        print(f"  out: {out.text if out else '(none)'}")
        print()
