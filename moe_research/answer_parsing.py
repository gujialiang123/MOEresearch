"""Strict + tolerant GSM8K answer parsing and length decomposition (v23+ plan §十)."""
import re

# strict: a number immediately after a `####` marker
_STRICT = re.compile(r"####\s*\$?\\?boxed\{?\s*([-+]?[0-9][0-9,]*\.?[0-9]*)")
_STRICT_SIMPLE = re.compile(r"####\s*([-+]?[0-9][0-9,]*\.?[0-9]*)")
# tolerant: also accept FINAL:, Answer:, $, \boxed{}
_TOL_MARKERS = [
    re.compile(r"####\s*\$?\s*\\?boxed\{\s*([-+]?[0-9][0-9,]*\.?[0-9]*)\s*\}"),
    re.compile(r"####\s*(?:Answer:?\s*)?\$?\s*([-+]?[0-9][0-9,]*\.?[0-9]*)"),
    re.compile(r"FINAL:?\s*\$?\s*([-+]?[0-9][0-9,]*\.?[0-9]*)"),
    re.compile(r"\\boxed\{\s*([-+]?[0-9][0-9,]*\.?[0-9]*)\s*\}"),
    re.compile(r"[Tt]he (?:final )?answer is\s*\$?\s*([-+]?[0-9][0-9,]*\.?[0-9]*)"),
]
_ANY_NUM = re.compile(r"[-+]?[0-9][0-9,]*\.?[0-9]*")


def _norm(s):
    if s is None:
        return None
    s = s.strip().replace(",", "").lstrip("+")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s if s not in ("", "-", "+") else None


def parse_strict(text: str):
    """Only accept a number right after `####`. Returns (value, status)."""
    m = _STRICT_SIMPLE.findall(text)
    if m:
        v = _norm(m[-1])
        return (v, "strict") if v is not None else (None, "parse_failure")
    return (None, "parse_failure")


def parse_tolerant(text: str):
    """Accept #### / FINAL: / boxed / 'answer is' forms. Returns (value, status)."""
    for rx in _TOL_MARKERS:
        m = rx.findall(text)
        if m:
            v = _norm(m[-1])
            if v is not None:
                return v, "tolerant"
    # last-resort: last number anywhere
    nums = _ANY_NUM.findall(text)
    if nums:
        v = _norm(nums[-1])
        if v is not None:
            return v, "tolerant_lastnum"
    return None, "parse_failure"


def parse_gold(answer: str):
    v, _ = parse_strict(answer)
    if v is not None:
        return v
    if "####" in answer:
        return _norm(answer.split("####")[-1])
    return None


def first_marker_char(text: str) -> int:
    """Char index of first `####` or FINAL:, else -1."""
    idxs = [text.find("####"), text.find("FINAL")]
    idxs = [i for i in idxs if i >= 0]
    return min(idxs) if idxs else -1
