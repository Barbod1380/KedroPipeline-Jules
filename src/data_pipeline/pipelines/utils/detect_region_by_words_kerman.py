from .CONSTS_kerman import CONSTS_kerman
from .detect_region_by_words_base import detect_region_by_words_base

def detect_region_by_words_kerman(input_string: str, rules=None, replacements=None) -> int:
    return detect_region_by_words_base(input_string, CONSTS_kerman, rules=rules, replacements=replacements)
