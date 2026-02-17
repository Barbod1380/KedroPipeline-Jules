from .CONSTS_gilan import CONSTS_gilan
from .detect_region_by_words_base import detect_region_by_words_base

def detect_region_by_words_gilan(input_string: str, rules=None, replacements=None) -> int:
    return detect_region_by_words_base(input_string, CONSTS_gilan, rules=rules, replacements=replacements)