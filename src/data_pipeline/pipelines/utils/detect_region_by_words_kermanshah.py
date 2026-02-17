from .CONSTS_kermanshah import CONSTS_kermanshah
from .detect_region_by_words_base import detect_region_by_words_base

def detect_region_by_words_kermanshah(input_string: str, rules=None, replacements=None) -> int:
    return detect_region_by_words_base(input_string, CONSTS_kermanshah, rules=rules, replacements=replacements)