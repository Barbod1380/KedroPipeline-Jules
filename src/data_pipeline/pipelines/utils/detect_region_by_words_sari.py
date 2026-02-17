from .CONSTS_sari import CONSTS_sari
from .detect_region_by_words_base import detect_region_by_words_base

def detect_region_by_words_sari(input_string: str, rules=None, replacements=None) -> int:
    return detect_region_by_words_base(input_string, CONSTS_sari, rules=rules, replacements=replacements)
