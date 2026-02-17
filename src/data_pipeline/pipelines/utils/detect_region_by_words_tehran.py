from .CONSTS_tehran import CONSTS_tehran
from .detect_region_by_words_base import detect_region_by_words_base

def detect_region_by_words_tehran(input_string: str, rules=None, replacements=None) -> int:
    return detect_region_by_words_base(
        input_string, 
        CONSTS_tehran, 
        rules=rules, 
        replacements=replacements
    )