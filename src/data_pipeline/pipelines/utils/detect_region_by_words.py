from .CONSTS.CONSTS import RULES, REPLACEMENT_DICT, ABBREVIATION_DICT, KEYWORDS_TO_CUT_BEFORE, KEYWORDS_TO_CUT_AFTER
from ..utils.load_rules import replace_phrases


def detect_region_by_words(input_string: str, rules=None, replacements=None) -> int:
    """
    Matches input string against preprocessed rules
    Returns 3-digit code if exactly one distinct rule matches
    Returns -1 if multiple distinct outputs match or no matches
    Modified to:
    1. Add spaces around the remaining string
    2. Remove all text after and including "کوچه" or "بن بست"
    3. Apply replacements
    """

    if not rules:
        rules = RULES

    if not replacements:
        replacements = REPLACEMENT_DICT

    # Add spaces around the remaining string
    input_string = " " + input_string.strip() + " "

    if " فرستنده " in input_string:
        return -1

    for abbr in ABBREVIATION_DICT:
        input_string = input_string.replace(abbr, ABBREVIATION_DICT[abbr])

    for keyword in KEYWORDS_TO_CUT_BEFORE:
        index = input_string.find(keyword)
        if index != -1:
            input_string = input_string[index:]

    for keyword in KEYWORDS_TO_CUT_AFTER:
        index = input_string.find(keyword)
        if index != -1:
            input_string = input_string[: index + 1]

    for remove_str in [" شهید ", " سید "]:
        input_string = input_string.replace(remove_str, " ")

    # Apply phrase replacements
    input_string = replace_phrases(input_string, replacements)

    matched_outputs = set()
    for conditions, number in rules:
        match = True
        for operator, phrase in conditions:
            if operator == " + ":
                if phrase not in input_string:
                    match = False
                    break
            elif operator == " - ":
                if phrase in input_string:
                    match = False
                    break
        if match:
            matched_outputs.add(number)
            # Early exit if we already have multiple distinct outputs
            if len(matched_outputs) >= 2:
                return -1

    return matched_outputs.pop() if len(matched_outputs) == 1 else -1