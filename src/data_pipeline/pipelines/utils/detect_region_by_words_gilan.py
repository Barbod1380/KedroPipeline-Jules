from .CONSTS_gilan.CONSTS_gilan import RULES, REPLACEMENT_DICT, ABBREVIATION_DICT, SPECIAL_PLACES
from .CONSTS_gilan.CONSTS_gilan import KEYWORDS_TO_CUT_BEFORE, KEYWORDS_TO_CUT_AFTER, PREFIX_TO_REMOVE
from .CONSTS_gilan.CONSTS_gilan import graph_model
from .graph_model import SequenceTree
from .load_rules import replace_phrases
from typing import List, Tuple


def preprocess_address(input_string: str) -> str:
    """Preprocess an address string."""
    input_string = input_string.replace(" عدد خیابان تهران ", " عدد_خیابان_تهران ")
    input_string = input_string.replace(" تهران نو ", " تهران_نو ")
    input_string = input_string.replace(" خیابان رشت ", " خیابان_رشت ")
    input_string = input_string.replace(" بخش مرکزی ", " بخش_مرکزی ")

    words = input_string.split()

    # Remove unwanted prefixes and suffixes
    while words and (
        words[0] in PREFIX_TO_REMOVE or (words[-1] == "کیرنده" and len(words) > 1)
    ):
        if words[0] in PREFIX_TO_REMOVE and len(words) > 1:
            words = words[1:]
        elif words[-1] == "کیرنده":
            words = words[:-2] if len(words) > 1 else words[:-1]
        else:
            break

    input_string = " ".join(words).strip()
    if not input_string:
        return ""

    input_string = f" {input_string} "

    # Apply abbreviations
    for abbr, full in ABBREVIATION_DICT.items():
        input_string = input_string.replace(abbr, full)

    # Cut before keywords
    for keyword in KEYWORDS_TO_CUT_BEFORE:
        idx = input_string.find(keyword)
        if idx != -1:
            input_string = input_string[idx + len(keyword) :]

    # Cut after keywords
    for keyword in KEYWORDS_TO_CUT_AFTER:
        idx = input_string.find(keyword)
        if idx != -1:
            input_string = input_string[:idx]

    # Remove prefixes
    for remove_str in [" شهید ", " سید "]:
        input_string = input_string.replace(remove_str, " ")

    input_string = input_string.replace("کوچه ", "کوچه_")
    input_string = input_string.replace("خیابان ", "خیابان_")
    input_string = input_string.replace("شهرک ", "شهرک_")
    input_string = input_string.replace("بلوار ", "بلوار_")
    input_string = input_string.replace("میدان ", "میدان_")
    input_string = input_string.replace("چهارراه ", "چهارراه_")
    input_string = input_string.replace("سازمان ", "سازمان_")
    input_string = input_string.replace("محله ", "محله_")
    input_string = input_string.replace("فلکه ", "فلکه_")

    words = input_string.split()
    # Repeat prefix removal after processing
    while words and (
        words[0] in PREFIX_TO_REMOVE or (words[-1] == "کیرنده" and len(words) > 1)
    ):
        if words[0] in PREFIX_TO_REMOVE and len(words) > 1:
            words = words[1:]
        elif words[-1] == "کیرنده":
            words = words[:-2] if len(words) > 1 else words[:-1]
        else:
            break

    return " ".join(words).strip()


def predict_by_graph(input_string: str, model: "SequenceTree") -> int:
    tokens = input_string.split()
    region = model.predict(tokens)["region"]
    return region


def predict_by_rule(input_string: str, rules=List[Tuple]) -> int:
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


def predict_by_specials(input_string: str, special_places: dict[str, int]) -> int:
    """
    Return the mapped value of the first special substring found in the input string.

    Args:
        input_string (str): The text to search within.
        special_places (dict[str, int]): A mapping of special substrings to
            integer labels or prediction values.

    Returns:
        int: The value associated with the first matching substring.
            Returns -1 if no special substring is found.
    """
    for special_place, region in special_places.items():
        if special_place in input_string:
            return region
    return -1


def detect_region_by_words_gilan(input_string: str, rules=None, replacements=None) -> int:
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

    input_string = preprocess_address(input_string)

    # Apply phrase replacements
    input_string = replace_phrases(input_string, replacements)

    input_string = " " + input_string.strip() + " "

    region = predict_by_specials(input_string, SPECIAL_PLACES)
    if region == -1:
        region = predict_by_graph(input_string, graph_model)
    if region == -1:
        region = predict_by_rule(input_string, rules)
    # region = REGION_BASKET_MAP.get(region, -1)

    return region
