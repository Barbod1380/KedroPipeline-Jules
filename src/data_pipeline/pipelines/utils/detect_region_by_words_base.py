from .graph_model import SequenceTree
from .load_rules import replace_phrases
from typing import List, Tuple, Any

def preprocess_address(
    input_string: str,
    city_name_replacements: List[Tuple[str, str]],
    prefix_to_remove: Any,
    suffix_to_remove: Any,
    abbreviation_dict: dict,
    keywords_to_cut_before: Any,
    keywords_to_cut_after: Any
) -> str:
    # City specific replacements
    for old, new in city_name_replacements:
        input_string = input_string.replace(old, new)

    words = input_string.split()

    # Remove unwanted prefixes and suffixes
    # Using suffix_to_remove for the end check
    while words and (
        words[0] in prefix_to_remove or (len(words) > 1 and words[-1] in suffix_to_remove)
    ):
        if words[0] in prefix_to_remove and len(words) > 1:
            words = words[1:]
        elif words[-1] in suffix_to_remove:
            words = words[:-2] if len(words) > 1 else words[:-1]
        else:
            break

    input_string = " ".join(words).strip()
    if not input_string:
        return ""

    input_string = f" {input_string} "

    # Apply abbreviations
    for abbr, full in abbreviation_dict.items():
        input_string = input_string.replace(abbr, full)

    # Cut before keywords
    for keyword in keywords_to_cut_before:
        idx = input_string.find(keyword)
        if idx != -1:
            input_string = input_string[idx + len(keyword) :]

    # Cut after keywords
    for keyword in keywords_to_cut_after:
        idx = input_string.find(keyword)
        if idx != -1:
            input_string = input_string[:idx]

    # Remove prefixes
    for remove_str in [" شهید ", " سید "]:
        input_string = input_string.replace(remove_str, " ")

    # Common replacements
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
        words[0] in prefix_to_remove or (len(words) > 1 and words[-1] in suffix_to_remove)
    ):
        if words[0] in prefix_to_remove and len(words) > 1:
            words = words[1:]
        elif words[-1] in suffix_to_remove:
            words = words[:-2] if len(words) > 1 else words[:-1]
        else:
            break

    return " ".join(words).strip()


def predict_by_graph(input_string: str, model: "SequenceTree") -> int:
    tokens = input_string.split()
    region = model.predict(tokens)["region"]
    return region


def predict_by_rule(input_string: str, rules: List[Tuple]) -> int:
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
    for special_place, region in special_places.items():
        if special_place in input_string:
            return region
    return -1


def detect_region_by_words_base(
    input_string: str,
    consts_module: Any,
    rules=None,
    replacements=None
) -> int:

    # Extract constants
    if rules is None:
        rules = getattr(consts_module, 'RULES')
    if replacements is None:
        replacements = getattr(consts_module, 'REPLACEMENT_DICT')

    city_name_replacements = getattr(consts_module, 'CITY_NAME_REPLACEMENTS', [])
    prefix_to_remove = getattr(consts_module, 'PREFIX_TO_REMOVE', [])
    suffix_to_remove = getattr(consts_module, 'SUFFIX_TO_REMOVE', [])
    abbreviation_dict = getattr(consts_module, 'ABBREVIATION_DICT', {})
    keywords_to_cut_before = getattr(consts_module, 'KEYWORDS_TO_CUT_BEFORE', [])
    keywords_to_cut_after = getattr(consts_module, 'KEYWORDS_TO_CUT_AFTER', [])
    graph_model = getattr(consts_module, 'graph_model')
    special_places = getattr(consts_module, 'SPECIAL_PLACES', None)


    # Add spaces around the remaining string
    input_string = " " + input_string.strip() + " "

    if " فرستنده " in input_string:
        return -1

    input_string = preprocess_address(
        input_string,
        city_name_replacements,
        prefix_to_remove,
        suffix_to_remove,
        abbreviation_dict,
        keywords_to_cut_before,
        keywords_to_cut_after
    )

    # Apply phrase replacements
    input_string = replace_phrases(input_string, replacements)

    input_string = " " + input_string.strip() + " "

    region = -1
    if special_places:
        region = predict_by_specials(input_string, special_places)

    if region == -1:
        region = predict_by_graph(input_string, graph_model)

    if region == -1:
        region = predict_by_rule(input_string, rules)

    return region
