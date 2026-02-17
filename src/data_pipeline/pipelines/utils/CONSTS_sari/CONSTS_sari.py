import os
import json
import pandas as pd
from ..load_rules import load_rules, process_phrases_to_dict_to_handle_underlines
from ..graph_model import SequenceTree

CONSTS_DIR = os.path.dirname(os.path.realpath(__file__))

VALID_REGIONS = pd.read_excel(
    os.path.join(CONSTS_DIR, "valid_regions.xlsx")
).values.flatten()

POSTCODE_CORRECTION = pd.read_excel(
    os.path.join(CONSTS_DIR, "postcode_correction.xlsx"),
    header=None,
    names=["postcode", "region"],
    index_col="postcode",
)


REGION_BASKET_MAP = {
    461: 59,
    471: 60,
}


PATTERN_PATH = os.path.join(CONSTS_DIR, "patterns.xlsx")
CORRECT_WORDS_SET = set(
    pd.read_excel(PATTERN_PATH, header=None)
    .iloc[:, 0]
    .str.split("[+_-]")
    .explode()
    .str.replace(".", "", regex=False)
    .to_list()
)
VALID_SPELL_CORRECTION_DICT = json.load(
    open(os.path.join(CONSTS_DIR, "spell_correction_dict.json"), encoding="utf-8")
)

RULES, df = load_rules(PATTERN_PATH)
REPLACEMENT_DICT = process_phrases_to_dict_to_handle_underlines(df)

ABBREVIATION_DICT = {
    " خ ": " خیابان ",
    " م ": " میدان ",
    " پ ": " پلاک",
    " ک ": " کوچه ",
}

KEYWORDS_TO_CUT_BEFORE = (
    " شهرستان ",
    " استان ",
    " روستا ",
    " منطقه ",
    " خیابان_ساری ",
    " نشانی ",
    " کیرنده ",
    " خریدار ",
    " ادرس ",
)

KEYWORDS_TO_CUT_AFTER = (" طبقه ", " ساختمان ", " مجتمع ", " برج ", " پلاک ")

PREFIX_TO_REMOVE = (
    "مقصد",
    "مازندران",
    "ساری",
    "خیابان_ساری",
    "بخش_مرکزی",
    "عدد",
    "شهر",
    "شهرستان",
    "استان",
    "روستا",
    "منطقه",
    "نشانی",
    "کیرنده",
    "خریدار",
    "ادرس",
    "کوفت",
    "اول",
)

graph_model = SequenceTree.load(os.path.join(CONSTS_DIR, "graph_rules.json"), "json")

CITY_NAME_REPLACEMENTS = [
    (" عدد خیابان تهران ", " عدد_خیابان_تهران "),
    (" تهران نو ", " تهران_نو "),
    (" خیابان ساری ", " خیابان_ساری "),
    (" بخش مرکزی ", " بخش_مرکزی "),
]

SUFFIX_TO_REMOVE = ("کیرنده",)