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

print(os.path.join(CONSTS_DIR, "postcode_correction.xlsx"))
print(f"POSTCODE CORRECTION SHAPE: {POSTCODE_CORRECTION.shape}")


REGION_BASKET_MAP = {
    413: 293,
    414: 294,
    415: 295,
    416: 296,
    417: 297,
    418: 298,
    419: 299,
    410: -1,
    431: 251,
    441: 252,
}

GARBAGE_POSTCODES = (
    "1318711111",
    "1111111111",
    "1311111111",
    "1318915513",
    "1495613311",
    "1816145825",
    "1816145835",
    "1816145875",
)

SPECIAL_POSTCODES = {
    "1114943311": 110,
    "1136844414": 110,  # sakhteman bank maskan
    "1144945633": 110,
    "1149943381": 110,
    "1149943141": 110,  # sazman barnameh va boodjeh
    "131851663": 130,  # bank resalat
    "1517813511": 199,  # sakhteman solh helal ahmar
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
    " خیابان_رشت ",
    " خیابان_کزمانشاه ",
    " نشانی ",
    " کیرنده ",
    " خریدار ",
    " ادرس ",
    " عدد_خیابان_تهران ",
)
KEYWORDS_TO_CUT_AFTER = (" طبقه ", " ساختمان ", " مجتمع ", " برج ", " پلاک ")

PREFIX_TO_REMOVE = (
    "کیلان"
    "مازندران",
    "مقصد",
    "کیانشهر",
    "رشت",
    "کرمانشاه",
    "تهران",
    "عدد_خیابان_تهران",
    "خیابان_رشت",
    "خیابان_کرمانشاه",
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

CITY_NAME_REPLACEMENTS = []

SUFFIX_TO_REMOVE = ("کیرنده",)