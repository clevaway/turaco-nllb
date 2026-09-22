ENG_LANG = "eng_Latn"
WES_LANG = "wes_Latn"

DIRECTIONS = {
    "en-wes": {
        "source_column": "eng",
        "target_column": "wes",
        "source_lang": ENG_LANG,
        "target_lang": WES_LANG,
        "marian_target_tag": ">>wes<<",
        "byt5_prefix": "translate English to Cameroon Pidgin: ",
    },
    "wes-en": {
        "source_column": "wes",
        "target_column": "eng",
        "source_lang": WES_LANG,
        "target_lang": ENG_LANG,
        "marian_target_tag": ">>eng<<",
        "byt5_prefix": "translate Cameroon Pidgin to English: ",
    },
}

SUPPORTED_BACKENDS = {"nllb", "marian", "byt5"}

