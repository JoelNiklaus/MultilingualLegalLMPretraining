import os
import re

from data import DATA_DIR

PLM_FOLDER = os.path.join(DATA_DIR, 'plms')
CUSTOM_TOK_FOLDER = os.path.join(PLM_FOLDER, 'legal-xlm-base')

isocode2lang = {
    'en': 'english',
    'fr': 'french',
    'de': 'german',
    'it': 'italian',
    'es': 'spanish',
    'pt': 'portuguese',
}

comparison_tokenizers = {
    'en': 'bert-base-uncased',
    'fr': 'camembert-base',
    'de': 'bert-base-german-cased',
    'it': 'dbmdz/bert-base-italian-xxl-cased',
    'es': 'dccuchile/bert-base-spanish-wwm-cased',
    'pt': 'neuralmind/bert-base-portuguese-cased',
}


def get_vocab_tok_folder(languages, vocab_size):
    lang = f"{isocode2lang[languages[0]]}-bert" if languages else 'xlm'
    lang_folder = os.path.join(PLM_FOLDER, f"legal-{lang}-base")
    return lang_folder + f'_{vocab_size // 1000}k'


def normalize_text(text):
    # normalize documents by removing bad information (multiple new lines, tabs, whitespace, etc.)
    return re.sub(r'\n+ ', '\n', re.sub(r'[\t  ]+', ' ', text))
