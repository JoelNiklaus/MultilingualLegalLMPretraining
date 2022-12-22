import os
import re

from data import DATA_DIR

PLM_FOLDER = os.path.join(DATA_DIR, 'plms')
CUSTOM_TOK_FOLDER = os.path.join(PLM_FOLDER, 'legal-xlm-base')

isocode2lang = {
    'bg': 'bulgarian',
    'cs': 'czech',
    'da': 'danish',
    'de': 'german',
    'el': 'greek',
    'en': 'english',
    'es': 'spanish',
    'et': 'estonian',
    'fi': 'finnish',
    'fr': 'french',
    'ga': 'irish',
    'hr': 'croatian',
    'hu': 'hungarian',
    'it': 'italian',
    'lt': 'lithuanian',
    'lv': 'latvian',
    'mt': 'maltese',
    'nl': 'dutch',
    'pl': 'polish',
    'pt': 'portuguese',
    'ro': 'romanian',
    'sk': 'slovak',
    'sl': 'slovenian',
    'sv': 'swedish',
}

comparison_tokenizers = {
    "bg": "iarfmoose/roberta-base-bulgarian",
    "cs": None,
    "da": "Maltehb/danish-bert-botxo",
    "de": "dbmdz/bert-base-german-cased",  # deepset/gbert-base
    "el": "nlpaueb/bert-base-greek-uncased-v1",
    "en": "roberta-base",  # etc.
    "es": "bertin-project/bertin-roberta-base-spanish",  # PlanTL-GOB-ES/roberta-base-bne
    "et": None,
    "fi": "TurkuNLP/bert-base-finnish-cased-v1",
    "fr": "camembert-base",  # dbmdz/bert-base-french-europeana-cased
    "ga": "DCU-NLP/bert-base-irish-cased-v1",
    "hr": None,
    "hu": None,
    "it": "Musixmatch/umberto-commoncrawl-cased-v1",  # dbmdz/bert-base-italian-xxl-cased
    "lt": None,
    "lv": None,
    "mt": None,
    "nl": "bert-base-dutch-cased",
    "pl": "dkleczek/bert-base-polish-uncased-v1",
    "pt": "neuralmind/bert-base-portuguese-cased",
    "ro": "dumitrescustefan/bert-base-romanian-uncased-v1",
    "sk": "gerulata/slovakbert",
    "sl": None,
    "sv": "KB/bert-base-swedish-cased",
}


def get_comparison_tokenizer(languages):
    return comparison_tokenizers[languages[0]] if languages and len(languages) == 1 else 'xlm-roberta-base'


def get_vocab_tok_folder(languages, vocab_size):
    if languages:
        if len(languages) == 1:
            lang = f"{isocode2lang[languages[0]]}-bert"
        elif languages == ['de', 'fr', 'it']:
            lang = "swiss-bert"
        else:
            raise ValueError(f"Unsupported languages: {languages}")
    else:
        lang = "xlm"
    lang_folder = os.path.join(PLM_FOLDER, f"legal-{lang}-base")
    return lang_folder + f'_{vocab_size // 1000}k'


def normalize_text(text):
    # normalize documents by removing bad information (multiple new lines, tabs, whitespace, etc.)
    return re.sub(r'\n+ ', '\n', re.sub(r'[\t  ]+', ' ', text))


def show_examples(dataset, tokenizer, num_examples=5):
    test_samples = dataset.take(num_examples)
    for example in test_samples:
        text = ' '.join(example['text'].split()[:500])
        print(text)
        print('-' * 150)
        print(tokenizer.tokenize(text))
        print('-' * 150)
