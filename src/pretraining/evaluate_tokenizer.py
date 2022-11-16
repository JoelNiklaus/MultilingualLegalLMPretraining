from pathlib import Path

from transformers import AutoTokenizer
import numpy as np
from preprocess_dataset import preprocess_dataset
from src.pretraining.tokenizer_utils import get_vocab_tok_folder, get_comparison_tokenizer

DATASET_SIZE = 1000  # increase to get a better estimate, decrease to get a faster estimate


def evaluate_tokenizer(vocab_size=64_000, languages=None, domain_types=None):
    """
    Compare different tokenizers
    :return:
    """
    print("Preparing dataset")
    # preprocess multilingual legal dataset
    test_datasets = preprocess_dataset(languages=languages, domain_types=domain_types, return_test_subsets=True)

    # Custom Tokenizer
    vocab_tok_folder = get_vocab_tok_folder(languages, vocab_size)
    if not Path(vocab_tok_folder).exists():
        print(f"Custom tokenizer not found for language '{languages}'. Please train it first.")
    else:
        print_fragmentation_per_language(test_datasets, vocab_tok_folder)

        # XLM-RoBERTa Tokenizer
        comparison_tokenizer = get_comparison_tokenizer(languages)
        print_fragmentation_per_language(test_datasets, comparison_tokenizer)


def print_fragmentation_per_language(test_datasets, tokenizer_name):
    print(f"Loading tokenizer {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    fr_text = ''
    for LANG, dataset in test_datasets.items():
        # show_examples(dataset, tokenizer)
        print(f"Calculating fragmentation ratio (tokens/words) for language `{LANG}`")

        def calc_fragmentation(batch):
            texts = batch['text']
            batch['tokens'] = tokenizer(texts)['input_ids']
            batch['words'] = [text.split() for text in texts]  # whitespace split

            fragmentation_ratios = []
            for tokens, words in zip(batch['tokens'], batch['words']):
                fragmentation_ratios.append(len(tokens) / len(words))
            return {'fragmentation_ratio': fragmentation_ratios}

        dataset = dataset.map(calc_fragmentation, batched=True, batch_size=1000)

        fr_text += f'{LANG}: {np.mean(dataset["fragmentation_ratio"]):.2f}\t'
        print(fr_text)
    print(f'{tokenizer_name} Tokenizer (vocab size: {tokenizer.vocab_size}) '
          f'fragmentation ratios (tokens / words): {fr_text}')


if __name__ == "__main__":
    """
    Run with 
    export PYTHONPATH=. && python src/pretraining/evaluate_tokenizer.py | tee evaluate_tokenizer.log
    """
    vocab_sizes = [32000, 64000, 128000]
    # vocab_sizes = [32000]
    languages = [['de'], ['fr'], ['it'], ['es'], ['pt'], None]  # None is for all languages
    # languages = [['de', 'fr']]
    for language in languages:
        for vocab_size in vocab_sizes:
            evaluate_tokenizer(vocab_size=vocab_size, languages=language)

"""
xlm-roberta-base Tokenizer (vocab size: 250002) fragmentation ratios (tokens / words):                                                          bg: 1.73 cs: 2.01        da: 1.97        de: 1.85        el: 2.05        en: 1.57        es: 1.53       et: 2.18        fi: 2.31        fr: 1.71        ga: 1.97        hr: 1.86        hu: 2.17        it: 1.69        lt: 2.08        lv: 2.10        mt: 3.19        nl: 1.82      pl: 1.95 pt: 1.61        ro: 1.76        sk: 2.00        sl: 1.86        sv: 1.87
/home/duser/MultilingualLegalLMPretraining/data/plms/legal-xlm-base_32k Tokenizer (vocab size: 32000) fragmentation ratios (tokens / words):    bg: 1.90   cs: 1.80        da: 1.88        de: 1.85        el: 1.87       en: 1.61 es: 1.57        et: 2.07        fi: 2.19        fr: 1.68        ga: 1.84        hr: 1.86        hu: 1.79        it: 1.71        lt: 1.94        lv: 1.92        mt: 2.31        nl: 1.59        pl: 1.90       pt: 1.61 ro: 1.72        sk: 1.88        sl: 1.90        sv: 1.78
/home/duser/MultilingualLegalLMPretraining/data/plms/legal-xlm-base_64k Tokenizer (vocab size: 64000) fragmentation ratios (tokens / words):    bg: 1.68   cs: 1.63        da: 1.72        de: 1.67        el: 1.66       en: 1.49 es: 1.46        et: 1.85        fi: 1.93        fr: 1.56        ga: 1.63        hr: 1.69        hu: 1.68        it: 1.57        lt: 1.74        lv: 1.73        mt: 2.13        nl: 1.49        pl: 1.69       pt: 1.49 ro: 1.58        sk: 1.73        sl: 1.73        sv: 1.61
/home/duser/MultilingualLegalLMPretraining/data/plms/legal-xlm-base_128k Tokenizer (vocab size: 128000) fragmentation ratios (tokens / words):  bg: 1.52 cs: 1.51        da: 1.63        de: 1.54        el: 1.51       en: 1.42 es: 1.38        et: 1.68        fi: 1.74        fr: 1.48        ga: 1.48        hr: 1.56        hu: 1.60        it: 1.49        lt: 1.60        lv: 1.58        mt: 2.01        nl: 1.43        pl: 1.54       pt: 1.41 ro: 1.48        sk: 1.60        sl: 1.60        sv: 1.50
"""
