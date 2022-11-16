import copy
from pathlib import Path

from transformers import AutoTokenizer
import numpy as np
from preprocess_dataset import preprocess_dataset
from src.pretraining.tokenizer_utils import get_vocab_tok_folder, comparison_tokenizers, show_examples

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
        comparison_tokenizer = comparison_tokenizers[languages[0]] \
            if languages or len(languages) == 1 else 'xlm-roberta-base'
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
