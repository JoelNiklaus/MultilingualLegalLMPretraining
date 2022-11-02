from pathlib import Path

from transformers import AutoTokenizer
import numpy as np
from preprocess_dataset import preprocess_dataset
from src.pretraining.tokenizer_utils import get_vocab_tok_folder, comparison_tokenizers


def evaluate_tokenizers(vocab_size=64_000, languages=None, domain_types=None):
    """
    Compare different tokenizers
    :return:
    """
    print("Preparing dataset")
    # preprocess multilingual legal dataset
    multilingual_legal_dataset_test_subsets = preprocess_dataset(languages=languages, domain_types=domain_types,
                                                                 return_test_subsets=True)

    # Custom Tokenizer
    vocab_tok_folder = get_vocab_tok_folder(languages, vocab_size)
    if Path(vocab_tok_folder).exists():
        print_fragmentation_per_language(multilingual_legal_dataset_test_subsets, vocab_tok_folder)

        # XLM-RoBERTa Tokenizer
        comparison_tokenizer = comparison_tokenizers[languages[0]] if languages else 'xlm-roberta-base'
        print_fragmentation_per_language(multilingual_legal_dataset_test_subsets, comparison_tokenizer)


def print_fragmentation_per_language(multilingual_legal_dataset_test_subsets, tokenizer_name):
    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    fr_text = ''
    for LANG in multilingual_legal_dataset_test_subsets:
        print(f"Calculating fragmentation ratio (tokens/words) for language `{LANG}`")

        def calc_fragmentation(batch):
            texts = batch['text']
            batch['tokens'] = tokenizer(texts)['input_ids']
            batch['words'] = [text.split() for text in texts]  # whitespace split

            fragmentation_ratios = []
            for tokens, words in zip(batch['tokens'], batch['words']):
                fragmentation_ratios.append(len(tokens) / len(words))
            return {'fragmentation_ratio': fragmentation_ratios}

        multilingual_legal_dataset_test_subsets[LANG] = multilingual_legal_dataset_test_subsets[LANG].map(
            calc_fragmentation, batched=True, batch_size=1000
        )

        fragmentation_ratios = [example['fragmentation_ratio'] for example in
                                list(multilingual_legal_dataset_test_subsets[LANG])]
        fr_text += f'{LANG}: {np.mean(fragmentation_ratios):.2f}\t'
    print(f'{tokenizer_name} Tokenizer (vocab size: {tokenizer.vocab_size}) '
          f'fragmentation ratios (tokens / words): {fr_text}')


if __name__ == "__main__":
    """
    Run with 
    export PYTHONPATH=. && python src/pretraining/evaluate_tokenizer.py | tee evaluate_tokenizer.log
    """
    vocab_sizes = [32000, 64000, 128000]
    # vocab_sizes = [32000]
    languages = [None, ['de'], ['fr'], ['it']]  # None is for all languages
    # languages = [['it']]
    for language in languages:
        for vocab_size in vocab_sizes:
            evaluate_tokenizers(vocab_size=vocab_size, languages=language)
