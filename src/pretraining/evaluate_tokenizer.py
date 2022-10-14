from transformers import AutoTokenizer
import os
from data import DATA_DIR
import numpy as np
from preprocess_dataset import preprocess_dataset
CUSTOM_TOK_FOLDER = os.path.join(DATA_DIR, 'plms', 'legal-xlm-base')


def evaluate_tokenizers():

    # preprocess multilingual legal dataset
    multilingual_legal_dataset_test_subsets = preprocess_dataset(return_test_subsets=True)

    # Custom Tokenizera
    for tokenizer_config in ['-32k', '-64k', '-128k']:
        for LANG in multilingual_legal_dataset_test_subsets:
            tokenizer = AutoTokenizer.from_pretrained(CUSTOM_TOK_FOLDER + tokenizer_config)
            fragmentation_ratio = []
            fr_text = ''
            for document in multilingual_legal_dataset_test_subsets[LANG]:
                fragmentation_ratio.append(len(tokenizer.tokenize(document['text'])) / len(document['text'].split()))
                fr_text += f'{LANG}: {np.mean(fragmentation_ratio):.2f}\t'
        print(f'Custom-Tokenizer{tokenizer_config}: {fr_text}')

    # XLM-RoBERTa Tokenizer
    for LANG in multilingual_legal_dataset_test_subsets:
        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
        fragmentation_ratio = []
        fr_text = ''
        for document in multilingual_legal_dataset_test_subsets[LANG]:
            fragmentation_ratio.append(len(tokenizer.tokenize(document['text'])) / len(document['text'].split()))
            fr_text += f'{LANG}: {np.mean(fragmentation_ratio):.2f}\t'
    print(f'XLM-RoBERTa-Tokenizer: {fr_text}')


if __name__ == "__main__":
    evaluate_tokenizers()