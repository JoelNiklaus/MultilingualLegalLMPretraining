import json
import tqdm
import os
import re
from multiprocessing import Pool
from tokenizers import normalizers
from data import DATA_DIR
from src.pretraining.preprocess_dataset import preprocess_dataset

MAX_SEQ_LENGTH = 500
GOAL_SEQUENCES_NUMBER = 6e8


def write_samples(dataset_number):
    custom_normalizer = normalizers.NFKD()
    dataset, dataset_goal_number, dataset_name = dataset_number
    total_count = 0
    temp_count = 0
    all_samples = 0
    file_number = 1
    out_file = open(
        os.path.join(DATA_DIR, 'mlm_dataset', 'chunks_512', f'{dataset_name}_train_{file_number}.jsonl'),
        'w', encoding='utf8')
    print(f'Processing for dataset {dataset_name} started!')
    # Read each document
    for sample in tqdm.tqdm(dataset):
        try:
            # Normalize the document
            text = custom_normalizer.normalize_str(sample['text'])
            # Replace multiple newline and whitespaces
            text = re.sub(r'(\n )+', r'\n ', re.sub(r'( *[\n\r]+ *)+', r'\n ', re.sub(r'[\t ]+', r' ', text)))
            # Split document in whitespaces
            ws_tokens = text.split(' ')
            prev_idx = 0
            # Chunk into sequences up to 500 tokens
            for idx in range(MAX_SEQ_LENGTH, len(ws_tokens) + MAX_SEQ_LENGTH, MAX_SEQ_LENGTH):
                if total_count >= dataset_goal_number:
                    out_file.close()
                    print(f'Processing for dataset {dataset_name} finished early with {total_count}/{all_samples}!')
                    return
                all_samples += 1
                if temp_count > 1000000:
                    out_file.close()
                    file_number += 1
                    temp_count = 0
                    out_file = open(os.path.join(DATA_DIR, 'mlm_dataset', 'chunks_512',
                                                 f'{dataset_name}_train_{file_number}.jsonl'), 'w',
                                    encoding='utf8')
                # Join 500 tokens in a sequence
                sample_text = ' '.join(ws_tokens[prev_idx:idx])
                prev_idx = idx
                # Compute percentage of alphabetical characters in relation to full sequence length
                alpha_text = re.sub(r'[^a-zA-Z ]', '', sample_text)
                alpha_percent = len(alpha_text) / len(sample_text)
                # Compute total chunk length
                text_length = len(sample_text.split())
                # Ignore sequences with more than 30% numbers or short sequences (less than 64 tokens)
                if alpha_percent > 0.7 and text_length > 64:
                    out_file.write(json.dumps({"text": sample_text, "language": sample["language"],
                                               "type": sample["type"], "jurisdiction": sample["jurisdiction"]}) + '\n')
                    total_count += 1
                    temp_count += 1
        except:
            continue

    try:
        out_file.close()
    except:
        print(f'Processing for dataset {dataset_name} finished with {total_count}/{all_samples}!')
        return
    print(f'Processing for dataset {dataset_name} finished with {total_count}/{all_samples}!')
    return


def split_documents():
    ''' set default hyperparams in default_hyperparams.py '''

    # Load all datasets across languages and types
    datasets, sampling_scores = preprocess_dataset(use_interleave_datasets=False)
    # Shuffle datasets to pick and write up to N entries (GOAL_SEQUENCES_NUMBER * sampling_score) that are going to be used.
    datasets = [(dataset.shuffle(seed=42, buffer_size=100_000), int(GOAL_SEQUENCES_NUMBER * sampling_score), dataset.config_name)
                for dataset, sampling_score in zip(datasets, sampling_scores)]

    # Launch pool to preprocess all datasets in parallel
    p = Pool(len(datasets))
    p.map(write_samples, datasets)
    p.close()


if __name__ == '__main__':
    split_documents()
