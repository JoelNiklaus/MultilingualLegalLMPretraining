import glob
import json
import tqdm
import os
import re
from multiprocessing import Pool

from tokenizers import normalizers
from data import DATA_DIR
from src.pretraining.preprocess_dataset import preprocess_dataset

MAX_SEQ_LENGTH = 500
GOAL_SEQUENCES_NUMBER = float('inf')  # set to a lower number to limit the maximum number of examples
VALIDATION_SIZE = 10_000  # ~10MB per configuration ==> some low-resource configs will only have a validation file

chunk_dir = os.path.join(DATA_DIR, 'mlm_dataset', 'chunks_512')


def write_samples(dataset_number):
    custom_normalizer = normalizers.NFKD()
    dataset, dataset_goal_number, dataset_name = dataset_number
    total_count, temp_count, all_samples = 0, 0, 0
    file_number = 1
    out_file = open_file(dataset_name, file_number, "validation")  # we save the first examples to the validation set
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
                if "validation" in out_file.name and temp_count > VALIDATION_SIZE:
                    # if we are saving to eval and we have enough samples in the eval set, switch to train
                    out_file.close()
                    temp_count = 0
                    total_count = 0
                    out_file = open_file(dataset_name, file_number, "train")
                # on average approx. 10GB per file, compresses (with xz) to around ~2-3GB (xz: ~75% compression ratio)
                if "train" in out_file.name and temp_count > 5_000_000:
                    # if we are saving to train, and we reached the max size per file, switch to the next file
                    out_file.close()
                    file_number += 1
                    temp_count = 0
                    out_file = open_file(dataset_name, file_number, "train")
                # Join 500 tokens in a sequence
                sample_text = ' '.join(ws_tokens[prev_idx:idx])
                prev_idx = idx
                # Compute percentage of alphabetical characters in relation to full sequence length
                punctuation = '!\"#$%&\'()*+,\-\./:;<=>?@\[\\\]\^_`{\|}~'
                alpha_text = re.sub(rf'[{punctuation}\d]', '', sample_text)  # remove numbers and punctuation
                alpha_percent = len(alpha_text) / len(sample_text)
                # Compute total chunk length
                text_length = len(sample_text.split())
                # Ignore sequences with more than 30% numbers or short sequences (less than 64 tokens)
                if alpha_percent > 0.7 and text_length > 64:
                    out_file.write(json.dumps({"text": sample_text, "language": sample["language"],
                                               "type": sample["type"], "jurisdiction": sample["jurisdiction"]}) + '\n')
                    total_count += 1
                    temp_count += 1
                all_samples += 1
        except:
            continue

    try:
        out_file.close()
    except:
        pass

    print(f'Processing for dataset {dataset_name} finished with {total_count}/{all_samples}!')
    return


def open_file(dataset_name, file_number, split):
    return open(os.path.join(chunk_dir, f'{dataset_name}_{split}_{file_number}.jsonl'), 'w', encoding='utf8')


def split_documents(use_sampling_scores=False):
    ''' set default hyperparams in default_hyperparams.py '''
    # Load all datasets across languages and types
    datasets, sampling_scores = preprocess_dataset(use_interleave_datasets=False)
    # Shuffle datasets to pick and write up to N entries (GOAL_SEQUENCES_NUMBER * sampling_score) that are going to be used.
    datasets = [(dataset.shuffle(seed=42, buffer_size=10_000),
                 int(GOAL_SEQUENCES_NUMBER * sampling_score) if use_sampling_scores else GOAL_SEQUENCES_NUMBER,
                 dataset.config_name)
                for dataset, sampling_score in zip(datasets, sampling_scores)]

    # Launch pool to preprocess all datasets in parallel
    p = Pool(len(datasets))
    p.map(write_samples, datasets)
    p.close()

    # Compress datasets
    print(f"Compressing datasets at {chunk_dir}")
    # Do this at the end because we use multithreading
    for path in glob.glob(os.path.join(chunk_dir, '*.jsonl')):
        os.system(f'xz -zkf -T0 {path}')  # -TO to use multithreading
        os.system(f'rm {path}')  # remove uncompressed file to save space


if __name__ == '__main__':
    """
    Run with 
    export PYTHONPATH=. && python src/data_preprocessing/prepare_mlm_dataset.py | tee prepare_mlm_dataset.log
    """
    split_documents()
