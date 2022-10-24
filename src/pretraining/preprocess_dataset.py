from datasets import load_dataset, interleave_datasets


def preprocess_dataset(return_test_subsets=False):
    # combine datasets into a large interleaved dataset
    datasets = []
    sampling_scores = []
    for LANG in ["bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "ga", "hr",
                 "hu", "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro", "sk", "sl", "sv"]:
        for DOMAIN_TYPE in ['legislation', 'caselaw', 'contracts', 'other']:
            try:
                dataset = load_dataset("joelito/Multi_Legal_Pile", f'{LANG}_{DOMAIN_TYPE}',
                                       split='train', streaming=True)
            except:
                print(f'There is no data for `{DOMAIN_TYPE}` in language `{LANG}`.')
                continue
            if DOMAIN_TYPE in ['caselaw', 'legislation']:
                sampling_scores.append(0.35)
            elif DOMAIN_TYPE in ['contracts', 'other']:
                sampling_scores.append(0.15)
            datasets.append(dataset)

    # normalize sampling scores
    sampling_scores = [sampling_score/sum(sampling_scores) for sampling_score in sampling_scores]
    print({dataset.config_name: sr for dataset, sr in zip(datasets, sampling_scores)})

    # interleave datasets with sampling rates into a single dataset
    multilingual_legal_dataset = interleave_datasets(datasets, probabilities=sampling_scores, seed=42)

    # split into training and evaluation subsets
    multilingual_legal_dataset_splits = {}
    multilingual_legal_dataset_splits['train'] = multilingual_legal_dataset
    multilingual_legal_dataset_splits['test'] = multilingual_legal_dataset.take(100000)

    if return_test_subsets:
        datasets = {}
        # split test subsets per language
        for LANG in ["bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "ga", "hr",
                     "hu", "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro", "sk", "sl", "sv"]:
            datasets[LANG] = multilingual_legal_dataset_splits['test'].\
                filter(lambda example: example['language'] == LANG)
            return datasets
    else:
        return multilingual_legal_dataset_splits

