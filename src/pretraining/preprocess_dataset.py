from datasets import load_dataset, interleave_datasets

_LANGUAGES = ['bg', 'cs', 'da', 'de', 'el', 'en', 'es', 'et', 'fi', 'fr', 'ga',
              'hr', 'hu', 'it', 'lt', 'lv', 'mt', 'nl', 'pl', 'pt', 'ro', 'sk', 'sl', 'sv']
_DOMAIN_TYPES = ['legislation', 'caselaw', 'contracts', 'other']


def preprocess_dataset(languages=None, domain_types=None, return_test_subsets=False):
    # combine datasets into a large interleaved dataset
    datasets = []
    sampling_scores = []
    # set defaults if they are not set
    if languages is None:
        languages = _LANGUAGES
    if domain_types is None:
        domain_types = _DOMAIN_TYPES
    for LANG in languages:
        for DOMAIN_TYPE in domain_types:
            try:
                dataset = load_dataset("joelito/Multi_Legal_Pile", f'{LANG}_{DOMAIN_TYPE}',
                                       split='train', streaming=True, use_auth_token=True)
                dataset = dataset.filter(lambda example: example['text'] and len(example['text']) > 0)

                print(f'Found data for `{DOMAIN_TYPE}` in language `{LANG}`.')
            except:
                print(f'There is no data for `{DOMAIN_TYPE}` in language `{LANG}`.')
                continue
            if DOMAIN_TYPE in ['caselaw', 'legislation']:
                sampling_scores.append(0.35)
            elif DOMAIN_TYPE in ['contracts', 'other']:
                sampling_scores.append(0.15)
            datasets.append(dataset)

    # normalize sampling scores
    sampling_scores = [sampling_score / sum(sampling_scores) for sampling_score in sampling_scores]
    print("Sampling Scores: ", {dataset.config_name: sr for dataset, sr in zip(datasets, sampling_scores)})

    # interleave datasets with sampling rates into a single dataset
    print("Interleaving datasets")
    multilingual_legal_dataset = interleave_datasets(datasets, probabilities=sampling_scores, seed=42,
                                                     stopping_strategy='all_exhausted')

    print("Example: ", list(multilingual_legal_dataset.take(1)))

    # split into training and evaluation subsets
    print("Splitting into training and evaluation subsets")
    multilingual_legal_dataset_splits = {}
    multilingual_legal_dataset_splits['train'] = multilingual_legal_dataset
    test_size = 10000 if len(languages) == 1 else 100000  # take less for test if we train monolingual models
    multilingual_legal_dataset_splits['test'] = multilingual_legal_dataset.take(test_size)

    print("Example: ", list(multilingual_legal_dataset.take(1)))

    if return_test_subsets:
        datasets = {}
        # split test subsets per language
        for LANG in languages:
            datasets[LANG] = multilingual_legal_dataset_splits['test']. \
                filter(lambda example: example['language'] == LANG)
        return datasets
    else:
        return multilingual_legal_dataset_splits
