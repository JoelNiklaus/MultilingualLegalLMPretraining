from datasets import load_dataset, interleave_datasets, Dataset

_LANGUAGES = ['bg', 'cs', 'da', 'de', 'el', 'en', 'es', 'et', 'fi', 'fr', 'ga',
              'hr', 'hu', 'it', 'lt', 'lv', 'mt', 'nl', 'pl', 'pt', 'ro', 'sk', 'sl', 'sv']
_DOMAIN_TYPES = ['legislation', 'caselaw', 'contracts', 'other']


def preprocess_dataset(languages=None, domain_types=None, use_interleave_datasets=True, return_test_subsets=False):
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
                if use_interleave_datasets:
                    dataset = dataset.filter(lambda example: example['text'] and len(example['text']) > 0)

                print(f'Found data for `{DOMAIN_TYPE}` in language `{LANG}`.')
                print("Example: ", list(dataset.take(1)))
            except:
                print(f'There is no data for `{DOMAIN_TYPE}` in language `{LANG}`.')
                continue
            if DOMAIN_TYPE in ['caselaw', 'legislation']:
                sampling_scores.append(0.45)  # caselaw and legislation are more important
            elif DOMAIN_TYPE in ['contracts', 'other']:  # chance of having toxic text is very low in legal text
                sampling_scores.append(0.05)  # but in the 'other' category, we are not sure about the quality
            datasets.append(dataset)

    # normalize sampling scores across languages
    sampling_scores = [sampling_score / sum(sampling_scores) for sampling_score in sampling_scores]
    print("Sampling Scores: ", {dataset.config_name: sr for dataset, sr in zip(datasets, sampling_scores)})

    if not use_interleave_datasets:
        return datasets, sampling_scores

    # interleave datasets with sampling rates into a single dataset
    print("Interleaving datasets")
    multilingual_legal_dataset = interleave_datasets(datasets, probabilities=sampling_scores, seed=42,
                                                     stopping_strategy='all_exhausted')

    # split into training and evaluation subsets
    print("Splitting into training and evaluation subsets")
    multilingual_legal_dataset_splits = {}
    test_size = 5_000 * len(languages)  # for each language 5_000
    multilingual_legal_dataset_splits['train'] = multilingual_legal_dataset.skip(test_size)
    multilingual_legal_dataset_splits['test'] = multilingual_legal_dataset.take(test_size)

    if return_test_subsets:
        # Convert to a normal dataset to prevent wierd issues with references with the iterable datasets
        map_style_test_dataset = Dataset.from_list(list(multilingual_legal_dataset_splits['test']))
        test_datasets = {}
        # split test subsets per language
        for LANG in languages:
            test_datasets[LANG] = map_style_test_dataset.filter(lambda x: x['language'] == LANG)
        return test_datasets
    else:
        return multilingual_legal_dataset_splits
