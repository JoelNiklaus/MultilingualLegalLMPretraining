from datasets import load_dataset, interleave_datasets


def preprocess_dataset(return_test_subsets=False):
    # combine datasets into a large interleaved dataset
    datasets = []
    for LANG in ['bg', 'cs', 'da', 'de', 'el', 'en', 'es', 'et', 'fi', 'fr', 'ga',
                 'hr', 'hu', 'it', 'lt', 'lv', 'mt', 'nl', 'pl', 'pt', 'ro', 'sk', 'sl', 'sv']:
        for DOMAIN_TYPE in ['legislation', 'caselaw']:
            try:
                dataset = load_dataset("lexlms/Multi_Legal_Pile", DOMAIN_TYPE, language=LANG,
                                       split='train', streaming=True)
            except:
                print(f'There is no dataset for `{DOMAIN_TYPE}` in language `{LANG}`.')
                continue
            datasets.append(dataset)

    # interleave datasets with sampling rates into a single dataset
    multilingual_legal_dataset = interleave_datasets(datasets, probabilities=[], seed=42)

    # split into training and evaluation subsets
    multilingual_legal_dataset = multilingual_legal_dataset.train_test_split(test_size=0.05, seed=42)

    if return_test_subsets:
        datasets = {}
        # split test subsets per language
        for LANG in ['bg', 'cs', 'da', 'de', 'el', 'en', 'es', 'et', 'fi', 'fr', 'ga',
                     'hr', 'hu', 'it', 'lt', 'lv', 'mt', 'nl', 'pl', 'pt', 'ro', 'sk', 'sl', 'sv']:
            datasets[LANG] = multilingual_legal_dataset['test'].filter(lambda example: example['language'] == LANG)
            return datasets
    else:
        return multilingual_legal_dataset

