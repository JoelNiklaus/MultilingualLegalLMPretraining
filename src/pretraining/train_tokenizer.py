from tokenizers import models, pre_tokenizers, decoders, processors, trainers
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast, AutoTokenizer
import os
import re
from data import DATA_DIR
from preprocess_dataset import preprocess_dataset

CUSTOM_TOK_FOLDER = os.path.join(DATA_DIR, 'plms', 'legal-xlm-base')
max_examples = int(10e6)


def batch_iterator(dataset):
    count = 0
    for example in iter(dataset):
        count += 1
        if count >= max_examples:
            break
        # normalize documents by removing bad information (multiple new lines, tabs, whitespace, etc.)
        yield re.sub(r'\n{2,}', r'\n', re.sub(r'(\t| | ){2,}', r' ', example['text']))
    yield 'End'


def main(vocab_size=64000, languages=None, domain_types=None):
    # configure tokenizer
    backend_tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))  # WordPiece for BERT
    backend_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)  # BertPreTokenizer for BERT
    backend_tokenizer.decoder = decoders.ByteLevel()  # WordPiece for BERT
    backend_tokenizer.post_processor = processors.RobertaProcessing(sep=("</s>", 2), cls=("<s>", 1),
                                                                    add_prefix_space=True,
                                                                    trim_offsets=True)  # BertProcessing for BERT

    # init tokenizer trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"],
        show_progress=True
    )  # WordPieceTrainer for BERT

    # preprocess multilingual legal dataset
    multilingual_legal_dataset = preprocess_dataset(languages=languages, domain_types=domain_types)

    # keep the training subset only for training the tokenizer
    dataset = multilingual_legal_dataset['train']

    print(list(dataset.take(1)))

    # train tokenizer
    backend_tokenizer.train_from_iterator(trainer=trainer, iterator=batch_iterator(dataset), length=max_examples)

    # save tokenizer
    new_roberta_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend_tokenizer,
        model_max_length=512,
        # padding_side="Set me if you want",
        # truncation_side="Set me if you want",
        # model_input_names="Set me if you want",
        bos_token='<s>',  # [CLS]
        eos_token='</s>',  # [SEP]
        unk_token='<unk>',  # [UNK]
        sep_token='</s>',  # [SEP]
        pad_token='<pad>',  # [PAD]
        cls_token='<s>',  # [CLS]
        mask_token='<mask>',  # [MASK]
    )

    new_roberta_tokenizer.save_pretrained(CUSTOM_TOK_FOLDER)
    tokenizer = AutoTokenizer.from_pretrained(CUSTOM_TOK_FOLDER)

    print(f'Trained BPE tokenizer with  a vocabulary of {vocab_size} sub-words successfully!')  # TODO adapt message

    test_samples = dataset.take(500)
    for example in test_samples:
        text = ' '.join(example['text'].split()[:500])
        print(text)
        print('-' * 150)
        print(tokenizer.tokenize(text))
        print('-' * 150)


if __name__ == "__main__":
    """
    Run with 
    export PYTHONPATH=. && python src/pretraining/train_tokenizer.py
    """
    main(vocab_size=32_000, languages=['fr'])
