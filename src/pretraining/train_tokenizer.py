import shutil

from tokenizers import models, pre_tokenizers, decoders, processors, trainers
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast, AutoTokenizer
import os
import re
from preprocess_dataset import preprocess_dataset
from src.pretraining.tokenizer_utils import CUSTOM_TOK_FOLDER, get_vocab_tok_folder

max_examples = int(1e7)  # 1e7


def batch_iterator(dataset):
    count = 0
    for example in iter(dataset):
        count += 1
        if count >= max_examples:
            break
        # normalize documents by removing bad information (multiple new lines, tabs, whitespace, etc.)
        yield re.sub(r'\n{2,}', r'\n', re.sub(r'(\t| | ){2,}', r' ', example['text']))
    yield 'End'


def train_tokenizers(vocab_size=64_000, languages=None, domain_types=None):
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

    # save and load tokenizer
    vocab_tok_folder = get_vocab_tok_folder(languages, vocab_size)
    new_roberta_tokenizer.save_pretrained(vocab_tok_folder)
    shutil.copy(os.path.join(CUSTOM_TOK_FOLDER, "config.json"), os.path.join(vocab_tok_folder, "config.json"))
    tokenizer = AutoTokenizer.from_pretrained(vocab_tok_folder)

    print(f'Trained BPE tokenizer with  a vocabulary of {vocab_size} sub-words successfully!')

    test_samples = dataset.take(5)
    for example in test_samples:
        text = ' '.join(example['text'].split()[:500])
        print(text)
        print('-' * 150)
        print(tokenizer.tokenize(text))
        print('-' * 150)


if __name__ == "__main__":
    """
    Run with 
    export PYTHONPATH=. && python src/pretraining/train_tokenizer.py | tee train_tokenizer.log
    """
    vocab_sizes = [32000, 64000, 128000]
    languages = [None, ['de'], ['fr'], ['it']]  # None is for all languages
    for language in languages:
        for vocab_size in vocab_sizes:
            train_tokenizers(vocab_size=vocab_size, languages=language)
