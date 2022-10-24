from tokenizers import models, normalizers, pre_tokenizers, decoders, processors, trainers
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast, AutoTokenizer
import os
from data import DATA_DIR
from preprocess_dataset import preprocess_dataset
CUSTOM_TOK_FOLDER = os.path.join(DATA_DIR, 'plms', 'legal-xlm-base')


def batch_iterator(dataset):
    count = 0
    for example in iter(dataset):
        count += 1
        if count >= 10e6:
            break
        yield example['text']
    yield 'End'


def main(vocab_size=64000):

    # configure tokenizer
    backend_tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    backend_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    backend_tokenizer.decoder = decoders.ByteLevel()
    backend_tokenizer.post_processor = processors.RobertaProcessing(sep=("</s>", 2), cls=("<s>", 1),
                                                                    add_prefix_space=True, trim_offsets=True)

    # init tokenizer trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"],
        show_progress=True
    )

    # preprocess multilingual legal dataset
    multilingual_legal_dataset = preprocess_dataset()

    # keep the training subset only for training the token
    dataset = multilingual_legal_dataset['train']

    # train tokenizer
    backend_tokenizer.train_from_iterator(trainer=trainer, iterator=batch_iterator(dataset), length=int(10e6))

    # save tokenizer
    new_roberta_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend_tokenizer,
        model_max_length=512,
        # padding_side="Set me if you want",
        # truncation_side="Set me if you want",
        # model_input_names="Set me if you want",
        bos_token='<s>',
        eos_token='</s>',
        unk_token='<unk>',
        sep_token='</s>',
        pad_token='<pad>',
        cls_token='<s>',
        mask_token='<mask>',
    )

    new_roberta_tokenizer.save_pretrained(CUSTOM_TOK_FOLDER)
    tokenizer = AutoTokenizer.from_pretrained(CUSTOM_TOK_FOLDER)

    print(f'Trained BPE tokenizer with  a vocabulary of {vocab_size} sub-words successfully!')

    test_samples = dataset.take(500)
    for example in test_samples:
        text = ' '.join(example['text'].split()[:500])
        print(text)
        print('-' * 150)
        print(tokenizer.tokenize(text))
        print('-' * 150)


if __name__ == "__main__":
    main()
