# Multilingual Legal Language Models

## Legal XLM Models trained on the Multilingual Legal Pile

| Model Name                                                                                          |  Layers / Units / Heads |  Vocab. | Parameters | 
|:----------------------------------------------------------------------------------------------------|------------------------:|--------:|-----------:|
| [`joelito/legal-xlm-roberta-base`](https://huggingface.co/joelito/legal-xlm-roberta-base)           |           12 / 768 / 12 |    128K |       123M | 
| [`joelito/legal-xlm-roberta-large`](https://huggingface.co/joelito/legal-xlm-roberta-large)         |          24 / 1024 / 16 |    128K |       355M | 
| [`joelito/legal-xlm-longformer-base`](https://huggingface.co/joelito/legal-xlm-longformer-base)     |           12 / 768 / 12 |    128K |       134M | 
| [`joelito/legal-swiss-roberta-base`](https://huggingface.co/joelito/legal-swiss-roberta-base)       |           12 / 768 / 12 |    128K |       123M | 
| [`joelito/legal-swiss-roberta-large`](https://huggingface.co/joelito/legal-swiss-roberta-large)     |          24 / 1024 / 16 |    128K |       355M | 
| [`joelito/legal-swiss-longformer-base`](https://huggingface.co/joelito/legal-swiss-longformer-base) |           12 / 768 / 12 |    128K |       134M | 

Monolingual **base** size models are available for the following
languages:
[Bulgarian](https://huggingface.co/joelito/legal-bulgarian-roberta-base),
[Czech](https://huggingface.co/joelito/legal-czech-roberta-base),
[Danish](https://huggingface.co/joelito/legal-danish-roberta-base),
[German](https://huggingface.co/joelito/legal-german-roberta-base),
[Greek](https://huggingface.co/joelito/legal-greek-roberta-base),
[English](https://huggingface.co/joelito/legal-english-roberta-base),
[Spanish](https://huggingface.co/joelito/legal-spanish-roberta-base),
[Estonian](https://huggingface.co/joelito/legal-estonian-roberta-base),
[Finnish](https://huggingface.co/joelito/legal-finnish-roberta-base),
[French](https://huggingface.co/joelito/legal-french-roberta-base),
[Irish](https://huggingface.co/joelito/legal-irish-roberta-base),
[Croatian](https://huggingface.co/joelito/legal-croatian-roberta-base),
[Hungarian](https://huggingface.co/joelito/legal-hungarian-roberta-base),
[Italian](https://huggingface.co/joelito/legal-italian-roberta-base),
[Lithuanian](https://huggingface.co/joelito/legal-lithuanian-roberta-base),
[Latvian](https://huggingface.co/joelito/legal-latvian-roberta-base),
[Maltese](https://huggingface.co/joelito/legal-maltese-roberta-base),
[Dutch](https://huggingface.co/joelito/legal-dutch-roberta-base),
[Polish](https://huggingface.co/joelito/legal-polish-roberta-base),
[Portuguese](https://huggingface.co/joelito/legal-portuguese-roberta-base),
[Romanian](https://huggingface.co/joelito/legal-romanian-roberta-base),
[Slovak](https://huggingface.co/joelito/legal-slovak-roberta-base),
[Slovenian](https://huggingface.co/joelito/legal-slovenian-roberta-base),
[Swedish](https://huggingface.co/joelito/legal-swedish-roberta-base)

Monolingual **large** size models are available for the following
languages:
[English](https://huggingface.co/joelito/legal-english-roberta-large),
[German](https://huggingface.co/joelito/legal-german-roberta-large),
[French](https://huggingface.co/joelito/legal-french-roberta-large),
[Italian](https://huggingface.co/joelito/legal-italian-roberta-large),
[Spanish](https://huggingface.co/joelito/legal-spanish-roberta-large),
[Portuguese](https://huggingface.co/joelito/legal-portuguese-roberta-large)

## Benchmarking on LEXTREME

See https://github.com/JoelNiklaus/LEXTREME

## Benchmarking on LexGLUE

See https://arxiv.org/abs/2306.02069.

## Code Base

### Train XLM

Create a TPU VM instance with the following script:

```shell
gcloud compute tpus tpu-vm create tpu1 --zone=europe-west4-a --accelerator-type=v3-8 --version=tpu-vm-pt-1.12
```

Connect to the instance:

```shell
gcloud compute tpus tpu-vm ssh tpu1 --zone europe-west4-a
```

Set up the environment:

```shell
git clone https://github.com/JoelNiklaus/MultilingualLegalLMPretraining
cd MultilingualLegalLMPretraining
sudo bash setup_tpu_machine.sh
```

Put your huggingface token in `data/__init__.py` and in `scripts/train_mlm_tpu.sh`.

Make sure that you delete the output_dir locally and the huggingface model repo (hub_model_id) before training.

For TPU acceleration use the following script:

```shell
sudo sh train_mlm_tpu.sh
```

For GPU acceleration use the following script:

```shell
sh train_mlm_gpu.sh
```

### Evaluate XLM

```shell
sh eval_mlm_gpu.sh
```

### Modify pre-trained XLM-R

```bash
export PYTHONPATH=.
python3 src/mod_teacher_model.py --teacher_model_path xlm-roberta-base --student_model_path data/plms/legal-xlm-base
```

### Longformerize pre-trained RoBERTa LM

```bash
export PYTHONPATH=.
python3 src/longformerize_model.py --roberta_model_path data/plms/legal-xlm-base --max_length 4096 --attention_window 128
```

## Pipeline

1. Train tokenizer (Only RoBERTa needed because we convert BERT models to RoBERTa)

```shell
    export PYTHONPATH=. && python3 src/pretraining/train_tokenizer.py | tee train_tokenizer.log
```

2. Evaluate tokenizer

```shell
    export PYTHONPATH=. && python3 src/pretraining/evaluate_tokenizer.py | tee evaluate_tokenizer.log
```

3. Mod Teacher Model

```shell
    export PYTHONPATH=. NAME=legal-xlm-roberta-base SIZE=128k  && python3 src/modding/mod_teacher_model.py --teacher_model_path xlm-roberta-base --student_model_path data/plms/${NAME}_${SIZE} --output_dir data/plms/${NAME} | tee mod_teacher_model.log
```

4. Train MLM (monolingual: 500K steps) (TPUs or GPUs)

```shell
    sudo sh scripts/train_mlm_tpu.sh | tee train_mlm_tpu.log
```

or

```shell
    sh scripts/train_mlm_gpu.sh | tee train_mlm_gpu.log
```

5. Evaluate MLM

```shell
    sh scripts/eval_mlm_gpu.sh | tee eval_mlm_gpu.log
```

6. Longformerize MLM

```shell
    export PYTHONPATH=. && python3 src/modding/longformerize_model.py --roberta_model_path joelito/legal-xlm-roberta-base --longformer_model_path data/plms/legal-xlm-longformer-base | tee longformerize_model.log
```

7. Train Longformer MLM (monolingual: 50K steps) (only GPUs!)

```shell
    sh scripts/train_mlm_longformer.sh | tee train_mlm_longformer.log
```

8. Evaluate Longformer MLM

```shell
    sh scripts/eval_mlm_gpu.sh | tee eval_mlm_gpu.log
```

## Troubleshooting

If you get a PermissionError: [Errno 13] Permission denied: set the permissions to 777.

If you get strange git lfs errors, delete the huggingface model repo and the output directory
