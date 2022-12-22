# Multilingual Legal Language Models

## Legal XLM Models trained on the Multilingual Legal Pile

| Model Name                                                                                       | Layers / Units /  Heads | Vocab. | Parameters | 
|:-------------------------------------------------------------------------------------------------|------------------------:|-------:|-----------:|
| [`lexlms/legal-xlm-base`](https://huggingface.co/lexlms/legal-xlm-base)                          |           12 / 768 / 12 |    64K |       110M | 
| [`lexlms/legal-xlm-longformer-base`](https://huggingface.co/lexlms/legal-xlm-longformer-base)    |           12 / 768 / 12 |    64K |       134M |

## Benchmarking on the Multilingual Legal Pile

| Model Name                         | Loss | Accuracy |              Legal |
|:-----------------------------------|-----:|---------:|-------------------:|
| `xlm-roberta-base`                 |    - |        - |                :x: |
| `lexlms/legal-xlm-base`            |    - |        - | :white_check_mark: |
| `lexlms/legal-xlm-longformer-base` |    - |        - | :white_check_mark: |

TODO: Present per language scores

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
python src/mod_teacher_model.py --teacher_model_path xlm-roberta-base --student_model_path data/plms/legal-xlm-base
```

### Longformerize pre-trained RoBERTa LM

```bash
export PYTHONPATH=.
python src/longformerize_model.py --roberta_model_path data/plms/legal-xlm-base --max_length 4096 --attention_window 128
```

## Pipeline

1. Train tokenizer (Only RoBERTa needed because we convert BERT models to RoBERTa)
```shell
    export PYTHONPATH=. && python src/pretraining/train_tokenizer.py | tee train_tokenizer.log
```
2. Evaluate tokenizer
```shell
    export PYTHONPATH=. && python src/pretraining/evaluate_tokenizer.py | tee evaluate_tokenizer.log
```
3. Mod Teacher Model
```shell
    export PYTHONPATH=. && python3 src/modding/mod_teacher_model.py --teacher_model_path xlm-roberta-base --student_model_path data/plms/legal-xlm-base_128k --output_dir data/plms/legal-xlm-base | tee mod_teacher_model.log
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
    export PYTHONPATH=. && python src/modding/longformerize_model.py | tee longformerize_model.log
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
