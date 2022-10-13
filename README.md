# Multilingual Legal Language Models

##  Legal XLM Models trained on the Multilingual Legal Pile

| Model Name                                                                                             | Layers / Units /  Heads | Vocab. | Parameters | 
|--------------------------------------------------------------------------------------------------------|-------------------------|--------|------------|
| [`lexlms/legal-xlm-base`](https://huggingface.co/lexlms/legal-xlm-base)                              | 12 / 768 / 12           | 64K    | 110M       | 
| [`lexlms/legal-xlmr-base`](https://huggingface.co/lexlms/legal-xlmr-base)                            | 12 / 768 / 12           | 64K    | 110M       | 
| [`lexlms/legal-xlm-longformer-base`](https://huggingface.co/lexlms/legal-xlm-longformer-base)        | 12 / 768 / 12           | 64K    | 134M       |


## Benchmarking on the Multilingual Legal Pile

| Model Name                          | Loss | Accuracy | Legal              |
|-------------------------------------|------|----------|--------------------|
| `xlm-roberta-base`                  | -    | -        | :x:                |
| `lexlms/legal-xlm-base`            | -    | -        | :white_check_mark: |
| `lexlms/legal-xlmr-base`           | -    | -        | :white_check_mark: |
| `lexlms/legal-xlm-longformer-base` | -    | -        | :white_check_mark: |

TODO: Present per language scores


## Code Base

### Train XLM

For TPU acceleration use the following script:

```shell
sh train_mlm_tpu.sh
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