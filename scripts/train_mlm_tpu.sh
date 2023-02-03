#!/bin/bash

# run with sudo bash train_mlm_tpu.sh

export PYTHONPATH=.
export WANDB_PROJECT="multilinguallegalpretraining"
export TOKENIZERS_PARALLELISM=true
export XRT_TPU_CONFIG="localservice;0;localhost:51011"

HF_AUTH_TOKEN='<hf_token>'
HF_NAME=joelito

HUB_MODEL_PATH=${HF_NAME}/${MODEL_NAME}


MODEL_MAX_LENGTH=512
MODEL_NAME=legal-french-roberta-base
MODEL_PATH=data/plms/${MODEL_NAME}
LANGUAGES=fr

MAX_STEPS=1000000 # for multilingual models and
MAX_STEPS=100000 # for short monolingual models so we can have one run for every language

STREAMING=True
CONTINUE_TRAINING=false

if [[ $CONTINUE_TRAINING == true ]]; then
  FREEZE_MODEL_ENCODER=False
  MODEL_NAME_OR_PATH=$HUB_MODEL_PATH
else
  FREEZE_MODEL_ENCODER=True
  MODEL_NAME_OR_PATH=$MODEL_PATH
fi

# 8 TPU cores on v3-8 x batch size x accumulation steps = 512
TOTAL_BATCH_SIZE=512
TPU_CORES=8
# Set BATCH_SIZE to 8 if "large" is present in MODEL_NAME, and to 16 if "base" is present in MODEL_NAME
if [[ $MODEL_NAME == *"large"* ]]; then
  BATCH_SIZE=8
  MLM_PROBABILITY=0.30
elif [[ $MODEL_NAME == *"base"* ]]; then
  BATCH_SIZE=16
  MLM_PROBABILITY=0.20
else
  # Set defaults for other models
  BATCH_SIZE=32
  MLM_PROBABILITY=0.15
fi
ACCUMULATION_STEPS=$(( ${TOTAL_BATCH_SIZE} / ${BATCH_SIZE} / ${TPU_CORES} ))

# 1M steps will take approx. 10 days
# larger mlm probability because of https://arxiv.org/abs/2202.08005

sudo python3 src/pretraining/xla_spawn.py --num_cores=${TPU_CORES} src/pretraining/train_mlm.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --streaming ${STREAMING} \
    --do_train \
    --do_eval \
    --output_dir ${MODEL_PATH}-mlm \
    --dataset_name joelito/MultiLegalPile_Chunks_500 \
    --languages ${LANGUAGES} \
    --logging_steps 1000 \
    --evaluation_strategy steps \
    --eval_steps 50000 \
    --save_strategy steps \
    --save_steps 50000 \
    --save_total_limit 5 \
    --max_steps ${MAX_STEPS} \
    --learning_rate 1e-4 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${ACCUMULATION_STEPS} \
    --eval_accumulation_steps ${ACCUMULATION_STEPS} \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --mlm_probability ${MLM_PROBABILITY} \
    --freeze_model_encoder ${FREEZE_MODEL_ENCODER} \
    --max_seq_length ${MODEL_MAX_LENGTH} \
    --pad_to_max_length \
    --line_by_line \
    --hub_model_id=${HUB_MODEL_PATH} \
    --hub_strategy=checkpoint \
    --push_to_hub \
    --hub_private_repo \
    --hub_token=${HF_AUTH_TOKEN} \
    --max_eval_samples 5000

