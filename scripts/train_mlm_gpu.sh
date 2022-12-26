export WANDB_PROJECT="multilinguallegalpretraining"
export PYTHONPATH=.

MODEL_MAX_LENGTH=512
MODEL_NAME=legal-german-roberta-base
MODEL_PATH=$SCRATCH/MultilingualLMPretraining/data/plms/${MODEL_NAME}
LANGUAGES=de

cp -r data/plms/${MODEL_NAME} ${MODEL_PATH}

HF_NAME=joelito

NUM_CPUS=32

# base
# 2 A100 GPUs x batch size x accumulation steps = 512
TOTAL_BATCH_SIZE=512
NUM_GPUS=2
BATCH_SIZE=16
ACCUMULATION_STEPS=$(expr ${TOTAL_BATCH_SIZE} / ${BATCH_SIZE} / ${NUM_GPUS})

python3 src/pretraining/train_mlm.py \
    --model_name_or_path ${MODEL_PATH} \
    --do_train \
    --do_eval \
    --output_dir ${MODEL_PATH}-mlm \
    --dataset_name joelito/MultiLegalPile_Chunks_500 \
    --languages ${LANGUAGES} \
    --streaming True \
    --preprocessing_num_workers ${NUM_CPUS} \
    --logging_steps 1000 \
    --evaluation_strategy steps \
    --eval_steps 50000 \
    --save_strategy steps \
    --save_steps 50000 \
    --save_total_limit 5 \
    --max_steps 1000000 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${ACCUMULATION_STEPS} \
    --eval_accumulation_steps ${ACCUMULATION_STEPS} \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --mlm_probability 0.20 \
    --freeze_model_encoder \
    --max_seq_length ${MODEL_MAX_LENGTH} \
    --pad_to_max_length \
    --line_by_line \
    --hub_model_id=${HF_NAME}/${MODEL_NAME} \
    --hub_strategy=checkpoint \
    --push_to_hub \
    --hub_private_repo \
    --hub_token=${AUTH_TOKEN} \
    --max_eval_samples 5000 \
    --bf16 \
    --bf16_full_eval

