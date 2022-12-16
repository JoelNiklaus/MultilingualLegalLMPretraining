# run with sudo bash train_mlm_tpu.sh

export WANDB_PROJECT="multilinguallegalpretraining"
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
export AUTH_TOKEN='<put your wandb token here>'
export PYTHONPATH=.

MODEL_MAX_LENGTH=512
MODEL_PATH='plms/legal-xlm-base'
#BATCH_SIZE=16 # maximum for large size models
BATCH_SIZE=32 # maximum for base size models
#ACCUMULATION_STEPS=4 # for large size models
ACCUMULATION_STEPS=2 # for base size models
# 8 TPU cores on v3-8 x batch size x accumulation steps = 512

# 1M steps will take approx. 10 days
# larger mlm probability because of https://arxiv.org/abs/2202.08005

sudo python3 src/pretraining/xla_spawn.py --num_cores=8 src/pretraining/train_mlm.py \
    --model_name_or_path data/${MODEL_PATH} \
    --do_train \
    --do_eval \
    --dataset_name joelito/MultiLegalPile_Chunks_500 \
    --output_dir data/${MODEL_PATH}-mlm \
    --overwrite_output_dir \
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
    --hub_model_id=joelito/legal-xlm-base \
    --hub_strategy=checkpoint \
    --push_to_hub \
    --hub_private_repo \
    --hub_token=${AUTH_TOKEN} \
    --max_eval_samples 10000

