export WANDB_PROJECT="multilinguallegalpretraining"
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES = 0,1,2,3,4,5,6,7

# Just longformerize the xlm and swiss models

MODEL_MAX_LENGTH=4096
MODEL_PATH='plms/legal-xlm-longformer-base'
BATCH_SIZE=4
ACCUMULATION_STEPS=16

python3 src/pretraining/train_mlm.py \
    --model_name_or_path data/${MODEL_PATH} \
    --do_train \
    --do_eval \
    --dataset_name joelito/MultiLegalPile_Chunks_4000 \
    --output_dir data/${MODEL_PATH}-mlm \
    --overwrite_output_dir \
    --logging_steps 1000 \
    --evaluation_strategy steps \
    --eval_steps 32000 \
    --save_strategy steps \
    --save_steps 32000 \
    --save_total_limit 3 \
    --max_steps 64000 \
    --learning_rate 3e-5 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${ACCUMULATION_STEPS} \
    --eval_accumulation_steps ${ACCUMULATION_STEPS} \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --mlm_probability 0.15 \
    --max_seq_length ${MODEL_MAX_LENGTH} \
    --pad_to_max_length \
    --line_by_line \
    --max_eval_samples 10000 \
    --fp16 \
    --fp16_full_eval \
