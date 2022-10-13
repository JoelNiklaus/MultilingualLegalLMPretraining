export WANDB_PROJECT="legal-xlm"
export PYTHONPATH=.

MODEL_MAX_LENGTH=512
MODEL_PATH='lexlms/legal-xlm-base'
BATCH_SIZE=32

python src/pretraining/eval_mlm.py \
    --model_name_or_path ${MODEL_PATH} \
    --do_eval \
    --dataset_name lexlms/Multi_Legal_Pile \
    --output_dir data/${MODEL_PATH}-eval \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --mlm_probability 0.15 \
    --max_seq_length ${MODEL_MAX_LENGTH} \
    --pad_to_max_length \
    --line_by_line
