MODEL_PATH='data/plms/legal-xlm-base'

python3 src/pretraining/run_mlm_flax.py \
    --model_name_or_path="${MODEL_PATH}" \
    --output_dir="${MODEL_PATH}-mlm" \
    --model_type="roberta" \
    --dataset_name="joelito/MultiLegalPile_Wikipedia_Filtered" \
    --dataset_config_name="de_all" \
    --max_seq_length="512" \
    --weight_decay="0.01" \
    --per_device_train_batch_size="128" \
    --per_device_eval_batch_size="128" \
    --learning_rate="3e-4" \
    --warmup_steps="1000" \
    --overwrite_output_dir \
    --num_train_epochs="18" \
    --adam_beta1="0.9" \
    --adam_beta2="0.98" \
    --logging_steps="500" \
    --save_steps="2500" \
    --eval_steps="2500" \
    --push_to_hub \
    --hub_model_id=joelito/legal-xlm-base \

