#!/bin/bash
export HF_HUB_OFFLINE=True
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1


# export twig_K=2
# export twig_T=3

DATA_PATH="{dir}/llava_v1_5_mix665k.json"
IMAGE_FOLDER="{dir}/dataset_images"
MODEL_NAME_OR_PATH="liuhaotian/llava-v1.5-7b"  # "{local_dir}/llava-v1.5-7b"

start_time=$(date +%s)
deepspeed twigvlm/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --version v1 \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/TwigVLM-llava1.5-7b-K${twig_K}-T${twig_T} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none


# record time
end_time=$(date +%s)

total_time=$((end_time - start_time))

hours=$((total_time / 3600))
minutes=$(((total_time % 3600) / 60))

echo "Total time: $hours hours $minutes mins"

