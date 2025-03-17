HF_HUB_OFFLINE=True
HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL_NAME="llava-v1.5-7b"
CKPT="TwigVLM-2f-lr1e4-3L-0205"
SPLIT="text_vqa"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m twig_inference.eval.model_vqa_loader \
        --model-path /mnt/pfs-mc0p4k/cv/team/zhenglihao/llm_common/$MODEL_NAME \
        --twig /mnt/pfs-mc0p4k/cv/team/wangmingyang/models/TwigVLM/$CKPT \
        --retained_tokens 192 \
        --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
        --image-folder ./playground/data/eval/textvqa/train_images \
        --answers-file ./playground/data/eval/textvqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --prompt-output-file ./playground/data/prompts/prompts.txt \
        --conv-mode vicuna_v1 &
done


wait

output_file=./playground/data/eval/textvqa/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/textvqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done



python -m twig_inference.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file $output_file