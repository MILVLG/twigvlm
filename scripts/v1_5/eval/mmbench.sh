export HF_HUB_OFFLINE=True
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL_NAME="llava-v1.5-7b"
CKPT="TwigVLM-2f-lr1e4-3L-0205"
SPLIT="mmbench"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python twig_inference/eval/model_vqa_mmbench.py   \
        --model-path /mnt/pfs-mc0p4k/cv/team/zhenglihao/llm_common/$MODEL_NAME \
        --twig /mnt/pfs-mc0p4k/cv/team/wangmingyang/models/TwigVLM/$CKPT \
        --retained_tokens 227 \
        --question-file ./playground/data/eval/mmbench/mmbench_dev_20230712.tsv \
        --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --single-pred-prompt \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done


wait

output_file=./playground/data/eval/mmbench/answers/$SPLIT/$CKPT/${MODEL_NAME}_sigle.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/mmbench/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

mkdir -p ./playground/data/eval/mmbench/answers_upload/$SPLIT/${CKPT}_sigle

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/mmbench_dev_20230712.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT/$CKPT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT/${CKPT}_sigle \
    --experiment ${MODEL_NAME}_sigle