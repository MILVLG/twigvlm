HF_HUB_OFFLINE=True
HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

R=0

# 解析参数
while getopts ":R:" opt; do
  case $opt in
    R)
      R="$OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      ;;
  esac
done

MODEL_NAME="llava-v1.5-7b"
CKPT="TwigVLM-llava-v1.5-7b-K2-T3"
SPLIT="pope"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m twigvlm.eval.model_vqa_loader \
        --model-path {path-dir}/$MODEL_NAME \
        --twig {path-dir}/$CKPT \
        --retained_tokens $R \
        --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
        --image-folder ./playground/data/eval/pope/val2014 \
        --answers-file ./playground/data/eval/pope/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait


output_file=./playground/data/eval/pope/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/pope/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python twigvlm/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file $output_file