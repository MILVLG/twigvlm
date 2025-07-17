export HF_HUB_OFFLINE=True
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

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
SPLIT="mmbench"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python twigvlm/eval/model_vqa_mmbench.py   \
        --model-path {path-dir}/$MODEL_NAME \
        --twig {path-dir}/$CKPT \
        --retained_tokens $R \
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