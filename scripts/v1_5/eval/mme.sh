export HF_HUB_OFFLINE=True
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

r=0

# 解析参数
while getopts ":r:" opt; do
  case $opt in
    r)
      r="$OPTARG"
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
SPLIT="mme"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m twigvlm.eval.model_vqa_loader \
        --model-path {path-dir}/$MODEL_NAME \
        --twig {path-dir}/$CKPT \
        --retained_tokens $r \
        --question-file ./playground/data/eval/MME/llava_mme.jsonl \
        --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
        --answers-file ./playground/data/eval/MME/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done



wait

output_file=./playground/data/eval/MME/answers/$SPLIT.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/MME/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# python -m llava.eval.model_vqa_loader \
#     --model-path liuhaotian/llava-v1.5-13b \
#     --question-file ./playground/data/eval/MME/llava_mme.jsonl \
#     --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
#     --answers-file ./playground/data/eval/MME/answers/llava-v1.5-13b.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment $SPLIT #llava-v1.5-13b

cd eval_tool

python calculation.py --results_dir answers/$SPLIT #llava-v1.5-13b