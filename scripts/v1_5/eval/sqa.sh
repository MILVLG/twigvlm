HF_HUB_OFFLINE=True
HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1

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
SPLIT="sqa"

#/data/llm_common/vicuna-7b-v1.5 
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m twigvlm.eval.model_vqa_science \
        --model-path {path-dir}/$MODEL_NAME \
        --twig {path-dir}/$CKPT \
        --retained_tokens $r \
        --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
        --image-folder ./playground/data/eval/scienceqa/images \
        --answers-file ./playground/data/eval/scienceqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --single-pred-prompt \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/scienceqa/answers/$SPLIT/$CKPT/caption_sqa.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/scienceqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


python twig_inference/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file $output_file \
    --output-file ./playground/data/eval/scienceqa/answers/${MODEL_NAME}_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/${MODEL_NAME}_result.json