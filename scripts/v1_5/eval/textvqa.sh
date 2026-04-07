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

# MODEL_NAME="llava-v1.5-7b"
# CKPT="TwigVLM-llava-v1.5-7b-K2-T3"
MODEL_NAME="llava-v1.5-7b"
CKPT="TwigVLM-leaf_modv7_7-lr1e-4-dynamic5-attnKL-stage2rl2v5_maxstep500-power2.0-group32-fdrop-fix"
SPLIT="text_vqa"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m twigvlm.eval.model_vqa_loader \
        --model-path /mnt/pfs-mc0p4k/cv/team/zhenglihao/llm_common/$MODEL_NAME \
        --twig /mnt/pfs-mc0p4k/cv/team/wangmingyang/models/Twigv2/shaozw/labs/TwigRL/checkpoints/stage2_0122/$CKPT \
        --retained_tokens $R \
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



python -m twigvlm.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file $output_file