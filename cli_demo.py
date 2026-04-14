from twigvlm.inference.builder import load_pretrained_model
from twigvlm.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from twigvlm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from twigvlm.conversation import conv_templates, SeparatorStyle
from twigvlm.inference.generator_utils.speculative_streamer import SpeculativeTextStreamer
from transformers import TextStreamer
from PIL import Image
import requests
import warnings
import argparse
from io import BytesIO
import time
import torch
import os
from transformers import logging
logging.set_verbosity_error()
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description="CLI demo for TwigVLM")
parser.add_argument('--base-model', type=str, default='liuhaotian/llava-v1.5-7b',
                    help='Base model path or name')
parser.add_argument('--twig-block', type=str, default='TwigVLM-llava-v1.5-7b-K2-T3',
                    help='Twig block checkpoint path or name')
parser.add_argument('--twig-K', type=int, default=2, help='Twig K value')
parser.add_argument('--twig-T', type=int, default=3, help='Twig T value')
parser.add_argument('--R', type=int, default=3, help='Avg retain visual tokens')
parser.add_argument('--stream', action='store_true', help='Enable streaming')
parser.add_argument('--image-file', type=str, required=True, help='Path to the image file or URL')
args = parser.parse_args()

os.environ['twig_K'] = str(args.twig_K)
os.environ['twig_T'] = str(args.twig_T)

# {"question_id": "20891675", "image": "n347706.jpg", "text": "", "category": "default"}
# Load the llava model
def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

model_path = args.base_model
twig = args.twig_block
device = "cuda"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    attn_implementation="eager",
    torch_type="bfloat16",
    twig=twig,
)

image_path = args.image_file
image_path = '/mnt/pfs-mc0p4k/cv/team/wangmingyang/twigvlm-leaf/playground/data/eval/gqa/images/images/n37274.jpg'
image = load_image(image_path)
image_tensor = process_images([image], image_processor, model.config)[0]
# Prepare conversation input
conv_mode = "vicuna_v1"
question = "What color is the floor sign in the photo?\nAnswer the question using a single word or phrase."
# question = input("USER: ")
if not question.strip():
    raise ValueError("Question cannot be empty.")
question = f"{DEFAULT_IMAGE_TOKEN}\n{question.strip()}"

conv = conv_templates[conv_mode].copy()
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

print(prompt_question)
input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size]
twigvlm_config = {
    "enable_pruning": True, 
    "avg_retain_rank": args.R, # avg retain visual tokens
    "generation_strategy": "self_speculative", # self_speculative | autoregressive
}

if args.stream:
    if twigvlm_config["generation_strategy"] == 'self_speculative':
        streamer = SpeculativeTextStreamer(tokenizer)
    else:
        streamer = TextStreamer(tokenizer)
else:
    streamer = None


model = model.generate(
    input_ids,
    images=image_tensor.unsqueeze(0).bfloat16().cuda(),
    do_sample=False,
    temperature=0.7,
    max_new_tokens=512,
    streamer=streamer, 
    eos_token_id=tokenizer.eos_token_id, # required
    image_sizes=image_sizes,
    twigvlm_config=twigvlm_config
)
print(model)
if not args.stream:
    text_outputs = tokenizer.batch_decode(model.predicted_tokens, skip_special_tokens=True)[0]
    print(text_outputs)

print(f"\nDecoding speed: {model.decoding_tokens_per_second} tokens/s")

# llava 1.5
# CUDA_VISIBLE_DEVICES=7 python cli_demo.py     --base-model /mnt/pfs-mc0p4k/cv/team/zhenglihao/llm_common/llava-v1.5-7b     --twig-block "/mnt/pfs-mc0p4k/cv/team/wangmingyang/models/Twigv2/shaozw/labs/TwigRL/checkpoints/stage2_0122/TwigVLM-leaf_modv7_7-lr1e-4-dynamic5-attnKL-stage2rl2v5_maxstep500-power2.0-group32-fdrop-fix"     --twig-K 2     --twig-T 3   --image-file "/mnt/pfs-mc0p4k/cv/team/wangmingyang/labs/LLaVA-upload/playground/data/eval/gqa/images/images/n260521.jpg" --R 64

# llava 1.6
# CUDA_VISIBLE_DEVICES=7 LEAF_MODULE=v7_7 python cli_demo.py     --base-model /mnt/pfs-mc0p4k/cv/team/wangmingyang/models/llava-v1.6-vicuna-7b     --twig-block "/mnt/pfs-mc0p4k/cv/team/wangmingyang/models/Twigv2/shaozw/labs/open_llava_next/checkpoints/TwigVLM-llavanext-leaf_modv7_7-2f-3L-lr1e-4-attnKL_v5_batchmean_b19_a1.0t1.0-predKL2_a0.1t5.0clip4."     --twig-K 2     --twig-T 3   --image-file "./assets/image.png"
