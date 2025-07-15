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
import os

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description="CLI demo for TwigVLM")
parser.add_argument('--base-model', type=str, default='liuhaotian/llava-v1.5-7b',
                    help='Base model path or name')
parser.add_argument('--twig-block', type=str, default='TwigVLM-llava-v1.5-7b-K2-T3',
                    help='Twig block checkpoint path or name')
parser.add_argument('--twig-K', type=int, default=2, help='Twig K value')
parser.add_argument('--twig-T', type=int, default=3, help='Twig T value')
parser.add_argument('--stream', action='store_true', help='Enable streaming')
parser.add_argument('--image-file', type=str, required=True, help='Path to the image file or URL')
args = parser.parse_args()

os.environ['twig_K'] = str(args.twig_K)
os.environ['twig_T'] = str(args.twig_T)


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
    attn_implementation="flash_attention_2",
    torch_type="float16",
    twig=twig,
)

image_path = args.image_file
image = load_image(image_path)
image_tensor = process_images([image], image_processor, model.config)[0]
# Prepare conversation input
conv_mode = "vicuna_v1"
# question = f"{DEFAULT_IMAGE_TOKEN}\nHow to cook this dish?"
question = input("USER: ")
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
    "attention_rank": 41, # retain visual tokens
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
    images=image_tensor.unsqueeze(0).half().cuda(),
    do_sample=False,
    temperature=0.7,
    max_new_tokens=1024,
    streamer=streamer, 
    eos_token_id=tokenizer.eos_token_id, # required
    image_sizes=image_sizes,
    twigvlm_config=twigvlm_config
)

if not args.stream:
    text_outputs = tokenizer.batch_decode(model.predicted_tokens, skip_special_tokens=True)[0]
    print(text_outputs)

print(f"\nDecoding speed: {model.decoding_tokens_per_second} tokens/s")