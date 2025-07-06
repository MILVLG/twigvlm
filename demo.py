from twig_inference.model.builder import load_pretrained_model
from twig_inference.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from twig_inference.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from twig_inference.conversation import conv_templates, SeparatorStyle
from twig_inference.model.language_model.self_speculation.speculative_streamer import SpeculativeTextStreamer
from transformers import TextStreamer
from PIL import Image
import requests
import warnings
from io import BytesIO

warnings.filterwarnings("ignore")
# Load the llava model
def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

model_path = "liuhaotian/llava-v1.5-7b"
twig = "{path-dir}/TwigVLM-2f-lr1e4-3L-0205"
device = "cuda"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    attn_implementation="flash_attention_2",
    torch_type="float16",
    twig=twig,
)

image_path = './assets/image.png'
image = load_image(image_path)
image_tensor = process_images([image], image_processor, model.config)[0]
# Prepare conversation input
conv_mode = "vicuna_v1"
question = f"{DEFAULT_IMAGE_TOKEN}\nHow to cook this dish?"

conv = conv_templates[conv_mode].copy()
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

print(prompt_question)
input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size]
twigvlm_config = {
    "enable_FastV": True, 
    "attention_rank": 41, # retain visual tokens
    "generation_strategy": "self_speculative", # self_speculative | autoregressive
}

if twigvlm_config["generation_strategy"] == 'self_speculative':
    streamer = SpeculativeTextStreamer(tokenizer)
else:
    streamer = TextStreamer(tokenizer)

cont = model.generate(
    input_ids,
    images=image_tensor.unsqueeze(0).half().cuda(),
    do_sample=False,
    temperature=0.7,
    max_new_tokens=1024,
    # streamer=streamer, 
    eos_token_id=tokenizer.eos_token_id, # required
    image_sizes=image_sizes,
    twigvlm_config=twigvlm_config
)
# print(cont)
text_outputs = tokenizer.batch_decode(cont.predicted_tokens, skip_special_tokens=True)[0]
print(f"Decoding speed: {cont.decoding_tokens_per_second} tokens/s")
print(text_outputs)