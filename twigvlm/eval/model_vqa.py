import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import time
from twigvlm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from twigvlm.conversation import conv_templates, SeparatorStyle
from twigvlm.inference.builder import load_pretrained_model
from twigvlm.utils import disable_torch_init
from twigvlm.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, attn_implementation="flash_attention_2", twig=args.twig)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if args.output_file is not None:
        output_file = os.path.expanduser(args.output_file)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        out_file = open(output_file, "w")
    
    decoding_time, num_tokens_generated = 0, 0
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]

        twigvlm_config = {
            "enable_pruning": True, 
            "attention_rank": args.retained_tokens, # retain visual tokens
            "generation_strategy": "self_speculative" # self_speculative | autoregressive
        }

        torch.cuda.synchronize()
        start = time.time()
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            image_sizes=[image.size],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            # no_repeat_ngram_size=3,
            max_new_tokens=1024,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            twigvlm_config=twigvlm_config
        )
        torch.cuda.synchronize()
        end = time.time()
        outputs = tokenizer.batch_decode(output_ids.predicted_tokens, skip_special_tokens=True)[0].strip()
        
        decoding_time += end-start
        num_tokens_generated += output_ids.num_tokens_generated
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()
    print(f"avg generation speed: {num_tokens_generated/decoding_time} tokens/s")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--twig", type=str, default=None)
    parser.add_argument("--retained_tokens", type=int, default=192)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--output-file", type=str, default=None)
    args = parser.parse_args()
    eval_model(args)
