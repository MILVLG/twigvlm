import argparse
import torch
import os
import json
from tqdm import tqdm
import uuid

from twigvlm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from twigvlm.conversation import conv_templates, SeparatorStyle
from twigvlm.inference.builder import load_pretrained_model
from twigvlm.utils import disable_torch_init
from twigvlm.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, KeywordsStoppingCriteria
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, prompt_output_file):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.prompt_output_file = prompt_output_file  # 保存 prompt 的 JSON 文件

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        #print(f"qs: {qs}")

        if args.single_pred_prompt:
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # print(f"prompt: {prompt}")

        # 保存每个 prompt 到文件
        question_id = line.get("question_id", index)  # 如果问题有 question_id 使用它，否则用 index
        with open(self.prompt_output_file, "a") as f:
            f.write(f"Question ID: {question_id}\n")
            f.write(f"Prompt:\n{prompt}\n")
            f.write("-" * 50 + "\n")  # 添加分隔符以区分各个 prompt

        # 打印相关信息
        #print(f"Processed Question ID: {question_id}")
        #print(f"Prompt:\n{prompt}\n")

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        image_sizes = [image.size]
        return input_ids, image_tensor, image_sizes

    def __len__(self):
        return len(self.questions)

# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, prompt_output_file, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, prompt_output_file)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, torch_type="float16", attn_implementation="flash_attention_2", twig=args.twig)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    # 创建 DataLoader
    data_loader = create_data_loader(
        questions,
        args.image_folder,
        tokenizer,
        image_processor,
        model.config,
        prompt_output_file=args.prompt_output_file
    )

    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        twigvlm_config = {
            "enable_pruning": True, 
            "attention_rank": args.retained_tokens, # retain visual tokens
            "generation_strategy": "self_speculative" # self_speculative | autoregressive
        }
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.half().cuda(),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                image_sizes=image_sizes,
                twigvlm_config=twigvlm_config)
        
        # outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = tokenizer.batch_decode(output_ids.predicted_tokens, skip_special_tokens=True)[0]
        # print(f"outputs: {outputs}")
        ans_id = uuid.uuid4().hex
        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": cur_prompt,
            "text": outputs,
            "answer_id": ans_id,
            "model_id": model_name,
            "metadata": {}
        }) + "\n")

        # 立即保存每个答案并打印处理状态
        ans_file.flush()
        #print(f"Saved answer for Question ID: {idx}")

    ans_file.close()


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
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--prompt-output-file", type=str, default="prompts.txt", help="File to save all generated prompts as JSON")
    args = parser.parse_args()

    eval_model(args)