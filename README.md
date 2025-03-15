# twigvlm

[[📖 Technical report]()\]&nbsp;&nbsp;&nbsp;&nbsp;[[🤗Huggingface]()\]

## Table of Contents

- [twigvlm](#twigvlm)
  - [Table of Contents](#table-of-contents)
  - [Highlights](#highlights)
  - [News](#news)
  - [Demo](#demo)
  - [Prerequisites](#prerequisites)
  - [Model-zoo](#model-zoo)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [License](#license)
  - [About us](#about-us)
  - [Citation](#citation)

## Highlights


## News
- March 15, 2025: Training and evaluation codes of the `TwigVLM` model are released.
- 
## Demo

## Prerequisites

1. Clone this repository and navigate to the folder 
``` shell
git clone git@github.com:ricar0/twigvlm.git
cd twigvlm
```
2. Install Package

We recommend using [Anaconda](https://www.anaconda.com/) to create a new environment for the project, and install the requirements with the following commands:
``` shell
conda create -n twigvlm python=3.10 -y
conda activate twigvlm
pip install -r requirements.txt
pip install flash-attn==2.4.2 --no-build-isolation
``` 
<!-- 3. Download the pretrained base models (i.e., Phi-2 and SigLIP) to your local directories. (optional)
``` shell
python scripts/download_models.py
```
The base models will be stored in `checkpoints/base` in default.
```
checkpoints
└── base
    └── siglip-so400m-patch14-384
    └── phi-2
``` -->
## Model-zoo
You can download the checkpoints of the twig in [huggingface]().


If you want to train an twig from scratch, please refers to [Training](#training).


## Training
The training pipeline and datasets of our TwigVLM models are directly inherited from [LLaVA-v1.5](https://github.com/haotian-liu/LLaVA). 

## Evaluation
We follow the evaluation of [LLaVA-v1.5](https://github.com/haotian-liu/LLaVA/tree/main). 

Before preparing task-specific data, you should download [eval.zip](https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view?usp=sharing) and unzip it to `./playground/data/eval`. For more specific instructions, please refer to [LLaVA's Evaluation.md](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md). 

Before evaluation, simply add some configs as shown below:

**Loading weights**

``` python
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    twig=twig
)

```

**Twig Config**

```python
twigvlm_config = {
    "enable_FastV": True, 
    "attention_rank": 41, # retain visual tokens
    "generation_strategy": "self_speculative" # self_speculative | autoregressive
}

cont = model.generate(
    input_ids,
    images=image_tensor.unsqueeze(0).half().cuda(),
    do_sample=False,
    temperature=0,
    max_new_tokens=1024,
    eos_token_id=tokenizer.eos_token_id, # required
    image_sizes=image_sizes,
    twigvlm_config=twigvlm_config
)
text_outputs = tokenizer.batch_decode(cont.predicted_tokens, skip_special_tokens=True)[0]
```
| Models | GQA |MMB  | SQA(IMG) | TextVQA | POPE |  MME | MMB  |MM-Vet|
|:--------:|:----:|:----:|:-------------:|:--------:|:-----:|:----:|:-------:|:-------:|:-------:|
| [LLaVA-v1.5-lora](https://github.com/haotian-liu/LLaVA) (7B) |79.10 | 63.00 |47.80 |  68.40 |58.20| 86.40 | 1476.9 | 66.10  |30.2|
| [TinyGPT-V](https://github.com/DLYuanGod/TinyGPT-V) (3B) | - | 33.60  | 24.80  |    -   |    -  | -| - | -  |-|
| [LLaVA-Phi](https://github.com/zhuyiche/llava-phi) (3B) | 71.40  | - | 35.90 |    68.40   |    48.60  | 85.00 | 1335.1 | 59.80 |28.9|
| [MobileVLM](https://github.com/Meituan-AutoML/MobileVLM) (3B) | - | 59.00  | - |    61.00   |    47.50   | 84.90 | 1288.9 | 59.60  |-|
| [MC-LLaVA](https://huggingface.co/visheratin/MC-LLaVA-3b) (3B) | 64.24 | 49.60  | 24.88 |    -   |    38.59   | 80.59 | - | -  |-|
| [**TwigVLM**](https://huggingface.co/MILVLG/imp-v1-3b) | 79.45 | 58.55 | 50.09 |69.96| 59.38 | 88.02| 1434.0 | 66.49 |33.1|

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](https://www.apache.org/licenses/LICENSE-2.0) file for details.

## About us
This project is maintained by the [MILVLG](https://github.com/MILVLG)@Hangzhou Dianzi University (HDU) led by Prof. Zhou Yu and Jun Yu, and is mainly developed by Zhenwei Shao and Xuecheng Ouyang. We hope our model may serve as a strong baseline to inspire future research on MSLM, as well as its derivative applications on mobile devices and robots. 

## Citation

If you use our model or refer our work in your studies, please cite:

```bibtex
@article{imp2024,
  title={Imp: Highly Capable Large Multimodal Models for Mobile Devices},
  author={Shao, Zhenwei and Yu, Zhou and Yu, Jun and Ouyang, Xuecheng and Lihao, Zheng and Zhenbiao, Gai and Mingyang, Wang and Jiajun, Ding},
  journal={arXiv preprint arXiv:2405.12107},
  year={2024}
}
```