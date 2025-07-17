# TwigVLM

This repository contains the official code of our [paper](https://arxiv.org/abs/2503.14075) accepted at ICCV 2025. TwigVLM is a general and effective framework that accelerates large visual language models (LVLMs) by “growing” a lightweight twig block on top of an early layer of the base VLM.

Compared to existing VLM acceleration methods that are purely based on visual token pruning, our TwigVLM not only retains better accuracy by employing a twig-guided token pruning (TTP) strategy, but also achieves higher generation speed by utilizing a self-speculative decoding (SSD) strategy. More specifically, the LLaVA-1.5-7B model with our TwigVLM can retain 96% of the original performance when 88.9% of visual tokens are pruned, and achieves a 154% improvement in generation speed, which establishes a new state-of-the-art in terms of both accuracy retention and generation speed in the field of VLM acceleration.

<p align="center" width="100%">
<img src="./assets/fig1.png" alt="TwigVLM" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

## Table of Contents

- [Prerequisites](#prerequisites)
- [Training](#training)
- [Evaluation](#evaluation)
- [Demo](#demo)
- [License](#license)
- [About us](#about-us)
- [Citation](#citation)
  
<!-- ## News
- July 5, 2025: Training and evaluation codes of the `TwigVLM` model are released. -->


## Prerequisites
0. To train the models, you will need a server with **at least 4 GPUs**, each with **more than 40GB of memory** (e.g., 4×NVIDIA A6000). For inference or testing, **a single GPU with >40GB memory** is sufficient.
1. Clone this repository and navigate to the folder:
``` shell
git clone https://github.com/MILVLG/twigvlm.git
cd twigvlm
```
2. Prepare the software environment. We recommend using [Anaconda](https://www.anaconda.com/) to create a new environment for the project, and install the requirements with the following commands:
``` shell
conda create -n twigvlm python=3.10 -y
conda activate twigvlm
pip install -r requirements.txt
pip install flash-attn==2.3.2 --no-build-isolation
``` 
3. Please note that **SDPA** is currently unsupported. We recommend using **FlashAttention-2** as the preferred backend. If FlashAttention-2 cannot be installed, the **eager** implementation can be used as a fallback option.
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


## Training

This section provides the instructions for training the TwigVLM for the LLaVA-1.5-7B model. Please refer to the original [LLaVA project](https://github.com/haotian-liu/LLaVA) to prepare the training data [here](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#visual-instruction-tuning) and the base model [LLaVA-1.5-7b](https://huggingface.co/haotian-liu/LLaVA-1.5-7b). After that, you can use the following script to train the TwigVLM:

``` shell
# Training the twig block
twig_K=2 twig_T=3 bash scripts/v1_5/train_twig.sh
```

where `twig_K` and `twig_T` are the position of the twig block and the number of twig layers, respectively (see the paper for details). 

The trained checkpoints will be stored in `./checkpoints/TwigVLM-llava1.5-7b-K2-T3` by default. The trained TwigVLM model (only the learned twig block) is available at [here](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EeMoUa43kk5CrClb6qGsXgkBDfoZtK4EFO7nPnB8Ma6hQA?download=1).

## Evaluation

This section provides the instructions for evaluating the TwigVLM and reproducing the results with LLaVA-1.5-7B reported in the paper. Before preparing task-specific data, you should download [eval.zip](https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view?usp=sharing) and unzip it to `./playground/data/eval`. For more specific instructions, please take a look at [LLaVA's Evaluation.md](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md). 

Example for evaluating GQA benchmark, where `-R` is the average number of retained visual tokens:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  twig_K=2 twig_T=3 bash scripts/v1_5/eval/gqa.sh  -R 192
```

Using our provided model, you can reproduce the following results in `R=192`. 
| Models | GQA | MME | MMBench | SQA(IMG) | TextVQA | VQAv2  | RelAcc |
|:--------:|:----:|:----:|:--------:|:--------:|:-----:|:----:|:-------:|
| [SparseVLM](https://github.com/Gumpest/SparseVLMs) | 57.6 | 1721 | 62.5 | 69.1 | 56.1 | 75.6 | 95.7% |
| [MustDrop](https://github.com/liuting20/MustDrop) | 58.2	| 1787 | 62.3 |	69.2 | 56.5	| 76 | 96.6% |
| [VisionZip](https://github.com/dvlab-research/VisionZip) | 59.3 | 1783 |63 | **68.9**	| 57.3 | 76.8	| 97.4% |
| [VisionZip*](https://github.com/dvlab-research/VisionZip) | 60.1	| 1834 | 63.4 | 68.2 | 57.8 |	77.4 | 98.3% |
| [**TwigVLM**](#) | **61.2** | **1848**| **64** | **68.9** | **58**  | **78.1** | **99.2%** |

Example for evaluating generation speed:
```
CUDA_VISIBLE_DEVICES=0  twig_K=2 twig_T=3 bash scripts/v1_5/eval/mmvet.sh  -R 64
```
Running on an `RTX 4090` GPU, the average generation speed is about 60.6 tokens/s. Note that different GPUs may exhibit fluctuations when handling parallel computations.


## Demo

To test some cases, you can use the provided `cli_demo.py` script. This script allows you to interactively ask questions about an image using the TwigVLM model.

```Python
python cli_demo.py \
    --base-model liuhaotian/llava-v1.5-7b \
    --twig-block "TwigVLM-llava-v1.5-7b-K2-T3" \
    --twig-K 2 \
    --twig-T 3 \
    --stream \
    --image-file "./assets/image.png"
```

The generation process is demonstrated in the following GIF image. The tokens in green are generated by the draft model.

<p align="center" width="100%">
<img src="./assets/demo.gif" alt="Stanford-Alpaca" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>


## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](https://www.apache.org/licenses/LICENSE-2.0) file for details.

## About us
This project is maintained by the [MILVLG](https://github.com/MILVLG) @ Hangzhou Dianzi University (HDU).  

## Citation

If this code is used in your research, please cite our paper:

```bibtex
@inproceedings{shao2025twigvlm,
  title={Growing a twig to accelerate large vision-language models},
  author={Shao, Zhenwei and Wang, Mingyang and Yu, Zhou and Pan, Wenwen and Yang, Yan and Wei, Tao and Zhang, Hongyuan and Mao, Ning and Chen, Wei and Yu, Jun},
  journal={IEEE International Conference on Computer Vision (ICCV)},
  year={2025}
}
```
