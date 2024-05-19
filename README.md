# Hybrid Decision Mamba: A Promising In-Context RL Method for Long-Term Memory

This repository implements HDM. The implementation in this repositorory is used in the work "Hybrid Decision Mamba: A Promising In-Context RL Method for Long-Term Memory".

## 1. Usage

All core code is located within the gym folder.

## Run an experiment
```shell
python3 experient.py
```

## 2. Installation
Here we give an example installation on CUDA == 10.1. For non-GPU & other CUDA version installation, please refer to the [PyTorch website](https://pytorch.org/get-started/locally/).

``` Bash
# create conda environment
conda create -n HDM python==3.6.1
conda activate HDM
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```
You will also need [mujoco-py](https://github.com/openai/mujoco-py). Follow the installation instructions [here](https://github.com/openai/mujoco-py).
