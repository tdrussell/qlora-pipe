## Environment Setup
Install Miniconda: https://docs.conda.io/en/latest/miniconda.html

Create the environment
```
conda create -n training python=3.12
conda activate training
```

Install Pytorch: https://pytorch.org/get-started/locally/

Install cuda toolkit (make sure it matches the cuda version you used for Pytorch), e.g.:
```
conda install nvidia/label/cuda-12.1.1::cuda-toolkit
```

Install packaging and ninja first, for flash-attn:
```
pip install packaging ninja
```

Install the dependencies:
```
pip install -r requirements.txt
```

## Training
```
deepspeed --num_gpus=1 train_lora_deepspeed.py --deepspeed --deepspeed_config ds_config_7b.json --config config_7b.toml
```