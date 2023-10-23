## Environment Setup
Install Miniconda: https://docs.conda.io/en/latest/miniconda.html

Create the environment
```
conda create -n training python=3.10
conda activate training
```

Install Pytorch: https://pytorch.org/get-started/locally/

Install cuda toolkit (make sure it matches the cuda version you used for Pytorch), e.g.:
```
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
```

Install the dependencies:
```
pip install -r requirements.txt
```

(Optional) Install flash attention and/or xformers:
```
pip install flash-attn
pip install xformers
```

## Training
```
deepspeed --num_gpus=1 train_lora_deepspeed.py --deepspeed --deepspeed_config ds_config_7b.json --config config_7b.toml
```