# qlora-pipe
A pipeline parallel training script for LLMs.

## About
This is a training script I made so that I can fine-tune LLMs using my workstation with four 4090s. It is developed first and foremost for me, myself, and I. It is scrappy and hacked together. It will likely *never* be a stable, well-supported training script like Axolotl. I am open sourcing the code in case it is useful to others, and also as a proof-of-concept that this kind of thing is possible.

That being said, if something doesn't work right, or you would like it to support some feature, feel free to raise an issue and I'll try look at it.

## Features
- Pipeline parallel training, for efficiently training large models that cannot fit on one GPU
- Supports QLoRA, LoRA, and full fine tuning
- Efficient model loading. Each process only loads the layers it needs, and quantizes and moves them to the GPU layer-by-layer. This means you can load a large model on a lot of GPUs even with limited system RAM.
- Load any dataset that Axolotl can, using the same YAML config file format
- Support for "raw text" training using either a structured list of documents in a JSON file, or a single txt file
- Support for resuming training from a checkpoint, including the dataloader state, to easily allow training in a piecemeal fashion
- Useful metrics logged to Tensorboard
- Ability to specify a separate, fixed evaluation dataset
- Currently supports Llama, Mistral, Mixtral, and Qwen-1.5 model families

## Installing
Clone the repository:
```
git clone --recurse-submodules https://github.com/tdrussell/qlora-pipe
```

If you alread cloned it and forgot to do --recurse-submodules:
```
git submodule init
git submodule update
```

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
You could also try installing nvcc on its own, but installing the whole cuda toolkit was always the easiest for me.

Install packaging and ninja first, for flash-attn:
```
pip install packaging ninja
```

(Optional) Install flash-attn manually if you want to specify how many jobs are used for compiling:
```
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

Install the dependencies:
```
pip install -r requirements.txt
```

## Training
Edit the config files in the examples directory to your liking. At minimum, change the paths at the top to point to your model and desired output directory. Per-device batch size, gradient accumulation steps, and optimizer settings are specified in the Deepspeed JSON config file. Everything else is in the TOML config file. Launch the training script:
```
NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --deepspeed_config examples/ds_config_7b.json --config examples/config_7b.toml
```
RTX 4000 series needs those 2 enviroment variables set. Other GPUs may not need them.

## Parallelism
Deepspeed handles pipeline- and data-parallelism. Set the --num_gpus flag to however many GPUs to want to use. The config option pipeline_stages determines the level of model parallelism. Then, the data parallelism is automatically set so that all GPUs are used.

For example with 4 GPUs, and pipeline_stages=2, there are two data-parallel instances of the model, each divided across 2 GPUs, with the first half of the layers on one GPU and the second half on the other.

The option gradient_accumulation_steps in the Deepspeed JSON config file determines the amount of pipelining when using pipeline parallelism (pipeline_stages>1). The higher the value, the more the GPUs can overlap computation. For example, with gradient_accumulation_steps=1, there is a single batch that gets passed between the GPUs forward, then in reverse for the backward pass. Only 1 GPU is active at a time, the others are idle. As gradient_accumulation_steps increases, you start pipelining multiple forward/backward batches. At the beginning and end of the step, some GPUs will always be idle. So as gradient_accumulation_steps approaches infinity, you approach 100% theoretical utilization. In practice, a value of 8 or so already gives good average utilization with 2 GPUs. With more GPUs, you may want to go higher.

## Dataset configuration
There are 3 options for specifying the dataset. Set the dataset_type field to one of:
- axolotl
  - Loads the dataset using the Axolotl codebase. Set dataset_path to a YAML file that contains the same dataset configuration you would use in Axolotl.
- doclist
  - Set dataset_path to glob pattern matching one or more JSON or JSONL files. Each file should be a list of objects containing a 'text' key. For each file, all of the text is logically concatenated together, before being sliced into sequences.
- textfile
  - Basically the same as doclist, except the dataset_path matches one or more txt files. Each text file is sliced into sequences.

You can read dataset_utils.py for details on what each of these options is doing.

## On sample packing (or the lack thereof)
Sample packing is not currently implemented. Instead, there is the option batch_size_tokens. If this field is set, the batch size in the Deepspeed config file is ignored, and instead the batch size is adjusted dynamically to target a fixed number of tokens per batch, per device. This was easier to implement than sample packing, and does basically the same thing. It is also efficient: if I set batch_size_tokens to a modest 10000 and train a 7B model with the Alpaca dataset, all my 4090s hit their 350W power limit cap. Unless I'm missing something (definitely possible), it seems there is no need to support sample packing.