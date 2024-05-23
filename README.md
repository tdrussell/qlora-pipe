# qlora-pipe
A pipeline parallel training script for LLMs.

Refer to the changelog at the bottom for details on updates.

## About
This is a training script I made so that I can fine-tune LLMs using my workstation with four 4090s. It is developed first and foremost for myself, with my own use cases in mind. It is scrappy and hacked together. It will likely *never* be a stable, well-supported training script like Axolotl. I am open sourcing the code in case it is useful to others, and also as a proof-of-concept that this kind of thing is possible.

That being said, if something doesn't work right, or you would like it to support some feature, feel free to raise an issue and I'll try to look at it.

## Features
- Pipeline parallel training, for efficiently training large models that cannot fit on one GPU
- Supports QLoRA, LoRA, and full fine tuning
- Quantize weights using either bitsandbytes or HQQ
- Efficient model loading. Each process only loads the layers it needs, and quantizes and moves them to the GPU layer-by-layer. This means you can load a large model on a lot of GPUs even with limited system RAM.
- Load any dataset that Axolotl can, using the same YAML config file format
- Support for "raw text" training using either a structured list of documents in a JSON file, or a single txt file
- Support for resuming training from a checkpoint, including the dataloader state, to easily allow training in a piecemeal fashion
- Useful metrics logged to Tensorboard
- Ability to specify a separate, fixed evaluation dataset
- Train on multiple datasets simultaneously, with different sampling ratios per dataset
- Models currently supported: Llama, Mistral, Mixtral, Qwen-1.5, Cohere (Command R), Phi-3 (mini and medium)

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

Install Pytorch **NIGHTLY** version (if before 2.4): https://pytorch.org/get-started/locally/

Until Pytorch is at 2.4, HQQ quantization requires Pytorch nightly if using Python 3.12, because of torch.compile().

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
Edit the config files in the examples directory to your liking. At minimum, change the paths at the top to point to your model and desired output directory. Per-device batch size and gradient accumulation steps are specified in the Deepspeed JSON config file. Everything else is in the TOML config file. Launch the training script:
```
NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --deepspeed_config examples/ds_config_7b.json --config examples/config_7b.toml
```
RTX 4000 series needs those 2 enviroment variables set. Other GPUs may not need them.

## Parallelism
Deepspeed handles pipeline- and data-parallelism. Set the --num_gpus flag to however many GPUs to want to use. The config option `pipeline_stages` determines the level of model parallelism. Then, the data parallelism is automatically set so that all GPUs are used.

For example with 8 GPUs, and pipeline_stages=4, a single instance of the model is divided across 4 GPUs. Because there are 8 GPUs total, there are then 2 data-parallel instances.

The option `gradient_accumulation_steps` in the Deepspeed JSON config file determines the amount of pipelining when using pipeline parallelism (pipeline_stages>1). The higher the value, the more the GPUs can overlap computation. For example, with gradient_accumulation_steps=1, there is a single batch that gets passed between the GPUs forward, then in reverse for the backward pass. Only 1 GPU is active at a time, the others are idle. As gradient_accumulation_steps increases, you start pipelining multiple forward/backward batches. At the beginning and end of the step, some GPUs will always be idle. So as gradient_accumulation_steps approaches infinity, you approach 100% theoretical utilization. In practice, a value of 8 or so already gives good average utilization with 2 GPUs. With more GPUs, you may want to go higher.

## Dataset configuration
There are 3 options for specifying each dataset. Set the `dataset_type` field to one of:
- axolotl
  - Loads the dataset using the Axolotl codebase. Set `dataset_path` to a YAML file that contains the same dataset configuration you would use in Axolotl.
- doclist
  - Set `dataset_path` to glob pattern matching one or more JSON or JSONL files. Each file should be a list of objects containing a 'text' key. For each file, all of the text is logically concatenated together, before being sliced into sequences.
- textfile
  - Basically the same as doclist, except the `dataset_path` matches one or more txt files. Each text file is sliced into sequences.

You can read dataset_utils.py for details on what each of these options is doing.

You can have multiple datasets. Just add additional `[[datasets]]` entries. When using multiple datasets, there are different ways to combine them.
- `dataset_combination_mode` = 'concatenate' (the default)
  - Just concatenates the datasets.
- `dataset_combination_mode` = 'interleave'
  - Uses the Huggingface Datasets library `interleave_datasets()` function.
  - Use the `dataset_interleave_stopping_strategy` setting to control when interleaving stops.
    - 'first_exhausted': stop when a dataset runs out of examples.
    - 'all_exhausted': stop when all datasets have run out of examples. This duplicates examples from smaller datasets.
  - When using the 'interleave' mode, datasets can have a relative `sample_weight`, which is a positive real number. This controls the relative proportion of the datasets when they are combined.
  - __IMPORTANT__: When using the 'interleave' mode, the manner in which the datasets are proportionally combined (i.e. sampled from) is affected  by the `batch_size_tokens` setting:
    - If `batch_size_tokens` is unset, it means you are treating each example equally. Every batch has the same number of examples, even though they may be different lengths. So, when interleaving datasets, the rows are sampled according to the relative proportions given by the `sample_weight`.
    - If using `batch_size_tokens`, it means you are treating each token equally. Every batch varies the number of examples (because they might have different lengths) so that the token count is approximately constant. So, when interleaving datasets, the sampling ratios are adjusted so that the number of *tokens*, not rows, drawn from different datasets matches the `sample_weight`. This is implemented by scaling the sampling probabilities by the average length of the dataset. You can read the `combine_datasets()` function in dataset_utils.py if this is confusing.
    - __Which of these should I use?__ Probably set `batch_size_tokens`. I think this is the better way to think about things, and it matches what sample packing would do. For example, in Axolotl, it is recommended to use sample packing, which packs multiple examples into a single sequence so that the sequence length is constant. This means, in the loss function, each token is being treated with equal weight, not each original row in the dataset. Using `batch_size_tokens` in this training script mimics that behavior, and thus when interleaving datasets, it samples from them so that the token ratios adhere to the sample_weight specified.
    - __Example__: you have datasets A and B. B's average row length is twice that of A. A has a sample_weight of 2, B has a sample_weight of 1.
      - Not setting batch_size_tokens: when interleaving, you get 2 rows of A for every row of B.
      - Using batch_size_tokens: when interleaving, you get 4 rows of A for every row of B. This is because A's rows are on average half the length of B's rows, so you need twice as many as before so that the number of tokens in each matches the 2:1 ratio you specified with the sample_weight.

## On sample packing (or the lack thereof)
Sample packing is not currently implemented. Instead, there is the option `batch_size_tokens`. If this field is set, the batch size in the Deepspeed config file is ignored, and instead the batch size is adjusted dynamically to target a fixed number of tokens per batch, per device. This was easier to implement than sample packing, and does basically the same thing. It is also efficient: if I set batch_size_tokens to a modest 10000 and train a 7B model with the Alpaca dataset, all my 4090s hit their 350W power limit cap. Unless I'm missing something (definitely possible), it seems there is no need to support sample packing.

## Changelog
### 2024-05-19
**The old config file format will break.** Quantization is configured slightly differently now. Read examples/config_7b.toml. It's only a few lines to change.
- Change how quantization is configured. Quantization is now its own table in the TOML file.
- Add HQQ quantization.
### 2024-04-28
- Add llama3 instruction formatting option when loading a ShareGPT formatted dataset using Axolotl.
- Automatically add BOS token for Llama 3.
- Add option for Unsloth activation checkpointing, which saves VRAM for a very small hit to performance.
### 2024-04-16
- Optimizer is now specified in the config.toml file.
- Can use AdamW8Bit optimizer.
- MLP offloading works again. For MoE, can offload a specified number of experts.
- Can have separate dtype for saved files.
- Cohere model support (command-r)
### 2024-04-07
Make sure to update requirements! Axolotl does some dynamic importing, so things will break in a very hard to diagnose way if you don't have a new dependency that was added.
- Removed the need for manually specifying cache directories for datasets. All dataset processing uses the Huggingface Datasets library and takes advantage of the automatic caching that it provides.
- Added the ability to specify multiple datasets, with different ways to combine them. __This breaks the old config format for datasets.__ Refer to the example config for what it should look like now.