---
library_name: peft
license: other
base_model: /root/datas/Dynamic-HyperLoRA-Residual/models/cyber-sec-12b
tags:
- base_model:adapter:/root/datas/Dynamic-HyperLoRA-Residual/models/cyber-sec-12b
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: 12B_alpaca_cyber_sec_v4_train
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# 12B_alpaca_cyber_sec_v4_train

This model is a fine-tuned version of `models/cyber-sec-12b` (local Gemma-class 12B base) on the cybersec_merged_random_data dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0002
- train_batch_size: 1
- eval_batch_size: 8
- seed: 42
- distributed_type: multi-GPU
- gradient_accumulation_steps: 16
- total_train_batch_size: 16
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 1

### Training results



### Framework versions

- PEFT 0.18.1
- Transformers 4.57.6
- Pytorch 2.9.1+cu128
- Datasets 4.0.0
- Tokenizers 0.22.2