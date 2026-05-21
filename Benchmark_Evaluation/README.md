# Benchmark Evaluation

This directory contains the scripts used to generate model answers and evaluate them against MACyber-INT benchmark references.

## Files

```text
Benchmark_Evaluation/
├── generate_answers.py       # Generate answers with an OpenAI-compatible chat API
├── generate_answers_vllm.py  # Generate answers with a local vLLM model
├── evaluate_model.py         # Evaluate model outputs with DashScope qwen3-max as the judge
├── batch_eval.py             # Batch generation and evaluation through an OpenAI-compatible API
├── batch_eval_vllm.py        # Batch generation with vLLM and evaluation with DashScope
├── split_datasets.py         # Split constructed benchmark files into train/test subsets
└── outputs/                  # Generated answers and evaluation outputs
```

## Data Layout

The batch scripts expect benchmark files under:

```text
MACyber-INT_benchmark/<category>/<dataset>.json
```

By default, outputs are written to:

```text
Benchmark_Evaluation/outputs/<model>/<category>/<dataset>/
├── model_output.json
└── eval_result.json
```

The repository-level `Evaluation_Results/` directory is reserved for finalized paper results.

## API Configuration

Answer generation uses an OpenAI-compatible chat completion endpoint:

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-5.5"
```

Evaluation uses DashScope with `qwen3-max` as the default judge model:

```bash
export DASHSCOPE_API_KEY="your_dashscope_api_key"
export DASHSCOPE_MODEL="qwen3-max"
```

## Single-Dataset Generation

```bash
python Benchmark_Evaluation/generate_answers.py \
  --input "MACyber-INT_benchmark/DNS Security Threat/dns-doh.json" \
  --output "Benchmark_Evaluation/outputs/MyModel/DNS Security Threat/dns-doh/model_output.json" \
  --type dns-doh \
  --api-model "$OPENAI_MODEL" \
  --base-url "$OPENAI_BASE_URL" \
  --api-key "$OPENAI_API_KEY"
```

## Single-Dataset Evaluation

```bash
python Benchmark_Evaluation/evaluate_model.py \
  --standard "MACyber-INT_benchmark/DNS Security Threat/dns-doh.json" \
  --model "Benchmark_Evaluation/outputs/MyModel/DNS Security Threat/dns-doh/model_output.json" \
  --output "Benchmark_Evaluation/outputs/MyModel/DNS Security Threat/dns-doh/eval_result.json" \
  --judge-model qwen3-max \
  --api-key "$DASHSCOPE_API_KEY"
```

## Threat Intelligence RAG

The generation scripts can prepend threat-intelligence reference examples to the system prompt. The unified RAG helper receives the full current sample object. If `meta.subcategory` exists in the known-attack knowledge base, it uses the known-attack channel; otherwise, it uses the unknown-attack similarity channel.

```text
MACyber-12B/Threat Intelligence RAG/known_attack_channel/known-attack_retrieval_augmented.py
MACyber-12B/Threat Intelligence RAG/known_attack_channel/known_attack_RAG.json
MACyber-12B/Threat Intelligence RAG/unknown_attack_channel/unknown-attack_retrieval_augmented.py
MACyber-12B/Threat Intelligence RAG/unknown_attack_channel/known_attack_result_key_Z.npy
MACyber-12B/Threat Intelligence RAG/unknown_attack_channel/known_attack_result_key_fams.json
MACyber-12B/Threat Intelligence RAG/unknown_attack_channel/known_attack_result_key_raw.json
```

Enable it with:

```bash
python Benchmark_Evaluation/generate_answers.py \
  --input "MACyber-INT_benchmark/IoT Security/CIC-BCCC-NRC2024.json" \
  --output "Benchmark_Evaluation/outputs/MyModel/IoT Security/CIC-BCCC-NRC2024/model_output.json" \
  --type CIC-BCCC-NRC2024 \
  --use-rag \
  --rag-top-k 3
```

## Batch Evaluation with an OpenAI-Compatible API

```bash
python Benchmark_Evaluation/batch_eval.py \
  --model MyModel \
  --api-model "$OPENAI_MODEL" \
  --base-url "$OPENAI_BASE_URL" \
  --api-key "$OPENAI_API_KEY" \
  --judge-api-key "$DASHSCOPE_API_KEY" \
  --judge-model qwen3-max \
  --use-rag \
  --rag-top-k 3
```

Run selected datasets:

```bash
python Benchmark_Evaluation/batch_eval.py \
  --model MyModel \
  --datasets dns-doh CIC-IDS-2017 \
  --api-model "$OPENAI_MODEL" \
  --base-url "$OPENAI_BASE_URL" \
  --api-key "$OPENAI_API_KEY" \
  --judge-api-key "$DASHSCOPE_API_KEY"
```

Use `--tiny N` to run a small random subset for debugging.

## Batch Evaluation with vLLM

```bash
python Benchmark_Evaluation/batch_eval_vllm.py \
  --model LocalModel \
  --model_dir /path/to/local/model \
  --judge-api-key "$DASHSCOPE_API_KEY" \
  --judge-model qwen3-max \
  --use-rag \
  --rag-top-k 3
```

Optional parameters:

```bash
--gpu 0.8        # GPU memory utilization for vLLM
--tiny 10        # debug on a small subset
--datasets ...   # restrict evaluation to selected datasets
--rag-db ...     # override known-attack knowledge base
```

## Scoring

`evaluate_model.py` computes a weighted score over four dimensions:

```text
reasoning: 40%
official:  30%
action:    20%
severity:  10%
```

`official` and `action` are exact-match fields. `severity` is scored by severity-level distance. `reasoning` is judged by DashScope `qwen3-max` using the reference evidence and analysis as the standard answer.
