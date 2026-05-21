# MACyber

MACyber is a cybersecurity benchmark and evaluation toolkit for structured security intelligence data. It provides the MACyber-INT benchmark, construction and evaluation scripts, finalized paper results, and the threat-intelligence RAG resources used by MACyber-12B.

## Repository Layout

```text
MACyber/
├── MACyber-INT_benchmark/        # Benchmark datasets grouped by domain
│   ├── Network Traffic Security/
│   ├── IoT Security/
│   ├── System Log Security/
│   ├── DNS Security Threat/
│   ├── Web Security Threat/
│   ├── Vulnerability Intelligence/
│   ├── Threat Intelligence/
│   ├── taxonomy.json
│   └── dataset_name_mapping.json
├── MACyber-12B/
│   ├── CyberLoRA/
│   └── Threat Intelligence RAG/   # Known/unknown attack retrieval resources
├── Benchmark_Construction/       # Raw data conversion and schema validation
├── Benchmark_Evaluation/         # Generation and evaluation scripts
├── Evaluation_Results/           # Final result tables used in the paper
├── requirements.txt
└── README.md
```

## Benchmark Data

The benchmark contains 31 datasets organized into seven high-level domains:

- Network Traffic Security
- IoT Security
- System Log Security
- DNS Security Threat
- Web Security Threat
- Vulnerability Intelligence
- Threat Intelligence

Each benchmark file is a JSON list stored as:

```text
MACyber-INT_benchmark/<domain>/<dataset>.json
```

The taxonomy is available at:

```text
MACyber-INT_benchmark/taxonomy.json
```

Formal paper dataset names and their corresponding internal `DATASET_THREAT_TYPES` keys are listed in:

```text
MACyber-INT_benchmark/dataset_name_mapping.json
```

## Data Schema

Each sample follows the MACyber schema:

```json
{
  "meta": {
    "category": "domain name",
    "subcategory": "dataset name"
  },
  "json": {
    "feature_name": "feature value"
  },
  "label": {
    "official": "threat label",
    "severity": "benign | suspicious | low | medium | high"
  },
  "reasoning": {
    "evidence": ["feature = value (security interpretation)"],
    "analysis": "First check ...; then verify ...; finally confirm ...; because ..., classify as ...."
  },
  "response": {
    "action": "none | monitor | block",
    "reason": "brief action rationale"
  }
}
```

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/qcydm/MACyber.git
cd MACyber
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For local vLLM inference, use a CUDA-enabled Linux environment compatible with `vllm` and your model checkpoint.

## API Configuration

Model answer generation uses an OpenAI-compatible chat completion API:

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-5.5"
```

Automatic judging uses DashScope Qwen3-Max by default:

```bash
export DASHSCOPE_API_KEY="your_dashscope_api_key"
export DASHSCOPE_MODEL="qwen3-max"
```

## Quick Evaluation

Generate answers for one dataset:

```bash
python Benchmark_Evaluation/generate_answers.py \
  --input "MACyber-INT_benchmark/DNS Security Threat/dns-doh.json" \
  --output "Benchmark_Evaluation/outputs/MyModel/DNS Security Threat/dns-doh/model_output.json" \
  --type dns-doh \
  --api-model "$OPENAI_MODEL" \
  --base-url "$OPENAI_BASE_URL" \
  --api-key "$OPENAI_API_KEY"
```

Evaluate generated answers:

```bash
python Benchmark_Evaluation/evaluate_model.py \
  --standard "MACyber-INT_benchmark/DNS Security Threat/dns-doh.json" \
  --model "Benchmark_Evaluation/outputs/MyModel/DNS Security Threat/dns-doh/model_output.json" \
  --output "Benchmark_Evaluation/outputs/MyModel/DNS Security Threat/dns-doh/eval_result.json" \
  --judge-model qwen3-max \
  --api-key "$DASHSCOPE_API_KEY"
```

Run batch evaluation:

```bash
python Benchmark_Evaluation/batch_eval.py \
  --model MyModel \
  --api-model "$OPENAI_MODEL" \
  --base-url "$OPENAI_BASE_URL" \
  --api-key "$OPENAI_API_KEY" \
  --judge-api-key "$DASHSCOPE_API_KEY" \
  --judge-model qwen3-max
```

Use `--tiny N` for a small debugging subset and `--datasets ...` to restrict evaluation to selected datasets.

## Threat-Intelligence RAG

Generation scripts support a threat-intelligence RAG mode:

```bash
python Benchmark_Evaluation/generate_answers.py \
  --input "MACyber-INT_benchmark/IoT Security/CIC-BCCC-NRC2024.json" \
  --output "Benchmark_Evaluation/outputs/MyModel/IoT Security/CIC-BCCC-NRC2024/model_output.json" \
  --type CIC-BCCC-NRC2024 \
  --use-rag \
  --rag-top-k 3
```

The RAG router receives the full current sample. If `meta.subcategory` exists in the known-attack knowledge base, it uses the known-attack channel. Otherwise, it uses the unknown-attack similarity channel.

```text
MACyber-12B/Threat Intelligence RAG/
├── known_attack_channel/
│   ├── known_attack_RAG.json
│   └── known-attack_retrieval_augmented.py
└── unknown_attack_channel/
    ├── unknown-attack_retrieval_augmented.py
    ├── known_attack_result_key_Z.npy
    ├── known_attack_result_key_fams.json
    └── known_attack_result_key_raw.json
```

## Benchmark Construction

Convert raw CSV records into the MACyber JSON schema:

```bash
python Benchmark_Construction/convert.py \
  --input-csv /path/to/raw.csv \
  --output-json /path/to/output.json \
  --category "Network Traffic Security" \
  --dataset "ExampleDataset" \
  --label Label
```

Validate benchmark files:

```bash
python Benchmark_Construction/validate_schema.py --data-dir MACyber-INT_benchmark
```

## Scoring

`Benchmark_Evaluation/evaluate_model.py` computes a weighted score over four dimensions:

```text
reasoning: 40%
official:  30%
action:    20%
severity:  10%
```

`official` and `action` are exact-match fields. `severity` is scored by severity-level distance. `reasoning` is judged by DashScope `qwen3-max` using the reference evidence and analysis.

## Results

`Evaluation_Results/` contains the finalized Excel tables used in the paper. Newly generated outputs are written under:

```text
Benchmark_Evaluation/outputs/
```

See `Benchmark_Evaluation/README.md` for detailed evaluation usage.
