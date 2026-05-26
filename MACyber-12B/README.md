---
language:
- en
library_name: transformers
pipeline_tag: text-generation
tags:
- cybersecurity
- threat-intelligence
- structured-data
- security-operations
- retrieval-augmented-generation
- lora
- gemma3
---

# MACyber-12B

MACyber-12B is an adaptive 12.19B-parameter cybersecurity language model designed to bridge security data silos across heterogeneous, structured data sources. Rather than treating cybersecurity as decontextualized question answering, it analyzes operational records from multiple security domains and produces unified, evidence-grounded outputs for anomaly detection, situation assessment, reasoning, and response recommendation.

MACyber-12B combines a dual-channel threat-intelligence retrieval mechanism with **CyberLoRA**, which supports single-step rapid adaptation for previously unseen threats through LoRA-based adaptation, HyperLoRA, and a hash-fingerprint threat vector library. On the MACyber-INT benchmark, the paper reports that MACyber-12B outperforms the average score of 13 LLM baselines by **21.42%** across four cybersecurity tasks. For unknown threats, CyberLoRA provides an average **3.18-point** gain with **23.2 ms** average adaptation latency.

MACyber-12B is developed together with **MACyber-INT**, a multi-source aligned cybersecurity intelligence benchmark spanning seven domains and more than 300 cyber threats from a 1.4 TB cybersecurity corpus. The released checkpoint uses the `Gemma3ForConditionalGeneration` architecture, while the released evaluation workflow targets serialized structured-security text records.

## Model Details

| Item | Description |
| --- | --- |
| Model name | MACyber-12B |
| Model type | Cybersecurity-specialized multimodal generative model used for structured-text analysis |
| Parameters | 12,187,325,040 |
| Architecture | `Gemma3ForConditionalGeneration` |
| Processor | `Gemma3Processor` |
| Weight dtype | `bfloat16` |
| Configured maximum context length | 131,072 tokens |
| Vision input configuration | 896 x 896 pixels, 256 image tokens |
| Primary language | English |
| Task | Structured security analysis and response generation |
| Framework | Transformers |
| Repository | [MACyber](https://github.com/qcydm/MACyber) |
| Model Hub | [qcy98/MACyber-12B](https://huggingface.co/qcy98/MACyber-12B) |
| Paper | *Bridging Cybersecurity Data Silos: From Multi-Source Unified Intelligence Benchmark to Adaptive Cybersecurity LLM* |

The configuration confirms a Gemma3 12B-class architecture. The exact public upstream checkpoint identifier and applicable weight redistribution license should be confirmed before publishing model weights on the Hugging Face Hub.

## Intended Uses

MACyber-12B is intended for research on:

- Security telemetry and threat-intelligence record analysis.
- Structured threat classification, severity assessment, and response recommendation.
- Retrieval-augmented cybersecurity situational awareness.
- Evaluation on the MACyber-INT benchmark.

It is not intended to autonomously block traffic, quarantine assets, or make production incident-response decisions without human review.

## Output Format

The model is designed to generate a structured assessment in the following form:

```json
{
  "evidence": [
    "feature = value (security interpretation)"
  ],
  "analysis": "First check ...; then verify ...; finally confirm ...; because ..., classify as ....",
  "action": "none | monitor | block",
  "official": "threat label",
  "severity": "benign | suspicious | low | medium | high"
}
```

## Threat-Intelligence RAG

The MACyber repository provides an optional retrieval-augmented generation module under `Threat Intelligence RAG/`:

- **Known-attack channel:** retrieves reference cases from a curated knowledge base when the sample belongs to a known dataset or attack domain.
- **Unknown-attack channel:** retrieves similar reference cases for samples outside the known-attack mapping using feature-based similarity search.

When enabled in the generation pipeline, retrieved references are appended to the system prompt as contextual examples before inference.

## CyberLoRA

The repository additionally provides CyberLoRA experimental assets for lightweight cybersecurity adaptation, including a single-step adaptation implementation and a LoRA artifact. The available LoRA training record reports:

| Hyperparameter | Value |
| --- | --- |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Learning rate | 2e-4 |
| Epochs | 1 |
| Per-device training batch size | 1 |
| Gradient accumulation steps | 16 |
| Scheduler | Cosine with 0.1 warmup ratio |

These artifacts document the adaptation workflow; users should verify which checkpoint files are included in a specific Hub release before loading an adapter or merged model.

## Configuration

The exported model configuration specifies:

| Setting | Value |
| --- | --- |
| Transformers version recorded in config | 4.57.6 |
| Text hidden size | 3,840 |
| Text layers | 48 |
| Text attention heads / KV heads | 16 / 8 |
| Text vocabulary size | 262,208 |
| Sliding attention window | 1,024 |
| Default sampling | `do_sample=True`, `top_k=64`, `top_p=0.95` |
| EOS token IDs | `1`, `106` |

The exported artifacts also include an Ollama `Modelfile` generated by LLaMA-Factory with a default runtime context setting of `4096`.

## Usage

```bash
pip install vllm "transformers>=4.57.6"
```

```python
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "qcy98/MACyber-12B"

llm = LLM(
    model=model_id,
    trust_remote_code=True,
    gpu_memory_utilization=0.9,
    tensor_parallel_size=1,
    max_model_len=8192,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

record = {
    "meta": {"category": "dns", "subcategory": "dns-doh"},
    "json": {"example_feature": "example_value"}
}

system_prompt = """You are an experienced cybersecurity expert. Analyze the input
record and output only valid JSON with the fields evidence, analysis, action,
official, and severity. The action must be block, monitor, or none. The
severity must be benign, suspicious, low, medium, or high."""

user_prompt = (
    "Please analyze the following feature data and output analysis results "
    "in the required JSON format:\n\n"
    + json.dumps(record, ensure_ascii=False, indent=2)
)

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

sampling_params = SamplingParams(
    temperature=0.3,
    top_p=0.9,
    max_tokens=1024,
)

outputs = llm.generate([prompt], sampling_params)
print(outputs[0].outputs[0].text.strip())
```

Although the checkpoint configuration inherits the Gemma3 architecture, the released MACyber evaluation workflow uses vLLM text generation over serialized structured security records.

For benchmark generation with the optional RAG module, use the generation scripts in the MACyber repository:

```bash
python Benchmark_Evaluation/generate_answers_vllm.py \
  --input "MACyber-INT_benchmark/DNS Security Threat/dns-doh.json" \
  --output "Benchmark_Evaluation/outputs/MACyber-12B/DNS Security Threat/dns-doh/model_output.json" \
  --type dns-doh \
  --model-path "/path/to/MACyber-12B" \
  --use-rag \
  --rag-top-k 3
```

In this repository script, `--model-path` must be a local downloaded checkpoint directory.

## Evaluation

MACyber-12B is evaluated with **MACyber-INT**, a structured cybersecurity benchmark containing 31 datasets across seven domains:

- Network Traffic Security
- IoT Security
- System Log Security
- DNS Security Threat
- Web Security Threat
- Vulnerability Intelligence
- Threat Intelligence

The MACyber evaluator assesses four components of each answer:

| Component | Weight |
| --- | ---: |
| Reasoning | 40% |
| Official label | 30% |
| Response action | 20% |
| Severity | 10% |

Benchmark data, evaluation scripts, and reported result tables are available in the [MACyber repository](https://github.com/qcydm/MACyber).

## Limitations

- Model outputs may contain incorrect classifications, severity assignments, unsupported evidence, or inappropriate response recommendations.
- Retrieval-augmented outputs depend on the coverage and quality of the retrieved reference records.
- Benchmark performance does not establish reliability for real-world incident response or threat hunting.
- The model should be used with analyst review in operational security workflows.

## Ethical and Safety Considerations

MACyber-12B is intended for defensive cybersecurity research and analysis. Users are responsible for validating generated assessments, avoiding disclosure of sensitive operational data, and complying with applicable law and organizational policy.

## Citation

Please cite the associated project paper:

> Chenyang Qiu, Boyuan Wang, Guoshun Nan, Deyu Meng, Tongchuan Xia, Danchen Guan, Yilin Peng, Chenyang Wang, and Xiaofeng Tao. *Bridging Cybersecurity Data Silos: From Multi-Source Unified Intelligence Benchmark to Adaptive Cybersecurity LLM.*

## License

The release license for MACyber-12B model weights must be specified consistently with the confirmed upstream base model license and the terms applicable to any included adaptation artifacts before public distribution. This model card intentionally does not assert a Hub license field until that verification is complete.
