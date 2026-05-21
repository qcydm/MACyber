"""Convert raw CSV records into the MACyber JSON schema with an OpenAI-compatible API."""
from __future__ import annotations

import asyncio
import json
import random
import os
import re
from pathlib import Path

import argparse

# ------------------------------------------------

import pandas as pd
from tqdm.asyncio import tqdm_asyncio

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "")
TIMEOUT = 300
CONCURRENCY = 20
RANDOM_SEED = 42
NUM_PRE_REQUEST = 5

# Runtime configuration populated from CLI arguments.
category = ""
DATASET = ""
LABEL = ""
EXAMPLE_DIR = None
EXAMPLES = []

def load_examples() -> list[dict]:
    """
    Loads few-shot examples from a JSON file in the example directory.
    Assumes there is one .json file in the directory.
    """
    json_files = list(EXAMPLE_DIR.glob('*.json'))
    if not json_files:
        print(f"--- WARNING: No example JSON file found in {EXAMPLE_DIR}. No few-shot examples will be used.")
        return []
    
    example_file = json_files[0]
    print(f"Loading examples from: {example_file}")
    try:
        with open(example_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"--- ERROR: Could not read or parse example file {example_file}: {e}")
        return []

def build_prompt(raw_str: str, attack_type: str) -> str:
    """
    Builds the prompt for the model API, including dynamically selected few-shot examples.
    """
    clean_str = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', raw_str)

    relevant_examples = [
        ex for ex in EXAMPLES 
        if ex.get("label", {}).get("official") == attack_type
    ]
    
    if relevant_examples:
        formatted_examples = ",\n".join(json.dumps(ex, indent=2) for ex in relevant_examples[:5])
    else:
        formatted_examples = "No examples provided, follow strict schema."
    
    prompt = f"""You are an expert security analyst bot. Your sole task is to analyze network traffic and respond with a single, valid JSON array. For each input instance, create one JSON object that strictly follows this nested structure and instructions.
---
SCHEMA & INSTRUCTIONS ---
Your output must be a JSON array `[...]`. Each object inside the array must represent an analysis result and STRICTLY adhere to the following JSON structure. The values below describe what you should output for each key:
[
  {{
    "meta": {{ "category": "{category}", "subcategory": "{DATASET}" }},
    "json": "The original data for this record, as a JSON object, after removing the label, {LABEL} keys. Do not put a string here, but the actual JSON object.",
    "label": {{
      "official": "Copy the exact value from the original {LABEL} key here. This is the ONLY place the value of {LABEL} key should appear.",
      "severity": "Choose EXACTLY ONE of: 'benign', 'suspicious', 'low', 'medium', 'high'.",
      "severity_code": "The integer code for the severity: benign=0, suspicious=1, low=2, medium=3, high=4.",
      "confidence": "Your confidence (float from 0.00 to 1.00) that your analysis is correct."
    }},
    "reasoning": {{
      "evidence": "Provide 3-5 key technical features as an array of strings. Each string MUST follow the format: 'FEATURE_NAME: VALUE(Brief analysis of why this is evidence)'. Do not mention the values of the label, {LABEL} keys in the analysis.",
      "analysis": "A step-by-step explanation of how the evidence proves the verdict. MUST follow the format: 'First check <key features>; then verify <key features>; finally confirm <key features>; because <comprehensive judgment reason>, classify as {LABEL}."
    }},
    "response": {{
      "action": "Recommended action. Choose EXACTLY ONE of: 'none', 'monitor', 'block'.",
      "reason": "Briefly justify the action. Do not mention the values of the values of label , {LABEL} keys."
    }}
  }}
]

---
FEW-SHOT EXAMPLES ---
The following are high-quality examples of the expected output format whose value of {LABEL} key is {attack_type}.
[
{formatted_examples}
]

---
IMPORTANT RULES (RECAP) ---
1. Your entire output MUST be a single, valid JSON array. No extra text or wrappers.
2. The structure MUST be nested as described above.
3. The values of the label, {LABEL} and subLabelCat keys MUST NOT appear in json, reasoning or response.
4. Do NOT use Markdown code blocks (do not wrap them with ' ' '), and do not include any explanatory text or preamble.
5. Note that there should be a distinction in the severity of several pieces of data in each group, and they should not all be the same. BUT if the value of the {LABEL} is benign, the severity must be 'benign'. 

---
INPUT DATA ---
Note that a value of 0 for the label key indicates benign traffic, while a value of 1 indicates malicious attack traffic.
Note that subsubLabelCat key denotes specific attack techniques.
Analyze the following {NUM_PRE_REQUEST} instances of the '{attack_type}' type and return the result as a valid JSON array.
{clean_str}
"""
    return prompt

def clean_json_string(s: str) -> str:
    """
    清洗模型返回的字符串，去除 markdown 代码块标记 (```json ... ```)
    """
    if not s:
        return ""
    # 去除 ```json 或 ``` 开头
    s = re.sub(r"^```(json)?", "", s.strip(), flags=re.IGNORECASE | re.MULTILINE)
    # 去除 ``` 结尾
    s = re.sub(r"```$", "", s.strip(), flags=re.MULTILINE)
    s = s.replace('\xa0', ' ').strip()
    return s.strip()

def call_model_sync(prompt: str) -> dict | None:
    """
    Synchronous function to call an OpenAI-compatible chat completion API.
    """
    from openai import OpenAI

    messages = [{'role': 'user', 'content': prompt}]
    try:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        request = {
            "model": MODEL,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 16000,
            "response_format": {"type": "json_object"},
        }
        try:
            response = client.chat.completions.create(**request)
        except Exception as exc:
            if "response_format" not in str(exc):
                raise
            request.pop("response_format", None)
            response = client.chat.completions.create(**request)
        content = response.choices[0].message.content
        cleaned_content = clean_json_string(content)
        try:
            return json.loads(cleaned_content)
        except json.JSONDecodeError:
            print(f"\n--- JSON PARSE ERROR ---")
            raise ValueError(f"JSON Parse Error. \nRaw: {content}\n cleaned: {cleaned_content}")

    except Exception as e:
        # Re-raise the exception to be caught by the async wrapper
        raise e


async def call_model(prompt: str) -> list | dict | None:
    """
    异步包装器。
    修改点：增加了指数退避（Exponential Backoff）策略，
    当遇到服务器拥堵时，等待时间会越来越长，避免频繁撞墙。
    """
    max_retries = 5  # 增加重试次数（原为3）
    base_delay = 5   # 基础等待秒数
    
    for attempt in range(max_retries):
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, call_model_sync, prompt
            )
            return result

        except Exception as e:
            error_str = str(e)
            print(f"\n[Attempt {attempt+1}/{max_retries}] API call failed: {e}")
            
            # 如果是最后一次尝试依然失败，则放弃
            if attempt == max_retries - 1:
                print(f"--- FAILED: Giving up after {max_retries} attempts. ---")
                return None
            
            # --- 智能等待策略 ---
            if "500" in error_str or "timed out" in error_str.lower():
                # 如果是服务器错误，等待时间长一点 (指数退避 + 随机抖动)
                sleep_time = (base_delay * (2 ** attempt)) + random.uniform(1, 5)
                print(f"Server error/timeout detected. Sleeping for {sleep_time:.2f}s...")
                await asyncio.sleep(sleep_time)
            
            elif "429" in error_str or "rate limit" in error_str.lower():
                # 如果是限流，等待时间更长
                sleep_time = 20 + random.uniform(1, 5)
                print(f"Rate limit hit. Sleeping for {sleep_time:.2f}s...")
                await asyncio.sleep(sleep_time)
            
            else:
                # 其他错误（如JSON解析失败），稍微等一下重试
                await asyncio.sleep(5)
                
    return None


async def process_batch(df_batch: pd.DataFrame) -> list | dict | None:
    """
    Processes a single batch of DataFrame rows.
    """
    if df_batch.empty:
        return None
        
    raw_str = df_batch.to_json(orient='records')
    attack_type = df_batch[LABEL].iloc[0]

    prompt = build_prompt(raw_str, attack_type)
    converted = await call_model(prompt)

    return converted


def parse_sample_counts(raw: str) -> dict:
    """Parse sample counts like '0:20,1:80' into a dictionary."""
    counts = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        key, value = item.split(":", 1)
        try:
            key = int(key)
        except ValueError:
            key = key.strip()
        counts[key] = int(value)
    return counts


def parse_args():
    parser = argparse.ArgumentParser(description="Convert raw CSV rows to MACyber JSON with an OpenAI-compatible API.")
    parser.add_argument("--input-csv", required=True, help="Input CSV file.")
    parser.add_argument("--output-json", required=True, help="Output JSON file.")
    parser.add_argument("--category", required=True, help="MACyber category name.")
    parser.add_argument("--dataset", required=True, help="Dataset/subcategory name.")
    parser.add_argument("--label", default="Target", help="Label column in the CSV.")
    parser.add_argument("--example-dir", default=None, help="Directory containing few-shot example JSON files.")
    parser.add_argument("--sample-counts", default="0:20,1:80", help="Per-label sample counts, e.g. '0:20,1:80'.")
    parser.add_argument("--batch-size", type=int, default=NUM_PRE_REQUEST, help="Rows per model request.")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY, help="Concurrent API requests.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Sampling random seed.")
    parser.add_argument("--model", default=MODEL, help="OpenAI-compatible model name.")
    parser.add_argument("--base-url", default=BASE_URL, help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default=API_KEY, help="OpenAI-compatible API key.")
    return parser.parse_args()


async def main():
    global category, DATASET, LABEL, EXAMPLE_DIR, EXAMPLES, MODEL, BASE_URL, API_KEY, NUM_PRE_REQUEST, CONCURRENCY

    args = parse_args()
    category = args.category
    DATASET = args.dataset
    LABEL = args.label
    EXAMPLE_DIR = Path(args.example_dir) if args.example_dir else Path()
    EXAMPLES = load_examples() if args.example_dir else []
    MODEL = args.model
    BASE_URL = args.base_url
    API_KEY = args.api_key
    NUM_PRE_REQUEST = args.batch_size
    CONCURRENCY = args.concurrency
    target_sample_counts = parse_sample_counts(args.sample_counts)

    if not args.api_key:
        raise RuntimeError("API key is required. Set OPENAI_API_KEY or pass --api-key.")
    
    input_csv_file = Path(args.input_csv)
    output_json_file = Path(args.output_json)

    if args.example_dir and not EXAMPLE_DIR.exists():
        print(f"--- WARNING: example 目录不存在: {EXAMPLE_DIR}. No few-shot examples will be used.")
    assert input_csv_file.exists(), f"输入文件 {input_csv_file} 不存在"
    output_json_file.parent.mkdir(parents=True, exist_ok=True)

    print("正在读取CSV文件...")
    df = pd.read_csv(input_csv_file)

    print(f"正在按指定数量抽样: {target_sample_counts}")
    sampled_parts = []
    for label_value, sample_count in target_sample_counts.items():
        group_df = df[df[LABEL] == label_value]
        if group_df.empty:
            print(f"--- WARNING: label {label_value} 没有数据，跳过。")
            continue
        sampled_parts.append(
            group_df.sample(
                n=min(len(group_df), sample_count),
                random_state=args.seed,
            )
        )
    target_df = pd.concat(sampled_parts, ignore_index=True).copy()
    
    print(f"共 {len(df)} 行。")
    print("抽样结果分布如下：")
    print(target_df[LABEL].value_counts())
    print(f"总计处理 {len(target_df)} 行。")

    all_batches = []
    attack_groups = target_df.groupby(LABEL)
    print("按攻击类型分组创建批次...")

    for attack_name, group_df in attack_groups:
        print(f"\n正在处理攻击类型: {attack_name} (共 {len(group_df)} 条)")

        for i in range(0, len(group_df), NUM_PRE_REQUEST):
            batch = group_df.iloc[i : i + NUM_PRE_REQUEST]
            if not batch.empty:
                all_batches.append(batch)
    
    num_total_batches = len(all_batches)
    print(f"将数据分为 {num_total_batches} 个小批次 (每批 {NUM_PRE_REQUEST} 行)。")
    print(f"启用并发模式：每次同时发送 {CONCURRENCY} 个请求。")

    with open(output_json_file, 'w', encoding='utf-8') as outfile:
        outfile.write('[\n')
        is_first_item = True
        progress_bar = tqdm_asyncio(total=num_total_batches, desc="Converting (Concurrent)")

        for i in range(0, num_total_batches, CONCURRENCY):
            current_batch_group = all_batches[i : i + CONCURRENCY]
            
            tasks = [process_batch(batch_df) for batch_df in current_batch_group]
            
            try:
                results_group = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result_or_exc in results_group:
                    if isinstance(result_or_exc, Exception):
                        print(f"\n--- ERROR in Concurrent Task ---")
                        print(f"错误详情: {result_or_exc}")
                        continue
                    
                    results = result_or_exc
                    if results and isinstance(results, list):
                        for result in results:
                            if isinstance(result, dict):
                                output_string = json.dumps(result, ensure_ascii=False, indent=2)
                                if is_first_item:
                                    outfile.write(output_string)
                                    is_first_item = False
                                else:
                                    outfile.write(',\n' + output_string)
                        outfile.flush()
                    elif results and isinstance(results, dict):
                        output_string = json.dumps(results, ensure_ascii=False, indent=2)
                        if is_first_item:
                            outfile.write(output_string)
                            is_first_item = False
                        else:
                            outfile.write(',\n' + output_string)
                        outfile.flush()

                progress_bar.update(len(current_batch_group))
                
                await asyncio.sleep(10) 

            except Exception as e:
                print(f"并发组处理发生严重错误: {e}")
        
        progress_bar.close()
        outfile.write('\n]\n')

    print(f"\n处理完成。数据已写入 {output_json_file}")


if __name__ == "__main__":
    asyncio.run(main())
