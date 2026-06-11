import json
import os
import glob
import re
import random

# ================= 配置区 =================
INPUT_DIR = "./"  # 读取当前目录下的 json
OUTPUT_DIR = "./gemma_converted/" # 转换后的文件存放目录

# 针对 Gemma 3 优化的 Instruction 结尾：增加了 evidence 和 analysis
NEW_OUTPUT_FORMAT = """### Output Format
You must first provide a detailed analysis of the given metrics in plain text. Explain your reasoning step-by-step.
After your analysis, output the final determination strictly enclosed within a markdown JSON block, like this:

```json
[
  {
    "evidence": [
      "Feature_name = Value (reasoning for this feature)"
    ],
    "analysis": "Your step-by-step logical reasoning process.",
    "action": "One of: block,monitor,none",
    "official": "The specific attack type/label",
    "severity": "One of: high,medium,low,benign,suspicious"
  }
]
```"""

# 适配 Gemma 的多样化分析引导头（使用标准的 Markdown 粗体或标题）
ANALYSIS_TEMPLATES = [
    "**Analysis Process:**\nLet's analyze the data step by step. We can observe {evidence_text}. {analysis_text} Given this assessment, the threat is identified as {official} with a severity level of {severity}. The recommended action is to {action}.",
    
    "**Investigation Summary:**\nBased on the input telemetry, key findings include: {evidence_text}. To elaborate: {analysis_text} Therefore, I classify this as {official} (Severity: {severity}, Action: {action}).",
    
    "**Security Assessment:**\nReviewing the security indicators, I note {evidence_text}. My reasoning process is as follows: {analysis_text} This aligns perfectly with a {official} classification. We must take a '{action}' action due to its {severity} nature.",
    
    "**Threat Correlation:**\nInitiating threat analysis. The evidence shows {evidence_text}. Breaking this down: {analysis_text} Concluding the analysis, the signature matches {official}, warranting a {severity} severity rating and a {action} response."
]

def parse_evidence(evidence_list):
    """解析 evidence 列表，提取特征和括号内的原因"""
    parsed_items = []
    for ev in evidence_list:
        match = re.match(r'^(.*?)(?:\s*\((.*?)\))?$', str(ev).strip())
        if match:
            feature = match.group(1).strip()
            reason = match.group(2).strip() if match.group(2) else "anomalous behavior detected"
            parsed_items.append(f"that [{feature}] indicates [{reason}]")
    
    if not parsed_items:
        return "no obvious anomaly signatures"
    
    return ", and ".join(parsed_items)

def process_single_item(item):
    """处理单条数据"""
    try:
        # 1. 解析原始 output 的 JSON 字符串
        original_output = json.loads(item["output"])[0]
        
        # 提取各个字段
        raw_evidences = original_output.get("evidence", [])
        analysis = original_output.get("analysis", "Further correlation is required.")
        official = original_output.get("official", "Unknown")
        severity = original_output.get("severity", "benign")
        action = original_output.get("action", "none")
        
        # 2. 清洗 evidence，将特征部分的 ":" 替换为 "="
        cleaned_evidences = []
        for ev in raw_evidences:
            match = re.match(r'^(.*?)(?:\s*\((.*?)\))?$', str(ev).strip())
            if match:
                feature = match.group(1).strip()
                # 使用正则把 "Key: Value" 或 "Key : Value" 替换为 "Key = Value"
                feature = re.sub(r'\s*:\s*', ' = ', feature)
                reason = match.group(2).strip() if match.group(2) else ""
                
                if reason:
                    cleaned_evidences.append(f"{feature} ({reason})")
                else:
                    cleaned_evidences.append(feature)
            else:
                cleaned_evidences.append(str(ev))

        # 3. 将清洗后的 evidence 解析为自然语言
        evidence_text = parse_evidence(cleaned_evidences)
        
        # 4. 随机选择一个模板，生成自然语言分析段落
        template = random.choice(ANALYSIS_TEMPLATES)
        analysis_block = template.format(
            evidence_text=evidence_text,
            analysis_text=analysis.capitalize(),
            official=official,
            severity=severity,
            action=action
        )
        
        # 5. 构建严格的、用于提取的 JSON 目标对象 (现在包含所有要求字段)
        final_json_obj = [
            {
                "evidence": cleaned_evidences,
                "analysis": analysis,
                "action": action,
                "official": official,
                "severity": severity
            }
        ]
        # 确保格式严格正确，缩进为2
        final_json_str = json.dumps(final_json_obj, ensure_ascii=False, indent=2)
        
        # 6. 组合最终的 output：自然语言分析 + Markdown JSON Block
        new_output_str = f"{analysis_block}\n\n```json\n{final_json_str}\n```"
        
        # 7. 替换 instruction
        new_instruction = re.sub(
            r'### Output Format.*', 
            NEW_OUTPUT_FORMAT, 
            item["instruction"], 
            flags=re.DOTALL
        )
        
        return {
            "instruction": new_instruction,
            "input": item["input"],
            "output": new_output_str
        }
    except Exception as e:
        print(f"  ❌ Error processing item: {e}")
        return None

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    json_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
    
    if not json_files:
        print("当前目录下未找到 .json 文件！")
        return
        
    total_processed = 0
    total_failed = 0
    
    for file_path in json_files:
        filename = os.path.basename(file_path)
        print(f"正在处理: {filename} ...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"  ❌ 文件 {filename} 不是有效的 JSON，跳过。")
                continue
                
        new_data = []
        for item in data:
            if isinstance(item, dict) and "output" in item and "instruction" in item:
                processed = process_single_item(item)
                if processed:
                    new_data.append(processed)
                else:
                    total_failed += 1
            else:
                new_data.append(item) 
                
        out_path = os.path.join(OUTPUT_DIR, filename)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)
            
        total_processed += len(new_data)
        print(f"  ✅ 成功保存至: {out_path} (有效数据: {len(new_data)} 条)")

    print("-" * 40)
    print(f"🎉 全部处理完成！共处理有效数据 {total_processed} 条，失败/跳过 {total_failed} 条。")

if __name__ == "__main__":
    main()