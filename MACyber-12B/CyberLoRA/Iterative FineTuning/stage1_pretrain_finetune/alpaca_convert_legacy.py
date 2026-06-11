import json
import os
import sys

# ================= 0. 输入配置：待处理文件列表 =================

# 请在这里维护需要转化的原始数据JSON的绝对路径列表
dns_list = [
# dns
"/data2/qcy/data_pattern/dns/converted/train/bccc_2024_mal_20251016_003244_train.json",
"/data2/qcy/data_pattern/dns/converted/train/DNS_exfiltration_benign_1000_train.json",
"/data2/qcy/data_pattern/dns/converted/train/CIRA_benign_1500_train.json",
"/data2/qcy/data_pattern/dns/converted/train/DNS_exfiltration_light_1000_train.json",
"/data2/qcy/data_pattern/dns/converted/train/CIRA_malicious_3500_train.json",
"/data2/qcy/data_pattern/dns/converted/train/DNS_exfiltration_heavy_1000_train.json",
"/data2/qcy/data_pattern/dns/converted/train/bccc_2024_exf_20251016_103047_train.json"
]


#log
log_list = ['/data2/qcy/data_pattern/log/split-anroid/training-anroid/output_data.json',
            '/data2/qcy/data_pattern/log/split-hdfs/training-hdfs-5007.json',
            '/data2/qcy/data_pattern/log/split-linux/training-linux/output_data.json',
            '/data2/qcy/data_pattern/log/split-proxifier/test-proxifier/output_data.json',
            '/data2/qcy/data_pattern/log/split-supercomputer/test-supercom-22/output_data.json'
            ]

#threat
threat_list = ['/data2/qcy/sft_workflow/threat/Pulsedive-Threats/training/Pulsedive-Threats.json',
               '/data2/qcy/data_pattern/threat/SABU-Alert/converted/SABU-Alert_train_no_answer.json']

#traffic&iot
traffic_and_iot_list = [
    "/data2/qcy/data_pattern/traffic_and_iot/train/NF-CSE-CIC-IDS2018-v2/NF-CSE-CIC-IDS2018-v2.json",
    "/data2/qcy/data_pattern/traffic_and_iot/train/CIC-ToN-IoT/CIC-ToN-IoT.json",
    "/data2/qcy/data_pattern/traffic_and_iot/train/CICIoMT 2024/CICIoMT 2024.json",
    "/data2/qcy/data_pattern/traffic_and_iot/train/CICADA-IIoT2024/CICADA-IIoT2024.json",
    "/data2/qcy/data_pattern/traffic_and_iot/train/CIC-IoT-DIAD2024/CIC-IoT-DIAD2024.json",
    "/data2/qcy/data_pattern/traffic_and_iot/train/CIC-IDS-2017/CIC-IDS-2017.json",
    "/data2/qcy/data_pattern/traffic_and_iot/train/CIC_IOT_2023/CIC_IOT_2023.json",
    "/data2/qcy/data_pattern/traffic_and_iot/train/NF-UNSW-NB15-v2/NF-UNSW-NB15-v2.json",
    "/data2/qcy/data_pattern/traffic_and_iot/train/CIC-BCCC-NRC2024/CIC-BCCC-NRC2024.json",
    "/data2/qcy/data_pattern/traffic_and_iot/train/NF-ToN-IoT-v2/NF-ToN-IoT-v2.json",
    "/data2/qcy/data_pattern/traffic_and_iot/train/iscxids2012/iscxids2012.json",
    "/data2/qcy/data_pattern/traffic_and_iot/train/CIC-BoT-IoT/CIC-BoT-IoT.json",
    "/data2/qcy/data_pattern/traffic_and_iot/train/NF-BoT-IoT-v2/NF-BoT-IoT-v2.json",
    "/data2/qcy/data_pattern/traffic_and_iot/train/NF-UQ-NIDS-v2/NF-UQ-NIDS-v2.json",
    "/data2/qcy/data_pattern/traffic_and_iot/train/CICEVSE2024/CICEVSE2024.json"
]

#url
url_list = []

# vulnerablity
vulnerablity_list = ['/data2/qcy/data_pattern/vulnerablity/aliyun/converted/training/output_data.json',
                     '/data2/qcy/data_pattern/vulnerablity/exploit-db/converted/training/output_data.json',
                     '/data2/qcy/data_pattern/vulnerablity/talos/converted/training/output_data.json']

SOURCE_FILES = None

# 定义用于替换的路径片段 (将 data_pattern 替换为 data_pattern_alpaca)
PATH_SOURCE_KEY = "/data_pattern/"
PATH_TARGET_KEY = "/data_pattern_alpaca/"

# ================= 1. 全局配置：专家人设与核心能力 =================

CORE_EXPERTISE = [
    "DNS Forensics & Resolution Analysis",
    "IoT Endpoint Security Monitoring",
    "User & Entity Behavior Analytics",
    "Cyber Threat Intelligence Profiling",
    "Network Intrusion Detection System",
    "Malicious Infrastructure & URL Analysis",
    "Vulnerability Assessment & Risk Management"
]

expertise_list_str = "\n".join([f"{i+1}. {item}" for i, item in enumerate(CORE_EXPERTISE)])
GLOBAL_TASK = "to conduct cross-dimensional correlation analysis on multi-source heterogeneous data to uncover hidden security threats"
SEVERITY_LEVELS = ["high", "medium", "low", "benign", "suspicious"]
ACTION = ["block", "monitor", "none"]

# ================= 2. 领域配置：具体数据的上下文映射 =================

DOMAIN_CONTEXT_MAP = {
    "DNS": "analyzing DNS query logs, focusing on abnormal resolution patterns, DGA domains, and tunneling behavior",
    "IoT": "analyzing IoT device network traffic, focusing on abnormal connection rates, flag anomalies, and unauthorized protocols",
    "Log": "analyzing system operation and file upload logs, focusing on sensitive data exfiltration and abnormal user processes",
    "Threat": "analyzing threat intelligence feeds, focusing on IOC correlation, malware families (e.g., Zeus), and C2 infrastructure",
    "Traffic": "analyzing raw network flow (PCAP/Flow) features, focusing on packet length statistics, inter-arrival times, and flow duration",
    "URL": "analyzing URL and web service details, focusing on botnet C2 patterns, malicious IP reputation, and hosting infrastructure",
    "Vulnerability": "analyzing vulnerability reports (CVE), focusing on exploit maturity, attack vectors, and potential impact on confidentiality"
}

# ================= 3. 核心 Prompt 生成函数 =================

def get_instruction_text(mate_field_str):
    current_context = DOMAIN_CONTEXT_MAP.get(mate_field_str, f"analyzing {mate_field_str} related security data")
    return f"""You are a Multi-source Heterogeneous Cybersecurity Expert. You possess deep proficiency in the following 7 core domains:
{expertise_list_str}

Your core strength lies in {GLOBAL_TASK}. You excel at identifying attack chains that span across network traffic, endpoint logs, and threat intelligence.

### Current Analysis Context
While you are an expert in all the above fields, for this specific task, you are focusing on **{mate_field_str}**:
> Context: You are currently {current_context}.

### Output Format
You must output the result strictly in the following JSON format:
[
  {{
    "evidence": ["metric_observation_1", "metric_observation_2"], 
    "analysis": "First, analyze the specific metrics in the {mate_field_str} data; Then, correlate with your general security knowledge; Finally, conclude the threat type...",
    "official": "The specific attack type/label",
    "severity": "One of: {",".join(SEVERITY_LEVELS)}",
    "action": "One of: {",".join(ACTION)}"
  }}
]
"""

def clean_key_name(key):
    if "consecutie" in key:
        return key.replace("consecutie", "consecutive")
    return key

def normalize_severity(sev_str):
    if not sev_str or not isinstance(sev_str, str):
        return "benign"
    if sev_str.lower() in [s.lower() for s in SEVERITY_LEVELS]:
        return sev_str
    return "benign"

# ================= 核心处理逻辑 =================

def process_file(file_path):
    """
    读取文件，转换数据，并写入到新的对应目录中。
    返回: 成功写入的文件绝对路径 (str) 或 None (失败)
    """
    # 1. 检查输入文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        return None

    # 2. 计算输出路径
    # 逻辑：将路径中的 /data_pattern/ 替换为 /data_pattern_alpaca/
    if PATH_SOURCE_KEY in file_path:
        output_file_path = file_path.replace(PATH_SOURCE_KEY, PATH_TARGET_KEY)
    else:
        # 如果路径中不包含关键字，为了安全起见，输出到当前目录的 alpaca_out 文件夹下
        print(f"   ⚠️ Warning: Path keyword '{PATH_SOURCE_KEY}' not found in path. Saving to ./alpaca_fallback/")
        filename = os.path.basename(file_path)
        output_file_path = os.path.join(os.getcwd(), "alpaca_fallback", filename)

    # 确保输出目录存在
    output_dir = os.path.dirname(output_file_path)
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"   Created directory: {output_dir}")
        except Exception as e:
            print(f"❌ Error creating directory {output_dir}: {e}")
            return None

    print(f"Processing: {file_path}")
    print(f"   -> To: {output_file_path}")

    # 3. 读取数据
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None
    
    # 动态提取 Meta 信息
    try:
        first_record_meta = data[0].get('meta', {})
        category = first_record_meta.get('category', 'Unknown')
        subcategory = first_record_meta.get('subcategory', 'Unknown')
        mate_field_val = f"{category} {subcategory}"
    except (IndexError, AttributeError):
        mate_field_val = "Network Traffic"
        print(f"   Warning: Could not extract meta info, using default: {mate_field_val}")

    current_instruction = get_instruction_text(mate_field_val)
    processed_data = []

    # 遍历处理记录 (保留了原来的 2000 条限制)
    limit = min(2000, len(data))
    
    for i in range(limit):
        try:
            record_item = data[i]
            source_record = record_item.get('json', {})
            output_record = {key: value for key, value in record_item.items() if key != 'json'}
            
            # --- 健壮性处理 ---
            raw_reasoning = output_record.get('reasoning', {})
            if not isinstance(raw_reasoning, dict):
                reasoning = {"evidence": [], "analysis": str(raw_reasoning)}
            else:
                reasoning = raw_reasoning

            raw_response = output_record.get('response', {})
            if not isinstance(raw_response, dict):
                response = {"action": str(raw_response)}
            else:
                response = raw_response

            raw_label = output_record.get('label', {})
            if not isinstance(raw_label, dict):
                label = {"official": "None", "severity": "benign"}
            else:
                label = raw_label

            # 构建 Input
            input_lines = []
            if isinstance(source_record, dict):
                for key, value in source_record.items():
                    if key != "human_label_threat":
                        clean_key = clean_key_name(key)
                        input_lines.append(f"{clean_key}: {value}")
            final_input = "\n".join(input_lines)

            # 构建 Output
            official_raw = label.get('official', "None")
            output_dict = [
                {
                    "evidence": reasoning.get('evidence', []),
                    "analysis": reasoning.get('analysis', "No analysis provided."),
                    "official": official_raw,
                    "severity": normalize_severity(label.get('severity', "benign")),
                    "action": response.get('action', "none")
                }
            ]

            output_str = json.dumps(output_dict, ensure_ascii=False)

            entry = {
                "instruction": current_instruction,
                "input": final_input,
                "output": output_str
            }
            processed_data.append(entry)

        except Exception as e:
            continue

    # 4. 保存文件
    try:
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(processed_data, outfile, indent=2, ensure_ascii=False)
        print(f"✅ Success. Saved {len(processed_data)} records.")
        return output_file_path
    except Exception as e:
        print(f"❌ Error saving output: {e}")
        return None

# ================= 主程序入口 =================

if __name__ == "__main__":
    if not SOURCE_FILES:
        print("Warning: SOURCE_FILES list is empty. Please edit the script to add file paths.")
        sys.exit(0)

    print(f"Starting batch conversion for {len(SOURCE_FILES)} files...\n")
    
    # 用于记录成功生成的文件的路径
    successful_conversions = []

    for file_path in SOURCE_FILES:
        result_path = process_file(file_path)
        if result_path:
            successful_conversions.append(result_path)
        print("-" * 50)

    # 将成功转换的文件路径写入本地 txt 文件，便于后续数据集注册
    manifest_filename = "converted_file_paths.txt"
    manifest_path = os.path.join(os.getcwd(), manifest_filename)
    
    try:
        with open(manifest_path, 'a', encoding='utf-8') as f:
            for path in successful_conversions:
                f.write(path + "\n")
        print(f"\n📄 Manifest saved: {manifest_path}")
        print(f"Total converted files: {len(successful_conversions)}")
    except Exception as e:
        print(f"❌ Error saving manifest file: {e}")

    print("\nAll tasks completed.")
