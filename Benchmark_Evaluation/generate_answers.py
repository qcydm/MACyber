import json
import os
import time
import logging
import threading
import argparse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# OpenAI-compatible API configuration
API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# API调用配置
MAX_RETRIES = 3
RETRY_DELAY = 2
API_DELAY = 0.5
REQUEST_TIMEOUT = 30
MAX_WORKERS = 16  # 并行线程数
SAVE_INTERVAL = 100  # 每处理多少条保存一次

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def call_api_with_retry(messages: list, max_retries: int = MAX_RETRIES):
    """带重试机制的API调用"""
    from openai import OpenAI

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.3,
                top_p=0.9,
                max_tokens=2000
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"API调用异常（尝试{attempt + 1}/{max_retries}）: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY)

    return None

# 数据集名称到威胁类型选项的映射
DATASET_THREAT_TYPES = {
    # Pulsedive-Threats: 威胁数据集
    'Pulsedive-Threats': ['Mirai', 'Ramdo', 'Qadars', 'Bedep', 'Tiny Banker', 'RDP Attack', 'Tor Proxy', 'ZLoader', 'Kraken', 'Vawtrak'],
    # traffic:流量数据集
    'traffic': ['Analysis', 'Backdoor', 'Benign', 'DoS', 'Exploits', 'Fuzzers', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms'],
    # dns-doh: CIRA-CIC-DoHBrw-2020 (DoH流量检测)
    'dns-doh': ['benign', 'malicious'],
    # dns-exf: CIC-Bell-EXF-DNS-2021 & BCCC-EXF (数据渗透)
    'dns-exf': ['benign', 'exfiltration_light', 'exfiltration_heavy'],
    # dns-mal: BCCC-CIC-Bell-DNS-2024-Mal (综合威胁)
    'dns-mal': ['benign', 'spam', 'phishing', 'malware'],
    # SABU-Alert Threat
    'sabu-alert': ["DGA Malware Attack",
        "Botnet Coordinated Activity",
        "DNS Tunneling",
        "Botnet Node-C2 Server Communication",
        "Low-Frequency Malicious IP Probing/Reconnaissance",
        "Low-threat tentative probing",
        "Benign"
    ],
    #--------------------IoT---------------------------------------------------------------------------------
    # NFTON
    'NFTON': ['Benign', 'backdoor', 'ddos', 'dos', 'injection', 'mitm', 'password', 'ransomware', 'scanning', 'xss'],
    # NFBOT
    'NFBOT': ['Benign', 'DDoS', 'DoS', 'Reconnaissance', 'Theft'],
    # CICTON
    'CICTON': ['Benign', 'backdoor', 'ddos', 'dos', 'injection', 'mitm', 'password', 'ransomware', 'scanning', 'xss'],
    #-------------------------------------------------------------------------------------------------------------
    #----------------------Traffic----------------------------------------------------------------------------------
    # # NFCSE
    # 'NFCSE': ['Benign', 'Bot', 'Brute Force -Web', 'Brute Force -XSS', 'DDOS attack-HOIC', 'DDOS attack-LOIC-UDP', 'DDoS attacks-LOIC-HTTP', 'DoS attacks-GoldenEye', 'DoS attacks-Hulk', 'DoS attacks-SlowHTTPTest', 'DoS attacks-Slowloris', 'FTP-BruteForce', 'Infilteration', 'SQL Injection', 'SSH-Bruteforce'],
    # # UNSW
    # 'UNSW': ['Analysis', 'Backdoor', 'Benign', 'DoS', 'Exploits', 'Fuzzers', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms'],
    # # NFUQ
    # 'NFUQ': ['Analysis', 'Backdoor', 'Benign', 'Bot', 'Brute Force', 'DDoS', 'DoS', 'Exploits', 'Fuzzers', 'Generic', 'Infilteration', 'Reconnaissance', 'Shellcode', 'Theft', 'Worms', 'injection', 'mitm', 'password', 'ransomware', 'scanning', 'xss'],
    #------------------------log---------------------------------------------------------------------------------
    # others
    # log:日志数据集
    'log-HDFS':['benign','suspicious','malicious'],
    'log-android':['benign', 'malicious', 'suspicious'],
    'log-linux':['benign', 'malicious', 'suspicious'],
    'log-proxifier':['benign', 'malicious', 'suspicious'],
    'supercomputer':['Hardware Failure', 'benign', 'high', 'low', 'malicious', 'suspicious'],
    'log-test':['benign','suspicious','malicious'],
    #--------Traffic & IoT --------------
    'CIC-BCCC-NRC2024': ['ACK Flood', 'Backdoor', 'Benign Traffic', 'DDoS ACK Fragmentation', 'DDoS HTTP Flood', 'DDoS ICMP Flood', 'DDoS ICMP Fragmentation', 'DDoS PSHACK Flood', 'DDoS RSTFIN Flood', 'DDoS TCP SYN Flood', 'DDoS UDP Flood', 'Dictionary Brute Force', 'DoS DNS Flood', 'DoS ICMP Flood', 'DoS SYN Flood', 'DoS TCP Flood', 'DoS UDP Flood', 'MITM', 'MITM ARP Spoofing', 'MQTT Brute Force', 'MQTT DDoS Publish Flood', 'MQTT DoS Connect Flood', 'MQTT DoS Publish Flood', 'MQTT Malformed', 'Mirai ACK Flood', 'Mirai HTTP Flood', 'Mirai Host Brute Force', 'Mirai UDP Flood', 'Mirai UDP Plain', 'OS Fingerprinting', 'Password Attack', 'Port Scanning', 'Ransomware', 'Recon Host Discovery', 'Recon OS Scan', 'Recon Ping Sweep', 'Recon Port Scan', 'Recon Vulnerability Scan', 'SQL Injection', 'SYN Flood', 'Scan Aggressive', 'Scan Host Port', 'Scan Port OS', 'Scan UDP Attack', 'Sparta SSH Brute Force', 'Telnet Brute Force', 'Uploading Attack', 'Vulnerability Scanner', 'XSS'],
    'CIC-BoT-IoT': ['Benign', 'DDoS', 'DoS', 'Reconnaissance', 'Theft'],
    'CIC-IDS-2017': ['BENIGN', 'Bot', 'DDoS', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest', 'DoS slowloris', 'FTP-Patator', 'Heartbleed', 'Infiltration', 'PortScan', 'SSH-Patator', 'Web Attack - Brute Force', 'Web Attack - Sql Injection', 'Web Attack - XSS'],
    'CIC-IoT-DIAD2024': ['ARP Spoofing', 'BruteForce', 'DDoS ACK Fragmentation', 'DDoS ICMP Flood', 'DDoS-HTTP Flood', 'DDoS-ICMP_Fragmentation', 'DNS Spoofing', 'DoS SYN Flood', 'DoS-HTTP_Flood', 'DoS-UDP_Flood', 'Mirai', 'Uploading_Attack', 'VulnerabilityScan', 'XSS', 'benign', 'sqlinjection'],
    'CIC-ToN-IoT': ['Benign', 'backdoor', 'ddos', 'dos', 'injection', 'mitm', 'password', 'ransomware', 'scanning', 'xss'],
    'CICADA-IIoT2024': ['benign', 'cleanup', 'collection', 'command and control', 'credential access', 'discovery', 'exfiltration', 'lateral movement', 'persistence'],
    'CICEVSE2024': ['Aggressive-scan', 'aggressive-scan', 'benign', 'icmp-flood', 'icmp-fragmentation', 'os-fingerprinting', 'port-scan', 'portscan', 'push-ack-flood', 'service-detection', 'service-detection-scan', 'slowLoris-scan', 'slowloris-scan', 'syn-flood', 'syn-stealth', 'syn-stealth-scan', 'synonymous-ip', 'synonymous-ip-flood', 'tcp-flood', 'udp-flood', 'vulnerability-scan'],
    'CICIoMT 2024': ['ARP_Spoofing', 'Benign', 'MQTT-DDoS-Connect_Flood', 'MQTT-DDoS-Publish_Flood', 'MQTT-DoS-Connect_Flood', 'MQTT-DoS-Publish_Flood', 'MQTT-Malformed_Data', 'Recon-OS_Scan', 'Recon-Ping_Sweep', 'Recon-Port_Scan', 'Recon-VulScan', 'TCP_IP-DDoS-ICMP', 'TCP_IP-DDoS-SYN', 'TCP_IP-DDoS-TCP', 'TCP_IP-DDoS-UDP', 'TCP_IP-DoS-ICMP', 'TCP_IP-DoS-SYN', 'TCP_IP-DoS-TCP', 'TCP_IP-DoS-UDP'],
    'CIC_IOT_2023': ['Backdoor_Malware', 'Benign', 'BrowserHijacking', 'CommandInjection', 'DDoS', 'DNS_Spoofing', 'DictionaryBruteForce', 'DoS', 'MITM-ArpSpoofing', 'Mirai', 'Reconnaissance', 'SqlInjection', 'Uploading_Attack', 'VulnerabilityScan', 'XSS'],
    'NF-BoT-IoT-v2': ['Benign', 'DDoS', 'DoS', 'Reconnaissance', 'Theft'],
    'NF-CSE-CIC-IDS2018-v2': ['Benign', 'Bot', 'Brute Force -Web', 'Brute Force -XSS', 'DDOS attack-HOIC', 'DDOS attack-LOIC-UDP', 'DDoS attacks-LOIC-HTTP', 'DoS attacks-GoldenEye', 'DoS attacks-Hulk', 'DoS attacks-SlowHTTPTest', 'DoS attacks-Slowloris', 'FTP-BruteForce', 'Infilteration', 'SQL Injection', 'SSH-Bruteforce'],
    'NF-ToN-IoT-v2': ['Benign', 'backdoor', 'ddos', 'dos', 'injection', 'mitm', 'password', 'ransomware', 'scanning', 'xss'],
    'NF-UNSW-NB15-v2': ['Analysis', 'Backdoor', 'Benign', 'DoS', 'Exploits', 'Fuzzers', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms'],
    'NF-UQ-NIDS-v2': ['Analysis', 'Backdoor', 'Benign', 'Bot', 'Brute Force', 'DDoS', 'DoS', 'Exploits', 'Fuzzers', 'Generic', 'Infilteration', 'Reconnaissance', 'Shellcode', 'Theft', 'Worms', 'injection', 'mitm', 'password', 'ransomware', 'scanning', 'xss'],
    'iscxids2012': ['Attack', 'Normal'],
    #-------url-------
    'Feodo-Tracker-ipblocklist': ['Dridex', 'QakBot', 'TrickBot', 'BazarLoader', 'Emotet'],
    'ISCX-URL2016': ['Defacement', 'benign', 'spam', 'malware', 'phishing'],
    'Malicious-URLs': ['benign', 'malicious'],
    'PhiUSIIL_URL_Dataset': ['Benign', 'RAT', 'Ransomware', 'Stealer', 'Trojan'],
    #--------Vulneralbility --------------
    'aliyun': ["injection_attack", "authentication_failure", "access_control_breach", "code_execution", "data_exposure", "client_side_attack", "file_manipulation", "logic_flaw", "resource_exhaustion", "deserialization_exploit"],
    'talos': ["injection_attack", "authentication_failure", "access_control_breach", "code_execution", "data_exposure", "client_side_attack", "file_manipulation", "logic_flaw", "resource_exhaustion", "deserialization_exploit"],
    'exploit-db': ["injection_attack", "authentication_failure", "access_control_breach", "code_execution", "data_exposure", "client_side_attack", "file_manipulation", "logic_flaw", "resource_exhaustion", "deserialization_exploit"]
}


def map_dataset_to_strategy(dataset_name):
    """
    验证数据集名称是否存在于配置中

    Args:
        dataset_name: 用户输入的数据集名称

    Returns:
        bool: 是否有效
    """
    if dataset_name not in DATASET_THREAT_TYPES:
        available_datasets = '\n  '.join(sorted(DATASET_THREAT_TYPES.keys()))
        raise ValueError(
            f"未知的数据集名称: '{dataset_name}'\n"
            f"支持的数据集名称有:\n  {available_datasets}\n\n"
            f"如需添加新数据集，请在 DATASET_THREAT_TYPES 中配置。"
        )
    return True

import json
import os
from pathlib import Path

CASE_DIR = Path(__file__).parent.parent / "training"  
DEFAULT_RAG_ROOT = Path(__file__).resolve().parents[1] / "MACyber-12B" / "Threat Intelligence RAG"
DEFAULT_RAG_DB = DEFAULT_RAG_ROOT / "known_attack_channel" / "known_attack_RAG.json"
DEFAULT_UNKNOWN_RAG_DB_PREFIX = DEFAULT_RAG_ROOT / "unknown_attack_channel" / "known_attack_result"

RAG_ENABLED = False
RAG_DB_PATH = DEFAULT_RAG_DB
RAG_TOP_K = 3
KNOWN_RAG_SUBCATEGORY_CACHE = {}


def _load_module(module_name: str, module_path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load RAG module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize_meta_value(value) -> str:
    return str(value or "").strip().lower()


def _load_known_rag_subcategories(rag_db_path=RAG_DB_PATH) -> set[str]:
    cache_key = str(Path(rag_db_path).resolve())
    if cache_key not in KNOWN_RAG_SUBCATEGORY_CACHE:
        with Path(rag_db_path).open("r", encoding="utf-8") as f:
            knowledge_base = json.load(f)
        KNOWN_RAG_SUBCATEGORY_CACHE[cache_key] = {
            _normalize_meta_value(item.get("meta", {}).get("subcategory"))
            for item in knowledge_base
            if isinstance(item, dict)
        }
    return KNOWN_RAG_SUBCATEGORY_CACHE[cache_key]


def _is_known_rag_sample(sample_data, rag_db_path=RAG_DB_PATH) -> bool:
    meta_data = sample_data.get("meta", {}) if isinstance(sample_data, dict) else {}
    subcategory = _normalize_meta_value(meta_data.get("subcategory"))
    if not subcategory:
        return False
    return subcategory in _load_known_rag_subcategories(rag_db_path)


def load_rag_reference(sample_data, top_k: int = RAG_TOP_K, rag_db_path=RAG_DB_PATH) -> str:
    if not _is_known_rag_sample(sample_data, rag_db_path):
        module_path = DEFAULT_RAG_ROOT / "unknown_attack_channel" / "unknown-attack_retrieval_augmented.py"
        module = _load_module("unknown_attack_retrieval_augmented", module_path)
        return module.build_rag_context(sample_data, db_prefix=DEFAULT_UNKNOWN_RAG_DB_PREFIX, top_k=top_k)

    rag_dir = Path(rag_db_path).resolve().parent
    module_path = rag_dir / "known-attack_retrieval_augmented.py"
    module = _load_module("known_attack_retrieval_augmented", module_path)
    build_rag_context = module.build_rag_context

    meta_data = sample_data.get("meta", {}) if isinstance(sample_data, dict) else {}
    return build_rag_context(meta_data, knowledge_base_path=rag_db_path, top_k=top_k)

def load_few_shot_examples(top_k: int = 15) -> str:
    """
    读取 CASE_DIR 下前 top_k 个 json 文件，按 1.json 2.json ... 排序
    返回可直接塞进 prompt 的字符串
    """
    if not CASE_DIR.is_dir():
        raise FileNotFoundError(f"请确保目录存在: {CASE_DIR.resolve()}")

    # 按文件名数字升序取前 top_k 个
    files = sorted(CASE_DIR.glob("*.json"), key=lambda p: int(p.stem))[:top_k]
    if len(files) < top_k:
        raise RuntimeError(f"目录下 json 不足 {top_k} 个")

    parts = []
    for idx, fp in enumerate(files, 1):
        try:
            with fp.open(encoding="utf-8") as f:
                content = json.load(f)
        except Exception as e:
            raise RuntimeError(f"读取 {fp} 失败: {e}")
        # 保留原始 JSON 缩进，方便模型照抄
        raw = json.dumps(content, ensure_ascii=False, indent=2)
        parts.append(f"样例 {idx}:\n{raw}")

    return "\n\n".join(parts)

def get_system_prompt(dataset_name, rag_reference: str = ""):
    """
    Return the corresponding system prompt based on the dataset name
    """
    threat_types = DATASET_THREAT_TYPES.get(dataset_name, ['benign', 'malicious'])
    threat_types_str = ', '.join([f'"{t}"' for t in threat_types])
    rag_block = f'\n\nThe following is the example you can reference:\n{rag_reference}' if rag_reference else ''
    return f"""Do not ask me any questions, just output according to the format and complete as required. You are an experienced cybersecurity expert. Your task is to analyze features in the input data, identify potential security threats, and provide detailed analysis results.

You must output in the following JSON format (output only JSON, no extra content):
{{
  "evidence": [
    "Several brief pieces of evidence that can assist label judgment",
    "field1 = value1 (interpretation)",
    "field2 = value2 (interpretation)"
  ],
  "analysis": "First check <key features>; then verify <key features>; finally confirm <key features>; because <comprehensive judgment reason>, classify as <threat type>.",
  "action": "block/monitor/none",
  "official": "threat type label",
  "severity": "benign/suspicious/low/medium/high"
}}

Field requirements:
1. evidence: Must be an array of N elements, each element format "field_name = value (security interpretation of that feature)"
2. analysis: Must follow the format "First check...; then verify...; finally confirm...; because..., classify as..."
3. action: Can only be block, monitor or none
4. official: Threat type label, must choose one from the following options: {threat_types_str}
5. severity: Can only be benign, suspicious, low, medium or high
{rag_block}
"""


def generate_answer(json_data, meta_data, dataset_name):
    """
    Call large model to generate answers
    """
    try:
        from collections import OrderedDict
        sample_data = OrderedDict([("meta", meta_data), ("json", json_data)])
        rag_reference = load_rag_reference(sample_data, top_k=RAG_TOP_K, rag_db_path=RAG_DB_PATH) if RAG_ENABLED else ""
        system_prompt = get_system_prompt(dataset_name, rag_reference=rag_reference)

        # 构建包含meta和json的输入数据，使用OrderedDict确保meta在json前
        input_data = sample_data
        user_prompt = f"""Please analyze the following feature data and output analysis results in the required JSON format:

Feature data:
```json
{json.dumps(input_data, ensure_ascii=False, indent=2)}
```"""

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        result_text = call_api_with_retry(messages)
        if result_text:
            try:
                if '```json' in result_text:
                    result_text = result_text.split('```json')[1].split('```')[0].strip()
                elif '```' in result_text:
                    result_text = result_text.split('```')[1].split('```')[0].strip()
                answer = json.loads(result_text)

                required_fields = ['evidence', 'analysis', 'action', 'official', 'severity']
                for field in required_fields:
                    if field not in answer:
                        return {"status": "error", "message": f"Missing required field: {field}"}
                if not isinstance(answer['evidence'], list):
                    return {"status": "error", "message": "evidence must be a list format"}

                return {"status": "success", "answer": answer}
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed: {result_text[:200]}...")
                return {"status": "error", "message": f"JSON parsing error: {str(e)}", "raw_response": result_text[:500]}
        else:
            return {"status": "error", "message": "API call failed"}
    except Exception as e:
        logger.error(f"Error occurred when generating answer: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}
        
# def get_system_prompt(dataset_name):
#     """
#     根据数据集名称返回对应的系统提示词

#     Args:
#         dataset_name: 用户输入的数据集名称

#     Returns:
#         str: 完整的系统提示词
#     """
#     # 获取该数据集允许的威胁类型列表
#     threat_types = DATASET_THREAT_TYPES.get(dataset_name, ['benign', 'malicious'])
#     threat_types_str = '、'.join([f'"{t}"' for t in threat_types])
#     #-----------------few-shot prompt--------------------
#     few_shot_block = load_few_shot_examples(4)
#     #----------------------------------------------------
#     # 威胁类型说明字典
#     threat_type_descriptions = {
#     }

#     # 只显示当前数据集相关的威胁类型说明
#     relevant_descriptions = [
#         f'  - {threat_type_descriptions[t]}'
#         for t in threat_types
#         if t in threat_type_descriptions
#     ]
#     threat_descriptions_text = '\n'.join(relevant_descriptions)

#     return f"""你是一位经验丰富的网络安全专家。你的任务是：分析输入数据中的特征，识别潜在的安全威胁，并提供详细的分析结果。

# **你必须按照以下JSON格式输出（只输出JSON，不要任何额外内容）：**
# ```json
# {{
#   "evidence": [
#     "能够辅助label判断的描述的几个简短证据",
#   ],
#   "analysis": "First check <关键特征>; then verify <关键特征>; finally confirm <关键特征>; because <综合判断原因>, classify as <威胁类型>.",
#   "action": "block/monitor/none",
#   "official": "威胁类型标签",
#   "severity": "benign/suspicious/low/medium/high"
# }}
# ```

# **字段要求：**
# 1. **evidence**: 必须是N个元素的数组，每个元素格式为"字段名 = 数值 (对该特征的安全解读)"
# 2. **analysis**: 必须遵循"First check...; then verify...; finally confirm...; because..., classify as..."的格式
# 3. **action**: 只能是 block、monitor 或 none
# 4. **official**: 威胁类型标签，**必须从以下选项中选择一个**：{threat_types_str}
# 5. **severity**: 只能是 benign、suspicious、low、medium 或 high

# **威胁类型说明：**
# {threat_descriptions_text}
# """

# def generate_answer(json_data, meta_data, dataset_name):
#     """
#     调用大模型生成答案

#     Args:
#         json_data: 特征数据
#         meta_data: 元数据
#         dataset_name: 数据集名称

#     Returns:
#         dict: 包含5个字段的答案，或错误信息
#     """
#     try:
#         system_prompt = get_system_prompt(dataset_name)

#         # 构建包含meta和json的输入数据，使用OrderedDict确保meta在json前
#         from collections import OrderedDict
#         input_data = OrderedDict([
#             ("meta", meta_data),
#             ("json", json_data)
#         ])

#         user_prompt = f"""请分析以下特征数据，并按照要求的JSON格式输出分析结果：

# **特征数据：**
# ```json
# {json.dumps(input_data, ensure_ascii=False, indent=2)}
# ```
# """

#         messages = [
#             {'role': 'system', 'content': system_prompt},
#             {'role': 'user', 'content': user_prompt}
#         ]

#         result_text = call_api_with_retry(messages)

#         if result_text:
#             # 提取JSON
#             try:
#                 # 尝试提取可能被markdown包裹的JSON
#                 if '```json' in result_text:
#                     result_text = result_text.split('```json')[1].split('```')[0].strip()
#                 elif '```' in result_text:
#                     result_text = result_text.split('```')[1].split('```')[0].strip()

#                 answer = json.loads(result_text)

#                 # 验证必需字段
#                 required_fields = ['evidence', 'analysis', 'action', 'official', 'severity']
#                 for field in required_fields:
#                     if field not in answer:
#                         return {
#                             "status": "error",
#                             "message": f"缺少必需字段: {field}"
#                         }

#                 # 验证evidence是列表
#                 if not isinstance(answer['evidence'], list):
#                     return {
#                         "status": "error",
#                         "message": "evidence必须是列表格式"
#                     }

#                 return {
#                     "status": "success",
#                     "answer": answer
#                 }

#             except json.JSONDecodeError as e:
#                 logger.error(f"JSON解析失败: {result_text[:200]}...")
#                 return {
#                     "status": "error",
#                     "message": f"JSON解析错误: {str(e)}",
#                     "raw_response": result_text[:500]
#                 }
#         else:
#             return {
#                 "status": "error",
#                 "message": "API调用失败"
#             }

#     except Exception as e:
#         logger.error(f"生成答案时发生错误: {str(e)}", exc_info=True)
#         return {
#             "status": "error",
#             "message": str(e)
#         }

def process_single_sample(sample_data, index, dataset_name, pbar=None, stats=None, lock=None):
    """
    处理单个样本

    Args:
        sample_data: 输入的样本数据（包含json字段）
        index: 样本索引
        dataset_name: 数据集名称（由命令行参数指定）
        pbar: 进度条
        stats: 统计信息
        lock: 线程锁

    Returns:
        dict: 生成的答案或错误信息
    """
    try:
        json_data = sample_data.get('json', {})
        meta_data = sample_data.get('meta', {})

        if not json_data:
            if lock:
                with lock:
                    if stats:
                        stats['skipped'] += 1
                    if pbar:
                        pbar.update(1)
            return {
                'status': 'skipped',
                'index': index,
                'message': 'json字段为空'
            }

        # 验证数据集名称
        map_dataset_to_strategy(dataset_name)
        logger.info(f"处理样本 {index}: 数据集={dataset_name}")

        # 生成答案
        time.sleep(API_DELAY)
        result = generate_answer(json_data, meta_data, dataset_name)

        if result['status'] == 'success':
            logger.info(f"样本 {index} 生成成功: action={result['answer']['action']}, official={result['answer']['official']}")

            # 构建输出格式（只包含5个字段）
            output = result['answer']

            if lock:
                with lock:
                    if stats:
                        stats['success'] += 1
                    if pbar:
                        pbar.update(1)

            return {
                'status': 'success',
                'index': index,
                'output': output
            }
        else:
            logger.error(f"样本 {index} 生成失败: {result.get('message', '')}")

            if lock:
                with lock:
                    if stats:
                        stats['failed'] += 1
                    if pbar:
                        pbar.update(1)

            return {
                'status': 'error',
                'index': index,
                'message': result.get('message', '未知错误')
            }

    except Exception as e:
        logger.error(f"处理样本 {index} 失败: {str(e)}", exc_info=True)

        if lock:
            with lock:
                if stats:
                    stats['failed'] += 1
                if pbar:
                    pbar.update(1)

        return {
            'status': 'error',
            'index': index,
            'message': str(e)
        }

def update_progress_bar(pbar, stats):
    """更新进度条显示"""
    if stats:
        total = stats['success'] + stats['failed']
        if total > 0:
            success_rate = (stats['success'] / total * 100)
            pbar.set_postfix({
                '成功': stats['success'],
                '失败': stats['failed'],
                '跳过': stats['skipped'],
                '成功率': f"{success_rate:.1f}%"
            })

def save_results(output_file, results, incremental=False):
    """
    保存结果到JSON文件

    Args:
        output_file: 输出文件路径
        results: 结果列表
        incremental: 是否增量保存
    """
    try:
        if incremental and os.path.exists(output_file):
            # 读取已有结果
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)

            # 合并结果（使用字典去重）
            results_dict = {r['index']: r for r in existing_results}
            for r in results:
                results_dict[r['index']] = r

            # 转换回列表并排序
            merged_results = sorted(results_dict.values(), key=lambda x: x['index'])

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(merged_results, f, ensure_ascii=False, indent=2)
        else:
            # 直接保存
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"结果已保存到: {output_file}")

    except Exception as e:
        logger.error(f"保存结果失败: {str(e)}")
        raise

def generate_answers(input_file, output_file, dataset_name, max_workers=MAX_WORKERS, tiny=None, use_rag=False, rag_db_path=None, rag_top_k=RAG_TOP_K):
    """
    为数据集生成答案

    Args:
        input_file: 输入JSON文件路径
        output_file: 输出JSON文件路径
        dataset_name: 数据集名称（如 BCCC-CIC-Bell-DNS-EXF）
        max_workers: 最大线程数
        tiny: Tiny模式，随机抽取指定数量的样本（可选）
    """
    try:
        global RAG_ENABLED, RAG_DB_PATH, RAG_TOP_K
        RAG_ENABLED = use_rag
        RAG_DB_PATH = Path(rag_db_path) if rag_db_path else DEFAULT_RAG_DB
        RAG_TOP_K = rag_top_k

        # 验证数据集有效性
        map_dataset_to_strategy(dataset_name)

        print(f"\n{'=' * 70}")
        print("网络安全数据集答案生成程序")
        print(f"{'=' * 70}")
        print(f"输入文件: {input_file}")
        print(f"输出文件: {output_file}")
        print(f"数据集名称: {dataset_name}")
        print(f"生成模型: {MODEL_NAME}")
        print(f"并行线程数: {max_workers}")
        print(f"RAG: {'enabled' if RAG_ENABLED else 'disabled'}")
        if RAG_ENABLED:
            print(f"RAG知识库: {RAG_DB_PATH}")
            print(f"RAG top_k: {RAG_TOP_K}")
        print(f"{'=' * 70}\n")

        # 读取输入数据
        print(f"正在读取输入文件...")
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        # === 去除答案，防止泄露 ===
        for item in input_data:
            # 只保留纯特征，去掉 label & reasoning & response
            item['json'].pop('label', None)
            item['json'].pop('reasoning', None)
            item['json'].pop('response', None)
        print(f"总共 {len(input_data)} 条记录")

        # Tiny模式：随机抽取指定数量的样本
        if tiny is not None and tiny > 0:
            original_count = len(input_data)
            sample_count = min(tiny, original_count)
            input_data = random.sample(input_data, sample_count)
            print(f"🎲 Tiny模式: 从 {original_count} 条记录中随机抽取 {sample_count} 条")

        # 统计信息
        stats = {
            'success': 0,
            'failed': 0,
            'skipped': 0
        }

        # 创建线程锁
        lock = threading.Lock()

        print(f"\n开始生成答案...")
        print("=" * 70)

        # 创建进度条
        with tqdm(total=len(input_data),
                  desc="生成进度",
                  unit="条",
                  ncols=120) as pbar:

            # 使用线程池处理
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交任务
                future_to_index = {}
                for idx, sample in enumerate(input_data):
                    future = executor.submit(
                        process_single_sample,
                        sample,
                        idx,
                        dataset_name,  # 传递数据集名称参数
                        pbar,
                        stats,
                        lock
                    )
                    future_to_index[future] = idx

                # 收集结果
                all_outputs = [None] * len(input_data)  # 按索引存储输出
                completed_count = 0

                try:
                    for future in as_completed(future_to_index.keys()):
                        try:
                            result = future.result(timeout=REQUEST_TIMEOUT)

                            if result['status'] == 'success':
                                # 将输出存储到对应索引位置
                                all_outputs[result['index']] = result['output']

                            with lock:
                                update_progress_bar(pbar, stats)

                            # 每完成SAVE_INTERVAL条就保存一次
                            completed_count += 1
                            if completed_count % SAVE_INTERVAL == 0:
                                logger.info(f"\n💾 已处理 {completed_count} 条，保存进度...")
                                # 只保存已完成的结果
                                results_to_save = [
                                    {'index': i, 'output': output}
                                    for i, output in enumerate(all_outputs)
                                    if output is not None
                                ]
                                if results_to_save:
                                    save_results(output_file, results_to_save, incremental=True)

                        except Exception as e:
                            logger.error(f"任务执行失败: {str(e)}")
                            with lock:
                                stats['failed'] += 1
                                update_progress_bar(pbar, stats)

                finally:
                    # 保存最终结果（只保存成功的输出）
                    logger.info(f"\n💾 保存最终结果...")
                    final_outputs = [output for output in all_outputs if output is not None]

                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(final_outputs, f, ensure_ascii=False, indent=2)

                    logger.info(f"最终结果已保存: {len(final_outputs)} 条")

        # 打印统计结果
        print("\n" + "=" * 70)
        print("✅ 答案生成完成！")
        print(f"📊 统计结果:")
        print(f"   成功生成: {stats['success']} 条")
        print(f"   生成失败: {stats['failed']} 条")
        print(f"   跳过处理: {stats['skipped']} 条")
        if stats['success'] + stats['failed'] > 0:
            print(f"   成功率: {stats['success'] / (stats['success'] + stats['failed']) * 100:.1f}%")
        print(f"   输出文件: {output_file}")
        print("=" * 70)

    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}", exc_info=True)
        print(f"❌ 处理失败: {str(e)}")


# python generate_answers.py -i input.json -o output.json -t CIC-Bell-EXF-DNS-2021
def main():
    """主函数"""
    global API_KEY, BASE_URL, MODEL_NAME

    parser = argparse.ArgumentParser(
        description="网络安全数据集答案生成程序",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
  使用示例：
  python %(prog)s -i input.json -o output.json -t CIC-Bell-EXF-DNS-2021
  python %(prog)s --input test_data.json --output model_output.json --type BCCC-CIC-Bell-DNS-EXF --workers 16
  python %(prog)s -i domain_data.json -o output.json -t BCCC-CIC-Bell-DNS-Mal
        """
    )

    parser.add_argument('-i', '--input',
                        required=True,
                        help='输入JSON文件路径（包含网络安全特征数据）')

    parser.add_argument('-o', '--output',
                        default='model_output.json',
                        help='输出JSON文件路径 (默认: model_output.json)')

    parser.add_argument('-t', '--type',
                        dest='dataset_name',
                        required=True,
                        help='数据集名称 (必需)，如: BCCC-CIC-Bell-DNS-EXF, BCCC-CIC-Bell-DNS-Mal 等')

    parser.add_argument('--workers',
                        type=int,
                        default=MAX_WORKERS,
                        help=f'最大并发数 (默认: {MAX_WORKERS})')

    parser.add_argument('--tiny',
                        type=int,
                        default=None,
                        help='Tiny模式：随机抽取指定数量的样本进行处理（例如：--tiny 10）')

    parser.add_argument('--api-key',
                        default=API_KEY,
                        help='OpenAI-compatible API key (默认读取 OPENAI_API_KEY)')

    parser.add_argument('--base-url',
                        default=BASE_URL,
                        help='OpenAI-compatible base URL (默认读取 OPENAI_BASE_URL)')

    parser.add_argument('--api-model',
                        default=MODEL_NAME,
                        help='OpenAI-compatible model name (默认读取 OPENAI_MODEL)')

    parser.add_argument('--use-rag',
                        action='store_true',
                        help='Enable threat-intelligence RAG examples in the system prompt')

    parser.add_argument('--rag-db',
                        default=str(DEFAULT_RAG_DB),
                        help='Path to known_attack_RAG.json for the known-attack channel')

    parser.add_argument('--rag-top-k',
                        type=int,
                        default=RAG_TOP_K,
                        help='Number of RAG examples to include')

    args = parser.parse_args()
    API_KEY = args.api_key
    BASE_URL = args.base_url
    MODEL_NAME = args.api_model

    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        return

    # 验证数据集名称
    try:
        map_dataset_to_strategy(args.dataset_name)
        print(f"✓ 数据集名称: {args.dataset_name}\n")
    except ValueError as e:
        print(f"❌ {str(e)}")
        return

    # 生成答案
    generate_answers(
        args.input,
        args.output,
        args.dataset_name,
        max_workers=args.workers,
        tiny=args.tiny,
        use_rag=args.use_rag,
        rag_db_path=args.rag_db,
        rag_top_k=args.rag_top_k,
    )

if __name__ == "__main__":
    main()
