from __future__ import annotations

import json
import os
# 1. 强制 CUDA 使用 PCI 总线 ID 排序（和 nvidia-smi 保持一致）
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# 2. 指定使用哪张卡（此时的 '0' 就是 nvidia-smi 里的 '0'）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
PARA_SIZE = 1
# GPU_MEMORY_UTILIZATION = 0.275
import logging
import argparse
import random
from datetime import datetime
from collections import OrderedDict
from tqdm import tqdm

# 获取当前脚本所在目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 结果输出目录
RESULT_DIR = os.path.join(CURRENT_DIR, "result")

# 模型配置
MODEL_PATH = os.path.join(CURRENT_DIR, "Mistral-Small-3.2-24B-Instruct-2506")
# MODEL_PATH = os.path.join(CURRENT_DIR, "Foundation-Sec-8B")
# 推理配置
BATCH_SIZE = 16  # 每批处理的样本数
SAVE_INTERVAL = 16  # 每处理多少条保存一次（与批大小一致）
MAX_MODEL_LEN = 8192

# 采样参数
TEMPERATURE = 0.3
TOP_P = 0.9
MAX_TOKENS = 8192

# GPU_MEMORY_UTILIZATION = 0.3  # 降低到20%，避免显存不足
# GPU_MEMORY_UTILIZATION = 0.4  # 如果显存充足可以使用0.4
# 0.95比较激进，不能把GPU所有显存都占满，要留一部分，推荐0.9
# 补救prompt
FIX_PROMPT = """你上次输出格式有误，请严格按下面要求重新整理，只返回合法 JSON，不要任何解释。

必须同时包含且仅包含这5个字段：
{
  "evidence": ["字段名=数值(解读)", ...],
  "analysis": "First check...; then verify...; finally confirm...; because..., classify as...",
  "action": "block|monitor|none",
  "official": "威胁类型标签",
  "severity": "benign|suspicious|low|medium|high"
}

待整理内容：
{{raw}}

现在直接给出正确 JSON 结果：
"""



# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

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
    # log:日志数据集
    'log-HDFS':['benign','suspicious','malicious'],
    'log-anroid':['benign', 'malicious', 'suspicious'],
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
    #--------Vulneralbility --------------
    'aliyun': ["injection_attack", "authentication_failure", "access_control_breach", "code_execution", "data_exposure", "client_side_attack", "file_manipulation", "logic_flaw", "resource_exhaustion", "deserialization_exploit"],
    'talos': ["injection_attack", "authentication_failure", "access_control_breach", "code_execution", "data_exposure", "client_side_attack", "file_manipulation", "logic_flaw", "resource_exhaustion", "deserialization_exploit"],
    'exploit-db': ["injection_attack", "authentication_failure", "access_control_breach", "code_execution", "data_exposure", "client_side_attack", "file_manipulation", "logic_flaw", "resource_exhaustion", "deserialization_exploit"],
    #--------unkown --------------
    'ICS': ['low', 'high'],
    'Log_unknown': ['benign', 'suspicious', 'low','high'],
    'SDN-DDoS_Traffic_Dataset':['benign','ddos'],
    'PhiUSIIL_URL_Dataset':['Benign','RAT','Ransomware','Stealer','Trojan'],
    'host' : ['benign', 'suspicious activity', 'low risk attack', 'medium risk attack', 'high risk attack']
}

import json
import os
from pathlib import Path

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


def load_rag_reference(sample_data, top_k: int = RAG_TOP_K, rag_db_path: str | Path = RAG_DB_PATH) -> str:
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

import re


def fix_output(llm, tokenizer, raw: str, sampling_params) -> str:
    """
    把第一次解析失败的 raw 文本重新喂给模型，要求只输出 JSON
    """
    from vllm import SamplingParams

    messages = [
        {"role": "system", "content": "你是网络安全专家，只负责把文字整理成指定 JSON，不要解释。"},
        {"role": "user",   "content": FIX_PROMPT.format(raw=raw)}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # 用 greedy 防止再放飞
    greedy_param = SamplingParams(temperature=0, top_p=1, max_tokens=1024,
                                  stop=["```", "\n\n"])
    out = llm.generate([prompt], greedy_param, use_tqdm=False)[0]
    return out.outputs[0].text.strip()

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

#DO NOT RAISE ANY QUESTIONS. DO NOT THOUGHT CHAINS. ANSWER FOLLOWED THE GIVEN FORMAT. 
def get_system_prompt(dataset_name, rag_reference: str = ""):
    threat_types = DATASET_THREAT_TYPES.get(dataset_name, ['benign', 'malicious'])
    threat_types_str = ', '.join([f'"{t}"' for t in threat_types])
    rag_block = f'\n\nThe following is the example you can reference:\n{rag_reference}' if rag_reference else ''
    return f"""You are an experienced cybersecurity expert. Your task is to analyze features in the input data, identify potential security threats, and provide detailed analysis results.

You must output in the following JSON format (output only JSON, no extra content):
{{
  "evidence": [
    "5 brief string of evidence that can assist label judgment",
    "field1 = value1 (interpretation)",
    "field2 = value2 (interpretation)"
  ],
  "analysis": "First check <key features>; then verify <key features>; finally confirm <key features>; because <comprehensive judgment reason>, classify as <threat type>.",
  "action": "threat type label, must choose one from the following options: {threat_types_str}",
  "official": "threat type label",
  "severity": "benign/suspicious/low/medium/high"
}}

Field requirements:
1. evidence: Must be an array of N string elements, each element format "field_name = value (security interpretation)", array must not be empty, fill in at least 5 specific features!
2. analysis: Must follow the format "First check...; then verify...; finally confirm...; because..., classify as...", must not be omitted!
3. action: Can only be block, monitor or none
4. official: Threat type label, must choose one from the following options: {threat_types_str}
5. severity: Can only be benign, suspicious, low, medium or high
{rag_block}
"""

def build_user_prompt(json_data, meta_data):
    input_data = OrderedDict([("meta", meta_data), ("json", json_data)])
    return f"""Please analyze the following feature data and output analysis results in the required JSON format:

Feature data:
```json
{json.dumps(input_data, ensure_ascii=False, indent=2)}
```"""

def build_chat_messages(sample_data, dataset_name):
    """
    构建聊天消息格式

    Args:
        sample_data: 输入的样本数据（包含json字段）
        dataset_name: 数据集名称

    Returns:
        list: 消息列表 [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
    """
    json_data = sample_data.get('json', {})
    meta_data = sample_data.get('meta', {})

    rag_reference = load_rag_reference(sample_data, top_k=RAG_TOP_K, rag_db_path=RAG_DB_PATH) if RAG_ENABLED else ""
    system_prompt = get_system_prompt(dataset_name, rag_reference=rag_reference)
    user_prompt = build_user_prompt(json_data, meta_data)

    return [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt}
    ]

def extract_json_block(text: str) -> str:
    """
    清洗并提取 JSON 字符串
    新增功能：自动去除多余的双括号 {{...}}
    """
    # 1. 尝试提取 ```json ... ``` 或 ``` ... ```
    m = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', text)
    if m:
        text = m.group(1).strip()
    
    # start = text.find('[')
    # end = text.rfind(']')
    
    # if start != -1 and end != -1 and end > start:
    #     text = text[start:end+1].strip()
    # else:
    #     return ""

    # 2. 寻找最外层的 {...}
    start = text.find('{')
    end = text.rfind('}')
    
    if start != -1 and end != -1 and end > start:
        text = text[start:end+1].strip()
    else:
        return ""

    # ===== 新增：剥洋葱逻辑 =====
    # 如果字符串是 {{ ... }} 格式（双花括号包裹），循环去除直到只剩一层
    # 这里的 while 循环可以处理 {{{ ... }}} 这种极端情况
    while text.startswith('{{') and text.endswith('}}'):
        text = text[1:-1].strip()
    
    return text

def parse_model_output(result_text):
    """
    解析模型输出，提取JSON，包含自动修复机制
    """
    # ... (前面的代码保持不变) ...
    if not result_text:
        return {"status": "error", "message": "模型输出为空"}

    try:
        text = result_text.strip()
        text = text.replace("（", "(").replace("）", ")").replace("\n", " ") 
        
        json_str = extract_json_block(text) # 调用升级后的提取函数
        
        if not json_str:
            return {"status": "error", "message": "未提取到JSON有效片段"}

        # ... (中间的 json.loads / ast.literal_eval 解析逻辑保持不变) ...
        
        answer = None
        try:
            answer = json.loads(json_str)
        except json.JSONDecodeError:
            try:
                if json_str.strip().startswith('{'):
                    answer = ast.literal_eval(json_str)
            except Exception:
                pass

        if not isinstance(answer, dict):
            print('_' * 40)
            print(json_str)
            print('_' * 40)
            raise json.JSONDecodeError("无法解析为字典", json_str, 0)

        # ===== 新增：自动拆包逻辑 (针对 {"response": {...}} 这种情况) =====
        # 如果解析出来了，但必需字段都不在第一层，而在某个子字典里
        required_fields_check = ['evidence', 'analysis']
        
        # 检查当前层是否包含必需字段
        is_direct_hit = any(k in answer for k in required_fields_check)
        
        if not is_direct_hit:
            # 如果不在第一层，尝试寻找只有一个 Value 是字典的情况
            # 例如: {"wrapper": {"evidence": [], ...}}
            for key, val in answer.items():
                if isinstance(val, dict) and any(k in val for k in required_fields_check):
                    # 找到了！提升这一层及其内容，丢弃外壳
                    answer = val 
                    break

        # ===== Step 4: 补全与验证 (保持不变) =====
        answer.setdefault("evidence", [])
        answer.setdefault("analysis", "")
        
        # ... (后续验证代码保持不变) ...
        
        # 验证必需字段
        required_fields = ['evidence', 'analysis', 'action', 'official', 'severity']
        missing = [f for f in required_fields if f not in answer]
        if missing:
            return {
                "status": "error", 
                "message": f"缺少必需字段: {', '.join(missing)}"
            }

        # 验证 evidence 格式
        if not isinstance(answer['evidence'], list):
             if isinstance(answer['evidence'], str):
                 answer['evidence'] = [answer['evidence']]
             else:
                return {"status": "error", "message": "evidence必须是列表格式"}

        return {
            "status": "success",
            "answer": answer
        }

    except Exception as e:
        logger.error(f"解析异常: {str(e)}")
        return {
            "status": "error",
            "message": f"解析失败: {str(e)}",
            "raw_response": result_text[:500]
        }

def create_result_directory(dataset_name):
    """
    创建基于数据集名称和时间的结果子目录

    Args:
        dataset_name: 数据集名称

    Returns:
        str: 创建的子目录路径
    """
    # 生成时间戳 (格式: YYYYMMDD_HHMMSS)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建子目录名称: 数据集名称_时间戳
    subdir_name = f"{dataset_name}_{timestamp}"
    
    # 完整路径
    result_subdir = os.path.join(RESULT_DIR, subdir_name)
    
    # 创建目录（如果不存在）
    os.makedirs(result_subdir, exist_ok=True)
    
    return result_subdir


def load_existing_results(output_file):
    """
    加载已有的结果文件
    简单模式：只返回列表，用于比对数量
    """
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            logger.warning("已有结果文件格式不正确（不是列表），将视为无效")
            return []
        except Exception as e:
            logger.warning(f"读取已有结果文件失败: {e}，将视为无效")
            return []
    return []

def save_results(output_file, results):
    """
    保存结果到JSON文件

    Args:
        output_file: 输出文件路径
        results: 结果列表（只包含成功的输出）
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存到: {output_file}")
    except Exception as e:
        logger.error(f"保存结果失败: {str(e)}")
        raise

def generate_answers(input_file, output_file, dataset_name, batch_size=BATCH_SIZE, GPU_MEMORY_UTILIZATION = 0.1, tiny=None, model_path=MODEL_PATH, use_rag=False, rag_db_path=None, rag_top_k=RAG_TOP_K):
    """
    使用vLLM为数据集生成答案

    Args:
        input_file: 输入JSON文件路径
        output_file: 输出文件路径
        dataset_name: 数据集名称（如 traffic, Pulsedive-Threats）
        batch_size: 批处理大小
        tiny: Tiny模式，随机抽取指定数量的样本（可选）
        model_path: 模型路径
    """
    try:
        global RAG_ENABLED, RAG_DB_PATH, RAG_TOP_K
        RAG_ENABLED = use_rag
        RAG_DB_PATH = Path(rag_db_path) if rag_db_path else DEFAULT_RAG_DB
        RAG_TOP_K = rag_top_k

        # 验证数据集有效性
        map_dataset_to_strategy(dataset_name)

        print(f"\n{'=' * 70}")
        print("网络安全数据集答案生成程序 (vLLM本地推理版)")
        print(f"{'=' * 70}")
        print(f"输入文件: {input_file}")
        print(f"输出文件: {output_file}")
        print(f"数据集名称: {dataset_name}")
        print(f"模型路径: {model_path}")
        print(f"批处理大小: {batch_size}")
        print(f"GPU显存利用率: {GPU_MEMORY_UTILIZATION}")
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
            # 使用更安全的方式删除键，避免某些字典类型不支持默认值参数
            for key in ['label', 'reasoning', 'response']:
                if key in item['json']:
                    del item['json'][key]
        print(f"总共 {len(input_data)} 条记录")

        # Tiny模式：随机抽取指定数量的样本
        if tiny is not None and tiny > 0:
            original_count = len(input_data)
            sample_count = min(tiny, original_count)
            input_data = random.sample(input_data, sample_count)
            print(f"🎲 Tiny模式: 从 {original_count} 条记录中随机抽取 {sample_count} 条")

        # 加载已有结果（断点续传）
        existing_results = load_existing_results(output_file)
    
        # 2. 比对数量
        if len(existing_results) == len(input_data):
            print(f"\n✅ 检测到输出文件已有 {len(existing_results)} 条结果，与输入数据量一致。")
            print("无需重新生成，跳过执行。")
            return  # 直接结束程序

        # 3. 如果数量不一致，进入覆盖模式（全部重跑）
        if len(existing_results) > 0:
            print(f"\n⚠️ 现有结果 ({len(existing_results)}条) 与 输入数据 ({len(input_data)}条) 数量不匹配。")
            print("即将重新运行并【覆盖】原有文件...")
        else:
            print(f"\n🚀 开始全新的生成任务...")

        # 4. 初始化状态：全部重跑
        samples_to_process = input_data  # 所有数据都要跑
        sample_indices = list(range(len(input_data))) # 索引 0 到 N-1
        
        # 初始化全空的输出列表（不继承旧数据）
        all_outputs = [None] * len(input_data)
        
        # 统计重置
        stats = {
            'success': 0,
            'failed': 0,
            'skipped': 0
        }
        
        print(f"需要处理的样本数: {len(samples_to_process)}")

        # 初始化 vLLM 引擎
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        print(f"\n正在加载模型: {model_path}")
        print("这可能需要一些时间...")

        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            tensor_parallel_size=PARA_SIZE, #是2倍数s
            max_model_len=MAX_MODEL_LEN,
            # kv_cache_dtype="fp8",
            max_num_seqs=32,
            # quantization="awq"
        )

        # 加载 tokenizer 用于 chat template
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS
        )

        print("✅ 模型加载完成\n")

        # 统计信息
        stats = {
            'success': len(existing_results),
            'failed': 0,
            'skipped': 0
        }

        # 存储所有输出（包括已有的和解析失败的）
        all_outputs = [None] * len(input_data)

        print(f"开始生成答案...")
        print("=" * 70)

        # 分批处理
        num_batches = (len(samples_to_process) + batch_size - 1) // batch_size

        with tqdm(total=len(samples_to_process), desc="生成进度", unit="条", ncols=120) as pbar:
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(samples_to_process))

                batch_samples = samples_to_process[batch_start:batch_end]
                batch_indices = sample_indices[batch_start:batch_end]

                # 构建批次的提示词
                prompts = []
                for sample in batch_samples:
                    messages = build_chat_messages(sample, dataset_name)
                    # 尝试使用 tokenizer 的 chat template
                    try:
                        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
                            prompt = tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=True
                            )
                # # === 强制 JSON 起手式，锁住 logits ===
                #             prompt = '{"evidence": [' + prompt.split('{"evidence": [')[-1]
                #         else:
                #             raise ValueError("No chat template")
                #     except (ValueError, AttributeError):
                #         system_content = messages[0]['content']
                #         user_content = messages[1]['content']
                #         prompt = f"### System:\n{system_content}\n\n### User:\n{user_content}\n\n### Assistant:\n"
                #     prompts.append(prompt)
                        else:
                            raise ValueError("No chat template")
                    except (ValueError, AttributeError):
                        # 如果不支持 chat template，使用简单的文本格式
                        system_content = messages[0]['content']
                        user_content = messages[1]['content']
                        prompt = f"### System:\n{system_content}\n\n### User:\n{user_content}\n\n### Assistant:\n"
                    prompts.append(prompt)

                # vLLM 批量推理
                try:
                    outputs = llm.generate(prompts, sampling_params)

                    # 处理输出
                    for i, output in enumerate(outputs):
                        idx = batch_indices[i]
                        result_text = output.outputs[0].text.strip()

                        # 解析模型输出
                        parsed = parse_model_output(result_text)

                        if parsed['status'] == 'success':
                            all_outputs[idx] = parsed['answer']
                            stats['success'] += 1
                        else:
                            logger.info(f"样本 {idx} 首次解析失败，即将重采样")
                            # ===== 重采样：同参数再生成一次 =====
                            RETRY = 10
                            for attempt in range(RETRY):
                                retry_outputs = llm.generate([prompts[i]], sampling_params, use_tqdm=False)[0]
                                retry_text = retry_outputs.outputs[0].text.strip()
                                logger.info(f"样本 {idx} 重采样输出: {repr(retry_text)}")
                                second_parsed = parse_model_output(retry_text)
                                if second_parsed['status'] == 'success':
                                    all_outputs[idx] = second_parsed['answer']
                                    stats['success'] += 1
                                    break
                                else:
                                    logger.info(f"样本 {idx} 第{attempt + 1}次解析失败，即将再次重采样")
                                logger.info(f"样本 {idx} 8B 原始输出: {repr(result_text)}")      # ① 看 8B 有没有字
                except Exception as e:
                    logger.error(f"批次 {batch_idx + 1} 推理失败: {str(e)}")
                    stats['failed'] += len(batch_samples)

                # 更新进度条
                pbar.update(len(batch_samples))
                pbar.set_postfix({
                    '成功': stats['success'],
                    '失败': stats['failed'],
                    '成功率': f"{stats['success'] / (stats['success'] + stats['failed']) * 100:.1f}%" if (stats['success'] + stats['failed']) > 0 else "N/A"
                })

                # 保存进度
                if (batch_idx + 1) % (SAVE_INTERVAL // batch_size + 1) == 0 or batch_idx == num_batches - 1:
                    logger.info(f"\n💾 保存进度: 已处理 {batch_end} 条...")
                    # 保存所有输出（包括解析失败的），确保顺序不变
                    final_outputs = [output for output in all_outputs if output is not None]
                    save_results(output_file, final_outputs)

        # 最终保存
        print(f"\n💾 保存最终结果...")
        # 保存所有输出（包括解析失败的），确保顺序不变
        final_outputs = [output for output in all_outputs if output is not None]
        save_results(output_file, final_outputs)

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

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="网络安全数据集答案生成程序 (vLLM本地推理版)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  python %(prog)s -i input.json -t traffic
  python %(prog)s --input test_data.json --type Pulsedive-Threats --batch-size 64
  python %(prog)s -i domain_data.json -t traffic --tiny 10 -o result.json

输出说明：
  所有输出文件自动保存到 result/<数据集名称>_<时间戳>/ 目录下
  例如: result/traffic_20251203_143025/output.json
  注意: 解析失败的结果也会保存到 output.json 中，格式为 {"status": "error", ...}
        """
    )

    parser.add_argument('-i', '--input',
                        required=True,
                        help='输入JSON文件路径（包含网络安全特征数据）')

    parser.add_argument('-o', '--output',
                        default='output.json',
                        help='输出JSON文件名，将自动放入 result/<数据集名称>_<时间戳>/ 目录 (默认: output.json)')

    parser.add_argument('-t', '--type',
                        dest='dataset_name',
                        required=True,
                        help='数据集名称 (必需)，如: traffic, Pulsedive-Threats')

    parser.add_argument('--batch-size',
                        type=int,
                        default=BATCH_SIZE,
                        help=f'批处理大小 (默认: {BATCH_SIZE})')

    parser.add_argument('--tiny',
                        type=int,
                        default=None,
                        help='Tiny模式：随机抽取指定数量的样本进行处理（例如：--tiny 10）')

    parser.add_argument('--model-path',
                        default=MODEL_PATH,
                        help=f'模型路径 (默认: {MODEL_PATH})')

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

    # 检查模型路径
    if not os.path.exists(args.model_path):
        print(f"❌ 模型路径不存在: {args.model_path}")
        return

    # 生成答案
    generate_answers(
        args.input,
        args.output,
        args.dataset_name,
        batch_size=args.batch_size,
        tiny=args.tiny,
        model_path=args.model_path,
        use_rag=args.use_rag,
        rag_db_path=args.rag_db,
        rag_top_k=args.rag_top_k,
    )

if __name__ == "__main__":
    main()
