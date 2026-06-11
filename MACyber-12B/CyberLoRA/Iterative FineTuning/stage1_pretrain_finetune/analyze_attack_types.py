#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计数据集文件中的网络攻击类型
"""

import json
import os
from collections import defaultdict

def load_file_list():
    """
    从alpaca_convert_new.py加载文件列表
    """
    # 从alpaca_convert_new.py中复制的文件列表
    dns_list = [
    "/data2/qcy/data_pattern/dns/converted/train/bccc_2024_mal_20251016_003244_train.json",
    "/data2/qcy/data_pattern/dns/converted/train/DNS_exfiltration_benign_1000_train.json",
    "/data2/qcy/data_pattern/dns/converted/train/CIRA_benign_1500_train.json",
    "/data2/qcy/data_pattern/dns/converted/train/DNS_exfiltration_light_1000_train.json",
    "/data2/qcy/data_pattern/dns/converted/train/CIRA_malicious_3500_train.json",
    "/data2/qcy/data_pattern/dns/converted/train/DNS_exfiltration_heavy_1000_train.json",
    "/data2/qcy/data_pattern/dns/converted/train/bccc_2024_exf_20251016_103047_train.json"
    ]

    log_list = ['/data2/qcy/data_pattern/log/split-anroid/training-anroid/output_data_cleaned.json',
                '/data2/qcy/data_pattern/log/split-hdfs/training-hdfs-5007_cleaned.json',
                '/data2/qcy/data_pattern/log/split-linux/training-linux/output_data_cleaned.json',
                '/data2/qcy/data_pattern/log/split-proxifier/test-proxifier/output_data_cleaned.json',
                '/data2/qcy/data_pattern/log/split-supercomputer/test-supercom-22/output_data_cleaned.json'
                ]

    threat_list = ['/data2/qcy/sft_workflow/threat/Pulsedive-Threats/training/Pulsedive-Threats.json',
                   '/data2/qcy/data_pattern/threat/SABU-Alert/converted/SABU-Alert_train_no_answer.json']

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

    url_list = ['/data2/qcy/data_pattern/url_update/train/Feodo-Tracker-ipblocklist/Feodo-Tracker-ipblocklist.json',
    '/data2/qcy/data_pattern/url_update/train/ISCX-URL2016/ISCX-URL2016_no_class_evidence.json',
    '/data2/qcy/data_pattern/url_update/train/Malicious-URLs/Malicious-URLs.json']

    vulnerablity_list = ['/data2/qcy/data_pattern/vulnerablity/aliyun/converted/training/output_data.json',
                         '/data2/qcy/data_pattern/vulnerablity/exploit-db/converted/training/output_data.json',
                         '/data2/qcy/data_pattern/vulnerablity/talos/converted/training/output_data.json']
    
    # 合并所有列表
    all_files = []
    all_files.extend(dns_list)
    all_files.extend(log_list)
    all_files.extend(threat_list)
    all_files.extend(traffic_and_iot_list)
    all_files.extend(url_list)
    all_files.extend(vulnerablity_list)
    
    return all_files

def analyze_file(file_path):
    """
    分析单个文件，统计攻击类型
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print(f"警告: {file_path} 不是JSON列表格式")
            return {}
        
        attack_types = defaultdict(int)
        
        for item in data:
            if 'label' in item and 'official' in item['label']:
                attack_type = item['label']['official']
                attack_types[attack_type] += 1
        
        return attack_types
        
    except Exception as e:
        print(f"错误分析文件 {file_path}: {e}")
        return {}

def main():
    """
    主函数
    """
    all_files = load_file_list()
    total_attack_types = defaultdict(int)
    file_results = {}
    
    print("开始分析数据集文件...")
    print("=" * 80)
    
    for file_path in all_files:
        if os.path.exists(file_path):
            print(f"分析文件: {file_path}")
            attack_types = analyze_file(file_path)
            file_results[file_path] = attack_types
            
            # 累计到总统计
            for attack_type, count in attack_types.items():
                total_attack_types[attack_type] += count
            
            if attack_types:
                print(f"  攻击类型统计:")
                for attack_type, count in sorted(attack_types.items()):
                    print(f"    - {attack_type}: {count}")
            else:
                print(f"  未找到攻击类型数据")
        else:
            print(f"文件不存在: {file_path}")
        print("-" * 80)
    
    # 打印总统计
    print("\n总攻击类型统计:")
    print("=" * 80)
    if total_attack_types:
        for attack_type, count in sorted(total_attack_types.items(), key=lambda x: x[1], reverse=True):
            print(f"{attack_type}: {count}")
    else:
        print("未找到任何攻击类型数据")
    
    print("\n分析完成！")

if __name__ == "__main__":
    main()