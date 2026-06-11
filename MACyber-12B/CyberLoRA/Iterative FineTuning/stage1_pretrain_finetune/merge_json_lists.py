#!/usr/bin/env python3
import json
import os

# 基础文件路径列表（第1-36行）
base_files = [
    "/data2/qcy/data_pattern_alpaca/v2/dns/converted/train/bccc_2024_mal_20251016_003244_train.json",
    "/data2/qcy/data_pattern_alpaca/v2/dns/converted/train/DNS_exfiltration_benign_1000_train.json",
    "/data2/qcy/data_pattern_alpaca/v2/dns/converted/train/CIRA_benign_1500_train.json",
    "/data2/qcy/data_pattern_alpaca/v2/dns/converted/train/DNS_exfiltration_light_1000_train.json",
    "/data2/qcy/data_pattern_alpaca/v2/dns/converted/train/CIRA_malicious_3500_train.json",
    "/data2/qcy/data_pattern_alpaca/v2/dns/converted/train/DNS_exfiltration_heavy_1000_train.json",
    "/data2/qcy/data_pattern_alpaca/v2/dns/converted/train/bccc_2024_exf_20251016_103047_train.json",
    "/data2/qcy/data_pattern_alpaca/v2/log/split-anroid/training-anroid/output_data_cleaned.json",
    "/data2/qcy/data_pattern_alpaca/v2/log/split-hdfs/training-hdfs-5007_cleaned.json",
    "/data2/qcy/data_pattern_alpaca/v2/log/split-linux/training-linux/output_data_cleaned.json",
    "/data2/qcy/data_pattern_alpaca/v2/log/split-proxifier/test-proxifier/output_data_cleaned.json",
    "/data2/qcy/data_pattern_alpaca/v2/log/split-supercomputer/test-supercom-22/output_data_cleaned.json",
    "/data2/qcy/data_pattern_alpaca/alpaca_fallback/Pulsedive-Threats.json",
    "/data2/qcy/data_pattern_alpaca/v2/threat/SABU-Alert/converted/SABU-Alert_train_no_answer.json",
    "/data2/qcy/data_pattern_alpaca/v2/traffic_and_iot/train/NF-CSE-CIC-IDS2018-v2/NF-CSE-CIC-IDS2018-v2.json",
    "/data2/qcy/data_pattern_alpaca/v2/traffic_and_iot/train/CIC-ToN-IoT/CIC-ToN-IoT.json",
    "/data2/qcy/data_pattern_alpaca/v2/traffic_and_iot/train/CICIoMT 2024/CICIoMT 2024.json",
    "/data2/qcy/data_pattern_alpaca/v2/traffic_and_iot/train/CICADA-IIoT2024/CICADA-IIoT2024.json",
    "/data2/qcy/data_pattern_alpaca/v2/traffic_and_iot/train/CIC-IoT-DIAD2024/CIC-IoT-DIAD2024.json",
    "/data2/qcy/data_pattern_alpaca/v2/traffic_and_iot/train/CIC-IDS-2017/CIC-IDS-2017.json",
    "/data2/qcy/data_pattern_alpaca/v2/traffic_and_iot/train/CIC_IOT_2023/CIC_IOT_2023.json",
    "/data2/qcy/data_pattern_alpaca/v2/traffic_and_iot/train/NF-UNSW-NB15-v2/NF-UNSW-NB15-v2.json",
    "/data2/qcy/data_pattern_alpaca/v2/traffic_and_iot/train/CIC-BCCC-NRC2024/CIC-BCCC-NRC2024.json",
    "/data2/qcy/data_pattern_alpaca/v2/traffic_and_iot/train/NF-ToN-IoT-v2/NF-ToN-IoT-v2.json",
    "/data2/qcy/data_pattern_alpaca/v2/traffic_and_iot/train/iscxids2012/iscxids2012.json",
    "/data2/qcy/data_pattern_alpaca/v2/traffic_and_iot/train/CIC-BoT-IoT/CIC-BoT-IoT.json",
    "/data2/qcy/data_pattern_alpaca/v2/traffic_and_iot/train/NF-BoT-IoT-v2/NF-BoT-IoT-v2.json",
    "/data2/qcy/data_pattern_alpaca/v2/traffic_and_iot/train/NF-UQ-NIDS-v2/NF-UQ-NIDS-v2.json",
    "/data2/qcy/data_pattern_alpaca/v2/traffic_and_iot/train/CICEVSE2024/CICEVSE2024.json",
    "/data2/qcy/data_pattern_alpaca/v2/url_update/train/Feodo-Tracker-ipblocklist/Feodo-Tracker-ipblocklist.json",
    "/data2/qcy/data_pattern_alpaca/v2/url_update/train/ISCX-URL2016/ISCX-URL2016_no_class_evidence.json",
    "/data2/qcy/data_pattern_alpaca/v2/url_update/train/Malicious-URLs/Malicious-URLs.json",
    "/data2/qcy/data_pattern_alpaca/v2/vulnerablity/aliyun/converted/training/output_data.json",
    "/data2/qcy/data_pattern_alpaca/v2/vulnerablity/exploit-db/converted/training/output_data.json",
    "/data2/qcy/data_pattern_alpaca/v2/vulnerablity/talos/converted/training/output_data.json"
]

# 需要重复2次的文件路径
double_files = [
    "/data2/qcy/data_pattern_alpaca/v2/traffic_and_iot/train/NF-CSE-CIC-IDS2018-v2/NF-CSE-CIC-IDS2018-v2.json",  # L121
    "/data2/qcy/data_pattern_alpaca/v2/traffic_and_iot/train/CIC-IDS-2017/CIC-IDS-2017.json",  # L151
    "/data2/qcy/data_pattern_alpaca/v2/traffic_and_iot/train/iscxids2012/iscxids2012.json",  # L181
    "/data2/qcy/data_pattern_alpaca/v2/traffic_and_iot/train/NF-UQ-NIDS-v2/NF-UQ-NIDS-v2.json",  # L199
    "/data2/qcy/data_pattern_alpaca/v2/traffic_and_iot/train/NF-UNSW-NB15-v2/NF-UNSW-NB15-v2.json",  # L163
    "/data2/qcy/data_pattern_alpaca/v2/threat/SABU-Alert/converted/SABU-Alert_train_no_answer.json"  # L115
]

# 需要重复50次的文件路径
fifty_files = [
    "/data2/qcy/data_pattern_alpaca/alpaca_fallback/Pulsedive-Threats.json"  # L109
]

def load_json_list(file_path):
    """加载JSON文件并返回其中的列表"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                print(f"警告: {file_path} 不是一个JSON列表，跳过")
                return []
    except Exception as e:
        print(f"错误: 无法加载 {file_path}: {e}")
        return []

def main():
    """主函数，拼接所有JSON列表"""
    merged_list = []
    
    # 加载基础文件
    print("加载基础文件...")
    for file_path in base_files:
        if os.path.exists(file_path):
            data = load_json_list(file_path)
            merged_list.extend(data)
            print(f"加载 {file_path}: {len(data)} 条记录")
        else:
            print(f"警告: {file_path} 不存在，跳过")
    
    # 加载需要重复2次的文件
    print("\n加载需要重复2次的文件...")
    for file_path in double_files:
        if os.path.exists(file_path):
            data = load_json_list(file_path)
            merged_list.extend(data)  # 第二次
            print(f"加载 {file_path}: {len(data)} 条记录 (重复第2次)")
        else:
            print(f"警告: {file_path} 不存在，跳过")
    
    # 加载需要重复50次的文件
    print("\n加载需要重复50次的文件...")
    for file_path in fifty_files:
        if os.path.exists(file_path):
            data = load_json_list(file_path)
            for i in range(49):  # 已经在基础文件中加载了1次，所以再加载49次
                merged_list.extend(data)
                print(f"加载 {file_path}: {len(data)} 条记录 (重复第{i+2}次)")
        else:
            print(f"警告: {file_path} 不存在，跳过")
    
    # 保存结果
    output_file = "/data2/qcy/data_pattern_alpaca/merged_json_list.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_list, f, ensure_ascii=False, indent=2)
    
    print(f"\n完成! 总共拼接了 {len(merged_list)} 条记录到 {output_file}")

if __name__ == "__main__":
    main()