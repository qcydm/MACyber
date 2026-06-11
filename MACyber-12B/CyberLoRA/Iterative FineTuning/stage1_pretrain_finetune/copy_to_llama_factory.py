import os
import shutil

# 源文件列表
datasets = [
    "/data2/qcy/data_pattern_alpaca/dns/converted/train/bccc_2024_mal_20251016_003244_train.json",
    "/data2/qcy/data_pattern_alpaca/dns/converted/train/DNS_exfiltration_benign_1000_train.json",
    "/data2/qcy/data_pattern_alpaca/dns/converted/train/CIRA_benign_1500_train.json",
    "/data2/qcy/data_pattern_alpaca/dns/converted/train/DNS_exfiltration_light_1000_train.json",
    "/data2/qcy/data_pattern_alpaca/dns/converted/train/CIRA_malicious_3500_train.json",
    "/data2/qcy/data_pattern_alpaca/dns/converted/train/DNS_exfiltration_heavy_1000_train.json",
    "/data2/qcy/data_pattern_alpaca/dns/converted/train/bccc_2024_exf_20251016_103047_train.json",
    "/data2/qcy/data_pattern_alpaca/log/split-anroid/training-anroid/output_data_cleaned.json",
    "/data2/qcy/data_pattern_alpaca/log/split-hdfs/training-hdfs-5007_cleaned.json",
    "/data2/qcy/data_pattern_alpaca/log/split-linux/training-linux/output_data_cleaned.json",
    "/data2/qcy/data_pattern_alpaca/log/split-proxifier/test-proxifier/output_data_cleaned.json",
    "/data2/qcy/data_pattern_alpaca/log/split-supercomputer/test-supercom-22/output_data_cleaned.json",
    "/data2/qcy/data_pattern_alpaca/alpaca_fallback/Pulsedive-Threats.json",
    "/data2/qcy/data_pattern_alpaca/threat/SABU-Alert/converted/SABU-Alert_train_no_answer.json",
    "/data2/qcy/data_pattern_alpaca/traffic_and_iot/train/NF-CSE-CIC-IDS2018-v2/NF-CSE-CIC-IDS2018-v2.json",
    "/data2/qcy/data_pattern_alpaca/traffic_and_iot/train/CIC-ToN-IoT/CIC-ToN-IoT.json",
    "/data2/qcy/data_pattern_alpaca/traffic_and_iot/train/CICIoMT 2024/CICIoMT 2024.json",
    "/data2/qcy/data_pattern_alpaca/traffic_and_iot/train/CICADA-IIoT2024/CICADA-IIoT2024.json",
    "/data2/qcy/data_pattern_alpaca/traffic_and_iot/train/CIC-IoT-DIAD2024/CIC-IoT-DIAD2024.json",
    "/data2/qcy/data_pattern_alpaca/traffic_and_iot/train/CIC-IDS-2017/CIC-IDS-2017.json",
    "/data2/qcy/data_pattern_alpaca/traffic_and_iot/train/CIC_IOT_2023/CIC_IOT_2023.json",
    "/data2/qcy/data_pattern_alpaca/traffic_and_iot/train/NF-UNSW-NB15-v2/NF-UNSW-NB15-v2.json",
    "/data2/qcy/data_pattern_alpaca/traffic_and_iot/train/CIC-BCCC-NRC2024/CIC-BCCC-NRC2024.json",
    "/data2/qcy/data_pattern_alpaca/traffic_and_iot/train/NF-ToN-IoT-v2/NF-ToN-IoT-v2.json",
    "/data2/qcy/data_pattern_alpaca/traffic_and_iot/train/iscxids2012/iscxids2012.json",
    "/data2/qcy/data_pattern_alpaca/traffic_and_iot/train/CIC-BoT-IoT/CIC-BoT-IoT.json",
    "/data2/qcy/data_pattern_alpaca/traffic_and_iot/train/NF-BoT-IoT-v2/NF-BoT-IoT-v2.json",
    "/data2/qcy/data_pattern_alpaca/traffic_and_iot/train/NF-UQ-NIDS-v2/NF-UQ-NIDS-v2.json",
    "/data2/qcy/data_pattern_alpaca/traffic_and_iot/train/CICEVSE2024/CICEVSE2024.json",
    "/data2/qcy/data_pattern_alpaca/url/ISCX-URL2016/converted_data/converted_All_BestFirst_train_extracted_transformed.json",
    "/data2/qcy/data_pattern_alpaca/url/ISCX-URL2016/converted_data/converted_All_Infogain_train_extracted_transformed.json",
    "/data2/qcy/data_pattern_alpaca/url/ISCX-URL2016/converted_data/converted_Defacement_BestFirst_train_extracted_transformed.json",
    "/data2/qcy/data_pattern_alpaca/url/ISCX-URL2016/converted_data/converted_Malware_BestFirst_train_extracted_transformed.json",
    "/data2/qcy/data_pattern_alpaca/url/ISCX-URL2016/converted_data/converted_Spam_Infogain_train_extracted_transformed.json",
    "/data2/qcy/data_pattern_alpaca/url/Malicious-URLs/converted_data/converted_url_day_train_extracted_transformed.json",
    "/data2/qcy/data_pattern_alpaca/vulnerablity/aliyun/converted/training/output_data.json",
    "/data2/qcy/data_pattern_alpaca/vulnerablity/exploit-db/converted/training/output_data.json",
    "/data2/qcy/data_pattern_alpaca/vulnerablity/talos/converted/training/output_data.json"
]

target_dir = "/data2/qcy/LLaMA-Factory/data"

def copy_files():
    # 确保目标目录存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created directory: {target_dir}")

    for source_path in datasets:
        if not os.path.exists(source_path):
            print(f"⚠️ Warning: File not found: {source_path}")
            continue

        # 获取文件名
        file_name = os.path.basename(source_path)
        # 获取父目录名称，用于区分重名文件
        parent_dir = os.path.basename(os.path.dirname(source_path))
        
        # 为了防止重名覆盖（如 output_data.json），给文件名加上父目录前缀
        new_file_name = f"{parent_dir}_{file_name}"
        destination = os.path.join(target_dir, new_file_name)

        try:
            shutil.copy2(source_path, destination)
            print(f"✅ Copied: {new_file_name}")
        except Exception as e:
            print(f"❌ Error copying {file_name}: {e}")

if __name__ == "__main__":
    copy_files()
    print("\n✨ All done!")
