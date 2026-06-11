import json
import os
from transformers import AutoTokenizer

# 1. 设置环境变量使用镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def count_alpaca_tokens(file_paths, output_txt, model_id, hf_token=None):
    """
    使用指定的 Gemma 模型统计 Token
    """
    print(f"正在从镜像站加载分词器: {model_id}...")
    
    try:
        # 2. 加载分词器 (如果模型是受限的，需要传入 token)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            token=hf_token,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"无法加载分词器，请检查 Token 或网络状态: {e}")
        return

    results = []
    
    for path in file_paths:
        if not os.path.exists(path):
            print(f"警告: 文件未找到 - {path}")
            results.append(f"文件: {path}\n状态: 文件不存在\n" + "-"*30)
            continue
        
        print(f"正在处理: {path}")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            token_counts = []
            for entry in data:
                # 拼接 Alpaca 字段
                text = (
                    entry.get("instruction", "") + 
                    entry.get("input", "") + 
                    entry.get("output", "")
                )
                # 使用 gemma 分词器
                tokens = tokenizer.encode(text)
                token_counts.append(len(tokens))
            
            if token_counts:
                avg_tokens = sum(token_counts) / len(token_counts)
                max_tokens = max(token_counts)
                results.append(
                    f"文件路径: {path}\n"
                    f"样本总数: {len(token_counts)}\n"
                    f"平均 Token 数: {avg_tokens:.2f}\n"
                    f"最大 Token 数: {max_tokens}\n"
                    + "-"*30
                )
            
        except Exception as e:
            results.append(f"文件路径: {path}\n状态: 处理出错 ({str(e)})\n" + "-"*30)

    with open(output_txt, 'w', encoding='utf-8') as f_out:
        f_out.write("\n".join(results))
    print(f"\n统计完成！结果已保存至: {output_txt}")

if __name__ == "__main__":
    # 在此处填入你的 Hugging Face Read Token
    # 获取地址: https://huggingface.co/settings/tokens
    MY_TOKEN = "wdM" 
    
    model_path = "/data2/qcy/local_evaluation_with_vllm/gemma-3-27b-it"
    
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
    
    count_alpaca_tokens(datasets, "gemma_token_stats.txt", model_path, hf_token=MY_TOKEN)
