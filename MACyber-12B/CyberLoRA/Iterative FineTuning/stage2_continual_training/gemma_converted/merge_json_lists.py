#!/usr/bin/env python3
import json
import os
import random  # 新增：用于随机选择数据

def load_json_list(file_path):
    """加载JSON文件并返回其中的列表（保留原有健壮的异常处理）"""
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

def get_all_json_files():
    """获取当前目录下所有JSON文件（排除输出文件避免循环读取）"""
    current_dir = os.getcwd()  # 获取当前工作目录
    json_files = []
    # 遍历当前目录所有文件
    for file_name in os.listdir(current_dir):
        # 筛选：以.json结尾 + 不是输出文件（避免读取自己生成的文件）
        if file_name.endswith('.json') and file_name != 'merged_random_data.json':
            file_path = os.path.join(current_dir, file_name)
            if os.path.isfile(file_path):  # 确保是文件而非目录
                json_files.append(file_path)
    return json_files

def main():
    """主函数：读取所有JSON文件→随机抽样→合并→输出"""
    merged_list = []
    # 1. 获取当前目录所有JSON文件
    json_files = get_all_json_files()
    if not json_files:
        print("警告: 当前目录未找到任何JSON文件！")
        return
    
    # 2. 遍历每个JSON文件，随机抽样
    print(f"找到 {len(json_files)} 个JSON文件，开始处理...\n")
    for file_path in json_files:
        # 加载文件数据
        raw_data = load_json_list(file_path)
        data_count = len(raw_data)
        if data_count == 0:
            print(f"{file_path}: 无有效数据，跳过")
            continue
        
        # 随机选择200条（不够则全选）
        sample_count = min(200, data_count)  # 确定抽样数量
        sampled_data = random.sample(raw_data, sample_count)  # 随机抽样
        
        # 合并到总列表
        merged_list.extend(sampled_data)
        print(f"{file_path}: 原始{data_count}条 → 抽样{sample_count}条")
    
    # 3. 保存合并结果
    output_file = "./merged_random_data.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_list, f, ensure_ascii=False, indent=2)
    
    print(f"\n完成！总共合并 {len(merged_list)} 条数据 → 输出到 {output_file}")

if __name__ == "__main__":
    main()