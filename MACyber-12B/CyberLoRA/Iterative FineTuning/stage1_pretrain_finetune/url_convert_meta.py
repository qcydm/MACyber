import json
import os
from pathlib import Path
from typing import List, Dict, Any
import traceback

# 全局变量：要处理的JSON文件绝对路径列表
JSON_FILES_TO_PROCESS = ['/data2/qcy/data_pattern/url/Feodo-Tracker-ipblocklist/converted_data/converted_ip_threat_intelligence_train_extracted.json',
 '/data2/qcy/data_pattern/url/ISCX-URL2016/converted_data/converted_All_BestFirst_train_extracted.json',
 '/data2/qcy/data_pattern/url/ISCX-URL2016/converted_data/converted_All_Infogain_train_extracted.json',
 '/data2/qcy/data_pattern/url/ISCX-URL2016/converted_data/converted_Defacement_BestFirst_train_extracted.json',
 '/data2/qcy/data_pattern/url/ISCX-URL2016/converted_data/converted_Malware_BestFirst_train_extracted.json',
 '/data2/qcy/data_pattern/url/ISCX-URL2016/converted_data/converted_Spam_Infogain_train_extracted.json',
 '/data2/qcy/data_pattern/url/Malicious-URLs/converted_data/converted_url_day_train_extracted.json']

def transform_json_content(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    转换JSON数据格式
    
    参数:
        data: 原始JSON数据列表
        
    返回:
        转换后的JSON数据列表
    """
    transformed_data = []
    
    for item in data:
        # 创建新的项目副本，避免修改原始数据
        new_item = item.copy()
        
        # 确保meta字段存在
        if "meta" in new_item and isinstance(new_item["meta"], dict):
            meta = new_item["meta"]
            
            # 保存原始category和subcategory值
            old_category = meta.get("category", "")
            old_subcategory = meta.get("subcategory", "")
            
            # 构建新的subcategory值
            new_subcategory = f"{old_category}-{old_subcategory}"
            
            # 更新meta字段
            meta["category"] = "url"  # 固定为小写"url"，根据您的示例
            meta["subcategory"] = new_subcategory
            
            # 更新项目中的meta
            new_item["meta"] = meta
        
        transformed_data.append(new_item)
    
    return transformed_data

def process_json_file(input_file_path: str) -> bool:
    """
    处理单个JSON文件
    
    参数:
        input_file_path: 输入文件路径
        
    返回:
        处理成功返回True，否则返回False
    """
    try:
        # 验证文件是否存在
        if not os.path.exists(input_file_path):
            print(f"错误: 文件不存在 - {input_file_path}")
            return False
        
        # 读取JSON文件
        with open(input_file_path, 'r', encoding='utf-8') as file:
            try:
                json_data = json.load(file)
            except json.JSONDecodeError as e:
                print(f"错误: JSON格式无效 - {input_file_path}")
                print(f"详细信息: {e}")
                return False
        
        # 验证数据格式
        if not isinstance(json_data, list):
            print(f"错误: JSON数据不是列表格式 - {input_file_path}")
            return False
        
        # 转换数据
        transformed_data = transform_json_content(json_data)
        
        # 构建输出文件路径（与原文件相同目录，添加_transformed后缀）
        input_path = Path(input_file_path)
        output_file_name = f"{input_path.stem}_transformed{input_path.suffix}"
        output_file_path = input_path.parent / output_file_name
        
        # 写入转换后的数据
        with open(output_file_path, 'w', encoding='utf-8') as file:
            json.dump(transformed_data, file, indent=2, ensure_ascii=False)
        
        print(f"成功: 已处理文件 {input_file_path}")
        print(f"      输出文件: {output_file_path}")
        
        return True
        
    except Exception as e:
        print(f"处理文件时发生错误: {input_file_path}")
        print(f"错误详情: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """
    主函数，处理所有JSON文件
    """
    print("=" * 60)
    print("JSON文件格式转换工具")
    print("=" * 60)
    
    # 检查是否有文件需要处理
    if not JSON_FILES_TO_PROCESS:
        print("警告: 没有指定要处理的JSON文件")
        print("请在 JSON_FILES_TO_PROCESS 列表中添加文件路径")
        return
    
    print(f"找到 {len(JSON_FILES_TO_PROCESS)} 个文件需要处理")
    print()
    
    # 处理每个文件
    success_count = 0
    failure_count = 0
    
    for file_path in JSON_FILES_TO_PROCESS:
        print(f"正在处理: {file_path}")
        if process_json_file(file_path):
            success_count += 1
        else:
            failure_count += 1
        print()
    
    # 输出处理结果
    print("=" * 60)
    print("处理完成!")
    print(f"成功: {success_count} 个文件")
    print(f"失败: {failure_count} 个文件")
    print("=" * 60)

if __name__ == "__main__":
    # 执行主函数
    main()