import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Union
import logging
from dataclasses import dataclass

FILE_PATHS = ['/data2/qcy/data_pattern/url/Feodo-Tracker-ipblocklist/converted_data/converted_ip_threat_intelligence_train.json',
 '/data2/qcy/data_pattern/url/ISCX-URL2016/converted_data/converted_All_BestFirst_train.json',
 '/data2/qcy/data_pattern/url/ISCX-URL2016/converted_data/converted_All_Infogain_train.json',
 '/data2/qcy/data_pattern/url/ISCX-URL2016/converted_data/converted_Defacement_BestFirst_train.json',
 '/data2/qcy/data_pattern/url/ISCX-URL2016/converted_data/converted_Malware_BestFirst_train.json',
 '/data2/qcy/data_pattern/url/ISCX-URL2016/converted_data/converted_Spam_Infogain_train.json',
 '/data2/qcy/data_pattern/url/Malicious-URLs/converted_data/converted_url_day_train.json']

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ConversionResult:
    """转换结果数据类"""
    input_path: str
    output_path: str
    success: bool
    error_message: Optional[str] = None
    samples_count: int = 0

class JSONBatchConverter:
    """JSON格式批量转换器"""
    
    def __init__(
        self,
        output_suffix: str = "_extracted",
        backup_original: bool = False,
        indent_size: int = 2
    ):
        """
        初始化转换器
        
        Args:
            output_suffix: 输出文件后缀
            backup_original: 是否备份原始文件
            indent_size: JSON缩进大小
        """
        self.output_suffix = output_suffix
        self.backup_original = backup_original
        self.indent_size = indent_size
        
    def convert_single_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None
    ) -> ConversionResult:
        """
        转换单个文件
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径（可选）
            
        Returns:
            ConversionResult: 转换结果
        """
        input_path = Path(input_path)
        
        try:
            # 验证输入文件
            if not input_path.exists():
                return ConversionResult(
                    input_path=str(input_path),
                    output_path="",
                    success=False,
                    error_message=f"输入文件不存在: {input_path}"
                )
            
            if input_path.suffix.lower() != '.json':
                logger.warning(f"文件 {input_path} 不是JSON文件扩展名，但仍尝试处理")
            
            # 读取输入文件
            with open(input_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    return ConversionResult(
                        input_path=str(input_path),
                        output_path="",
                        success=False,
                        error_message=f"JSON解析失败: {str(e)}"
                    )
            
            # 检查数据格式
            if not isinstance(data, dict):
                return ConversionResult(
                    input_path=str(input_path),
                    output_path="",
                    success=False,
                    error_message="JSON根元素不是字典对象"
                )
            
            if 'samples' not in data:
                return ConversionResult(
                    input_path=str(input_path),
                    output_path="",
                    success=False,
                    error_message="JSON中没有找到'samples'键"
                )
            
            samples_list = data['samples']
            
            if not isinstance(samples_list, list):
                return ConversionResult(
                    input_path=str(input_path),
                    output_path="",
                    success=False,
                    error_message="'samples'不是列表类型"
                )
            
            # 确定输出文件路径
            if output_path is None:
                output_path = self._generate_output_path(input_path)
            else:
                output_path = Path(output_path)
            
            # 确保输出目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 备份原始文件（如果需要）
            if self.backup_original:
                self._backup_file(input_path)
            
            # 写入输出文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(samples_list, f, ensure_ascii=False, indent=self.indent_size)
            
            logger.info(f"成功转换: {input_path} -> {output_path}")
            
            return ConversionResult(
                input_path=str(input_path),
                output_path=str(output_path),
                success=True,
                samples_count=len(samples_list)
            )
            
        except Exception as e:
            logger.error(f"处理文件 {input_path} 时发生错误: {str(e)}")
            return ConversionResult(
                input_path=str(input_path),
                output_path="",
                success=False,
                error_message=str(e)
            )
    
    def convert_batch(
        self,
        file_paths: List[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None
    ) -> List[ConversionResult]:
        """
        批量转换文件
        
        Args:
            file_paths: 文件路径列表
            output_dir: 输出目录（可选，默认与输入文件相同目录）
            
        Returns:
            List[ConversionResult]: 转换结果列表
        """
        results = []
        total_files = len(file_paths)
        
        logger.info(f"开始批量处理 {total_files} 个文件")
        
        for i, file_path in enumerate(file_paths, 1):
            logger.info(f"处理文件 {i}/{total_files}: {file_path}")
            
            # 如果需要指定输出目录
            if output_dir:
                input_path = Path(file_path)
                output_path = Path(output_dir) / f"{input_path.stem}{self.output_suffix}{input_path.suffix}"
                result = self.convert_single_file(file_path, output_path)
            else:
                result = self.convert_single_file(file_path)
            
            results.append(result)
        
        # 统计结果
        self._print_summary(results)
        
        return results
    
    def _generate_output_path(self, input_path: Path) -> Path:
        """
        生成输出文件路径
        
        Args:
            input_path: 输入文件路径
            
        Returns:
            输出文件路径
        """
        # 保留原始目录，只修改文件名
        stem = input_path.stem
        suffix = input_path.suffix
        return input_path.parent / f"{stem}{self.output_suffix}{suffix}"
    
    def _backup_file(self, file_path: Path) -> None:
        """
        备份文件
        
        Args:
            file_path: 文件路径
        """
        backup_path = file_path.parent / f"{file_path.stem}_backup{file_path.suffix}"
        try:
            import shutil
            shutil.copy2(file_path, backup_path)
            logger.debug(f"已备份原始文件到: {backup_path}")
        except Exception as e:
            logger.warning(f"备份文件失败: {str(e)}")
    
    def _print_summary(self, results: List[ConversionResult]) -> None:
        """
        打印转换结果摘要
        
        Args:
            results: 转换结果列表
        """
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        logger.info("\n" + "="*50)
        logger.info("转换结果摘要:")
        logger.info(f"总文件数: {len(results)}")
        logger.info(f"成功: {len(successful)}")
        logger.info(f"失败: {len(failed)}")
        
        if successful:
            total_samples = sum(r.samples_count for r in successful)
            logger.info(f"总样本数: {total_samples}")
        
        if failed:
            logger.warning("\n失败的文件:")
            for result in failed:
                logger.warning(f"  - {result.input_path}: {result.error_message}")
        
        logger.info("="*50)

def get_file_paths_from_user() -> List[Path]:
    """
    从用户输入获取文件路径列表
    
    Returns:
        文件路径列表
    """
    paths = []
    
    print("请输入文件路径列表（每行一个，空行结束）：")
    print("示例:")
    print("  /path/to/file1.json")
    print("  /path/to/file2.json")
    print("  /path/to/file3.json")
    print()
    
    while True:
        line = input().strip()
        if not line:  # 空行结束输入
            break
        
        path = Path(line)
        if path.exists():
            paths.append(path)
        else:
            logger.warning(f"文件不存在，已跳过: {line}")
    
    return paths

def load_file_paths_from_txt(file_path: Union[str, Path]) -> List[Path]:
    """
    从文本文件加载文件路径列表
    
    Args:
        file_path: 文本文件路径
        
    Returns:
        文件路径列表
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"路径列表文件不存在: {file_path}")
        return []
    
    paths = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # 跳过空行和注释
                path = Path(line)
                if path.exists():
                    paths.append(path)
                else:
                    logger.warning(f"文件不存在，已跳过: {line}")
    
    return paths

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='批量转换JSON文件格式')
    parser.add_argument('--input-files', nargs='+', help='输入文件路径列表')
    parser.add_argument('--list-file', help='包含文件路径列表的文本文件')
    parser.add_argument('--output-dir', help='输出目录（默认与输入文件相同目录）')
    parser.add_argument('--suffix', default='_extracted', help='输出文件后缀')
    parser.add_argument('--backup', action='store_true', help='备份原始文件')
    parser.add_argument('--indent', type=int, default=2, help='JSON缩进大小')
    parser.add_argument('--verbose', action='store_true', help='详细输出模式')
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # 初始化转换器
    converter = JSONBatchConverter(
        output_suffix=args.suffix,
        backup_original=args.backup,
        indent_size=args.indent
    )
    
    # 获取文件路径列表
    file_paths = []
    
    if args.list_file:
        file_paths = load_file_paths_from_txt(args.list_file)
    elif args.input_files:
        file_paths = [Path(p) for p in args.input_files]
    else:
        # 交互式获取文件路径
        # file_paths = get_file_paths_from_user()
        file_paths = FILE_PATHS
    
    if not file_paths:
        logger.error("没有找到有效的文件路径")
        sys.exit(1)
    
    # 执行批量转换
    results = converter.convert_batch(file_paths, args.output_dir)
    
    # 如果有失败的文件，返回非零退出码
    if any(not r.success for r in results):
        sys.exit(1)

# 示例使用方式
def example_usage():
    """示例使用方式"""
    
    # 方式1: 直接在代码中指定文件路径列表
    file_paths = [
        "/path/to/data1.json",
        "/path/to/data2.json",
        "/path/to/data3.json",
    ]
    
    converter = JSONBatchConverter()
    results = converter.convert_batch(file_paths)
    
    # 方式2: 从文本文件加载路径列表
    # 创建一个文本文件 file_list.txt，内容如下:
    # /path/to/data1.json
    # /path/to/data2.json
    # /path/to/data3.json
    
    paths = load_file_paths_from_txt("file_list.txt")
    converter.convert_batch(paths)
    
    # 方式3: 使用命令行
    # python script.py --input-files /path/to/data1.json /path/to/data2.json
    # python script.py --list-file file_list.txt --suffix "_converted" --backup

if __name__ == "__main__":
    main()