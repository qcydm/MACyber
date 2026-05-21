import json
import os
import time
import logging
import threading
import argparse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import re

def normalize(s: str) -> str:
    """全小写 + 去空格 + 去连字符"""
    return re.sub(r'\s+', '', s).lower()

# DashScope judge model configuration
API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
MODEL_NAME = os.getenv("DASHSCOPE_MODEL", "qwen3-max")

# API调用配置
MAX_RETRIES = 3
RETRY_DELAY = 2
API_DELAY = 0.5
REQUEST_TIMEOUT = 30
MAX_WORKERS = 16  # 并行线程数
SAVE_INTERVAL = 100  # 每处理多少条保存一次

# 评分权重配置（百分制+加权模式）
WEIGHTS = {
    'reasoning': 0.40,   # 推理过程权重 40%
    'official': 0.30,    # 威胁类型权重 30%
    'action': 0.20,      # 响应动作权重 20%
    'severity': 0.10     # 严重程度权重 10%
}

# 各维度满分
MAX_SCORES = {
    'reasoning': 40,
    'official': 30,
    'action': 20,
    'severity': 10
}

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def call_api_with_retry(messages: list, max_retries: int = MAX_RETRIES):
    """带重试机制的API调用"""
    import dashscope
    from dashscope import Generation

    dashscope.api_key = API_KEY
    for attempt in range(max_retries):
        try:
            response = Generation.call(
                api_key=API_KEY,
                model=MODEL_NAME,
                messages=messages,
                stream=False,
                result_format="message",
                temperature=0.3,
                top_p=0.9,
                max_tokens=1500
            )

            if response.status_code == 200:
                return response.output.choices[0].message.content.strip()
            elif response.status_code == 429:
                wait_time = RETRY_DELAY * (attempt + 1)
                logger.warning(f"API限流，等待{wait_time}秒后重试...")
                time.sleep(wait_time)
            else:
                logger.error(f"API返回错误状态码: {response.status_code}")

        except Exception as e:
            logger.error(f"API调用异常（尝试{attempt + 1}/{max_retries}）: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY)

    return None

def score_reasoning(standard_evidence, standard_analysis, model_evidence, model_analysis):
    """
    Call a large model to score the reasoning process (full score 40 points)

    Args:
        standard_evidence: evidence of the standard answer
        standard_analysis: analysis of the standard answer
        model_evidence: evidence from the tested model
        model_analysis: analysis from the tested model

    Returns:
        dict: contains evidence_score, analysis_score, and total score (full score 40)
    """
    try:
        system_prompt = """You are an expert judge in the field of cybersecurity, responsible for evaluating the quality of the model's feature analysis process and results.

Your task is: compare the standard answer with the tested model's output, and score the reasoning process of the tested model.

Scoring criteria:
1. Evidence score (20 points):
   - Whether key features are identified
   - Whether evidence is relevant to label and reasoning
   - Semantic consistency and similarity with the standard answer's evidence

2. Analysis score (20 points):
   - Whether reasoning logic is clear (First check → then verify → finally confirm → because → classify)
   - Whether causal relationships are reasonable
   - Whether conclusions are well supported
   - Semantic consistency with the provided answer
   - Similarity with the provided answer
   
Please output the scoring result in JSON format:
{
    "evidence_score": <score between 0-20>,
    "analysis_score": <score between 0-20>,
    "evidence_feedback": "<brief evaluation of evidence>",
    "analysis_feedback": "<brief evaluation of analysis>"
}"""

        user_prompt = f"""## Standard Answer

### Evidence:
{json.dumps(standard_evidence, ensure_ascii=False, indent=2)}

### Analysis:
{standard_analysis}

---

## Tested Model Output

### Evidence:
{json.dumps(model_evidence, ensure_ascii=False, indent=2)}

### Analysis:
{model_analysis}

---

Please refer to the standard answer, evaluate the reasoning quality of the tested model and score it."""

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        result_text = call_api_with_retry(messages)

        if result_text:
            # Try to extract JSON
            try:
                # Extract JSON that may be wrapped in markdown
                if '```json' in result_text:
                    result_text = result_text.split('```json')[1].split('```')[0].strip()
                elif '```' in result_text:
                    result_text = result_text.split('```')[1].split('```')[0].strip()

                score_result = json.loads(result_text)

                # Calculate total score (full score 40 points)
                total_score = score_result['evidence_score'] + score_result['analysis_score']
                score_result['reasoning_total_score'] = total_score

                return score_result
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed: {result_text}")
                return {
                    "evidence_score": 0,
                    "analysis_score": 0,
                    "reasoning_total_score": 0,
                    "evidence_feedback": "Scoring failed",
                    "analysis_feedback": f"JSON parsing error: {str(e)}"
                }
        else:
            return {
                "evidence_score": 0,
                "analysis_score": 0,
                "reasoning_total_score": 0,
                "evidence_feedback": "API call failed",
                "analysis_feedback": "API call failed"
            }

    except Exception as e:
        logger.error(f"Error occurred during scoring: {str(e)}")
        return {
            "evidence_score": 0,
            "analysis_score": 0,
            "reasoning_total_score": 0,
            "evidence_feedback": f"Error: {str(e)}",
            "analysis_feedback": f"Error: {str(e)}"
        }

def calculate_severity_score(standard_severity, model_severity):
    """
    计算severity字段的得分（满分10分）

    规则：
    - 完全匹配：10分
    - 偏差1级：2分（如 high↔medium, low↔suspicious）
    - 偏差2级+：0分

    Args:
        standard_severity: 标准答案的severity
        model_severity: 模型输出的severity

    Returns:
        int: severity得分（0/2/10）
    """
    severity_levels = ['benign', 'suspicious', 'low', 'medium', 'high']

    # 标准化输入
    standard = str(standard_severity or '').strip().lower()
    model = str(model_severity or '').strip().lower()

    # 完全匹配
    if standard == model:
        return 10

    # 检查是否都在有效范围内
    if standard not in severity_levels or model not in severity_levels:
        return 0

    # 计算级别差异
    standard_idx = severity_levels.index(standard)
    model_idx = severity_levels.index(model)
    diff = abs(standard_idx - model_idx)

    # 偏差1级得2分
    if diff == 1:
        return 2

    # 偏差2级及以上得0分
    return 0

def evaluate_single_sample(standard_item, model_output, index, pbar=None, stats=None, lock=None):
    """
    评估单个样本（四阶段评分 - 百分制+加权）

    评分结构（总分100）：
    - evidence & analysis（推理过程）：40分 → 转百分制 × 40%
    - official（威胁类型）：30分 → 转百分制 × 30%
    - action（响应动作）：20分 → 转百分制 × 20%
    - severity（严重程度）：10分 → 转百分制 × 10%

    Args:
        standard_item: 标准答案数据
        model_output: 被测模型输出
        index: 样本索引
        pbar: 进度条
        stats: 统计信息
        lock: 线程锁

    Returns:
        dict: 评估结果
    """
    try:
        # # 获取标准答案
        # standard_action = standard_item['response']['action']
        # standard_official = standard_item['label']['official']
        # standard_severity = standard_item['label']['severity']
        # standard_evidence = standard_item['reasoning']['evidence']
        # standard_analysis = standard_item['reasoning']['analysis']

        # 先取到真正的数据块
        # json_block = standard_item.get('json', standard_item)   # 如果已经是单层就回退
        # label_block = json_block.get('label', {})
        # reason_block = json_block.get('reasoning', {})
        # response_block = json_block.get('response', {})
        # 适配两种结构
        if 'json' in standard_item and 'label' in standard_item['json']:
            json_block = standard_item['json']          # 双层（generated）
        else:
            json_block = standard_item                  # 单层（output_data）

        label_block    = json_block.get('label', {})
        reason_block   = json_block.get('reasoning', {})
        response_block = json_block.get('response', {})
        
        standard_action = response_block.get('action', '')
        standard_official = label_block.get('official', '')
        standard_severity = label_block.get('severity', '')
        standard_evidence = reason_block.get('evidence', [])
        standard_analysis = reason_block.get('analysis', '')
        # 获取模型输出
        model_action = str(model_output.get('action', '') or '').strip().lower()
        model_official = str(model_output.get('official', '') or '').strip().lower()
        model_severity = str(model_output.get('severity', '') or '').strip().lower()
        model_evidence = model_output.get('evidence', [])
        model_analysis = model_output.get('analysis', '')

        # 初始化结果
        result = {
            'index': index,
            'standard_action': standard_action,
            'standard_official': standard_official,
            'standard_severity': standard_severity,
            'model_action': model_action,
            'model_official': model_official,
            'model_severity': model_severity,
            'action_correct': False,
            'official_correct': False,
            'severity_correct': False,
            'raw_scores': {
                'reasoning_score': 0,    # 原始得分 0-40
                'official_score': 0,      # 原始得分 0-30
                'action_score': 0,        # 原始得分 0-20
                'severity_score': 0       # 原始得分 0-10
            },
            'percentage_scores': {
                'reasoning_percentage': 0.0,  # 百分制 0-100
                'official_percentage': 0.0,    # 百分制 0-100
                'action_percentage': 0.0,      # 百分制 0-100
                'severity_percentage': 0.0     # 百分制 0-100
            },
            'total_score': 0.0,  # 加权总分 0-100
            'evaluation': {}
        }

        # 判断三个字段是否正确
        action_match = (model_action == standard_action)
        official_match = normalize(model_official) == normalize(standard_official)
        # official_match = (model_official == standard_official)
        severity_match = (model_severity == standard_severity)

        result['action_correct'] = action_match
        result['official_correct'] = official_match
        result['severity_correct'] = severity_match

        # 计算各字段得分（原始分数 + 百分制）
        # 1. Official得分（满分30分）
        official_score = 30 if official_match else 0
        official_percentage = (official_score / MAX_SCORES['official']) * 100
        result['raw_scores']['official_score'] = official_score
        result['percentage_scores']['official_percentage'] = official_percentage

        # 2. Action得分（满分20分）
        action_score = 20 if action_match else 0
        action_percentage = (action_score / MAX_SCORES['action']) * 100
        result['raw_scores']['action_score'] = action_score
        result['percentage_scores']['action_percentage'] = action_percentage

        # 3. Severity得分（满分10分：10/6/0）
        severity_score = calculate_severity_score(standard_severity, model_severity)
        severity_percentage = (severity_score / MAX_SCORES['severity']) * 100
        result['raw_scores']['severity_score'] = severity_score
        result['percentage_scores']['severity_percentage'] = severity_percentage

        # 4. 推理过程得分（满分40分）- 独立评估
        all_fields_correct = official_match and action_match and severity_match

        # 独立评分：所有样本都真实评估推理过程
        if all_fields_correct:
            # logger.info(f"样本 {index}: 三字段全对 ✓ 评估推理过程...")
            pass
        else:
            wrong_fields = []
            if not action_match:
                wrong_fields.append(f"action({model_action}≠{standard_action})")
            if not official_match:
                wrong_fields.append(f"official({model_official}≠{standard_official})")
            if not severity_match:
                wrong_fields.append(f"severity({model_severity}≠{standard_severity})")
            # logger.info(f"样本 {index}: 有字段错误 [{', '.join(wrong_fields)}]，评估推理过程...")

        # 调用API评分推理过程
        time.sleep(API_DELAY)
        score_result = score_reasoning(
            standard_evidence,
            standard_analysis,
            model_evidence,
            model_analysis
        )

        reasoning_score = score_result['reasoning_total_score']
        reasoning_percentage = (reasoning_score / MAX_SCORES['reasoning']) * 100
        result['raw_scores']['reasoning_score'] = reasoning_score
        result['percentage_scores']['reasoning_percentage'] = reasoning_percentage
        result['evaluation'] = score_result

        # 计算加权总分（百分制加权）
        total_score = (
            reasoning_percentage * WEIGHTS['reasoning'] +
            official_percentage * WEIGHTS['official'] +
            action_percentage * WEIGHTS['action'] +
            severity_percentage * WEIGHTS['severity']
        )
        result['total_score'] = round(total_score, 2)

        # 更新统计信息
        if lock:
            with lock:
                if stats:
                    if all_fields_correct:
                        stats['all_correct'] += 1
                    else:
                        stats['has_wrong'] += 1

                    # 累加百分制得分（所有样本）
                    stats['total_reasoning_percentage'] += reasoning_percentage
                    stats['total_official_percentage'] += official_percentage
                    stats['total_action_percentage'] += action_percentage
                    stats['total_severity_percentage'] += severity_percentage

                    # 分别统计各字段正确率
                    if action_match:
                        stats['action_correct'] += 1
                    if official_match:
                        stats['official_correct'] += 1
                    if severity_match:
                        stats['severity_correct'] += 1

                if pbar:
                    pbar.update(1)

        # logger.info(f"样本 {index}: 总分={total_score:.2f}/100 "
        #            f"(推理{reasoning_percentage:.1f}%*0.4 + official{official_percentage:.1f}%*0.3 + "
        #            f"action{action_percentage:.1f}%*0.2 + severity{severity_percentage:.1f}%*0.1)")

        return result

    except Exception as e:
        logger.error(f"评估样本 {index} 失败: {str(e)}")
        if lock:
            with lock:
                if stats:
                    stats['failed'] += 1
                if pbar:
                    pbar.update(1)

        return {
            'index': index,
            'error': str(e),
            'total_score': 0
        }

def update_progress_bar(pbar, stats):
    """更新进度条显示"""
    if stats:
        total_processed = stats['all_correct'] + stats['has_wrong'] + stats['failed']
        if total_processed > 0:
            all_correct_rate = (stats['all_correct'] / total_processed * 100)
            # 计算当前平均推理百分制得分
            avg_reasoning = stats['total_reasoning_percentage'] / total_processed if total_processed > 0 else 0

            pbar.set_postfix({
                '全对': stats['all_correct'],
                '有错': stats['has_wrong'],
                '全对率': f"{all_correct_rate:.1f}%",
                '平均推理分': f"{avg_reasoning:.1f}%"
            })

def save_results(output_file, results, incremental=False):
    """
    保存评测结果

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

from pathlib import Path
def evaluate_model_output(standard_file, model_output_file, output_file, max_workers=MAX_WORKERS, tiny=None):
    """
    评测模型输出

    Args:
        standard_file: 标准答案文件路径
        model_output_file: 模型输出文件路径
        output_file: 评测结果输出路径
        max_workers: 最大线程数
        tiny: Tiny模式，随机抽取指定数量的样本（可选）
    """
    try:
        summary_file = output_file.replace('.json', '_summary.json')
        should_evaluate = True
        if os.path.exists(summary_file):
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary_data = json.load(f)
                    overall_score = summary_data.get('overall_avg_score', 0)
                    if overall_score > 0:
                        print(f"发现已有评测结果 (得分: {overall_score:.2f})，跳过评测。")
                        should_evaluate = False
                    else:
                        print("发现已有评测结果，但评分为0，将重新进行评测...")
            except Exception as e:
                print(f"读取已有汇总文件失败: {e}，将重新评测...")

        if should_evaluate:
            print(f"\n{'=' * 70}")            
            print("DNS流量分析模型评测程序 (百分制+加权评分)")
            print(f"{'=' * 70}")
            print(f"标准答案: {standard_file}")
            print(f"模型输出: {model_output_file}")
            print(f"输出文件: {output_file}")
            print(f"评分模型: {MODEL_NAME}")
            print(f"并行线程数: {max_workers}")
            print(f"\n评分方法: 百分制 + 加权（独立评分）")
            print(f"权重配置: Reasoning(40%) + Official(30%) + Action(20%) + Severity(10%)")
            print(f"{'=' * 70}\n")

            # 读取标准答案
            print(f"正在读取标准答案...")
            with open(standard_file, 'r', encoding='utf-8') as f:
                standard_data = json.load(f)
            print(f"标准答案: {len(standard_data)} 条")

            # 读取模型输出
            print(f"正在读取模型输出...")
            if not Path(model_output_file).exists():
                return
            with open(model_output_file, 'r', encoding='utf-8') as f:
                model_outputs = json.load(f)
            print(f"模型输出: {len(model_outputs)} 条")

            # 检查数量是否匹配
            if len(standard_data) != len(model_outputs):
                # logger.warning(f"⚠️  数量不匹配: 标准答案 {len(standard_data)} vs 模型输出 {len(model_outputs)},将跳过评测。")
                # return

                # 取最小值
                min_len = min(len(standard_data), len(model_outputs))
                standard_data = standard_data[:min_len]
                model_outputs = model_outputs[:min_len]
                print(f"将评测前 {min_len} 条数据")

            # Tiny模式：随机抽取指定数量的样本
            if tiny is not None and tiny > 0:
                original_count = len(standard_data)
                sample_count = min(tiny, original_count)
                # 生成随机索引
                random_indices = sorted(random.sample(range(original_count), sample_count))
                # 按相同索引抽取两个数据集
                standard_data = [standard_data[i] for i in random_indices]
                model_outputs = [model_outputs[i] for i in random_indices]
                print(f"🎲 Tiny模式: 从 {original_count} 条记录中随机抽取 {sample_count} 条")

            # 统计信息
            stats = {
                'all_correct': 0,                      # 三字段全对的数量
                'has_wrong': 0,                        # 有字段错误的数量
                'action_correct': 0,                   # action正确的数量
                'official_correct': 0,                 # official正确的数量
                'severity_correct': 0,                 # severity完全正确的数量
                'failed': 0,                           # 评测失败的数量
                'total_reasoning_percentage': 0.0,     # 累计推理百分制得分
                'total_official_percentage': 0.0,      # 累计official百分制得分
                'total_action_percentage': 0.0,        # 累计action百分制得分
                'total_severity_percentage': 0.0       # 累计severity百分制得分
            }

            # 创建线程锁
            lock = threading.Lock()

            print(f"\n开始评测...")
            print("=" * 70)

            # 创建进度条
            with tqdm(total=len(standard_data),
                    desc="评测进度",
                    unit="条",
                    ncols=120) as pbar:

                # 使用线程池处理
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # 提交任务
                    future_to_index = {}
                    for idx in range(len(standard_data)):
                        future = executor.submit(
                            evaluate_single_sample,
                            standard_data[idx],
                            model_outputs[idx],
                            idx,
                            pbar,
                            stats,
                            lock
                        )
                        future_to_index[future] = idx

                    # 收集结果
                    results = []
                    completed_count = 0

                    try:
                        for future in as_completed(future_to_index.keys()):
                            try:
                                result = future.result(timeout=REQUEST_TIMEOUT)
                                results.append(result)

                                # 更新进度条
                                with lock:
                                    update_progress_bar(pbar, stats)

                                # 每完成SAVE_INTERVAL条就保存一次
                                completed_count += 1
                                if completed_count % SAVE_INTERVAL == 0:
                                    logger.info(f"\n💾 已处理 {completed_count} 条，保存进度...")
                                    save_results(output_file, results, incremental=True)

                            except Exception as e:
                                logger.error(f"任务执行失败: {str(e)}")
                                with lock:
                                    stats['failed'] += 1
                                    update_progress_bar(pbar, stats)

                    finally:
                        # 确保最后保存所有结果
                        if results:
                            logger.info(f"\n💾 保存最终结果...")
                            save_results(output_file, results, incremental=False)

            # 计算最终统计
            total_samples = len(results)
            all_correct_rate = stats['all_correct'] / total_samples * 100 if total_samples > 0 else 0
            action_accuracy = stats['action_correct'] / total_samples * 100 if total_samples > 0 else 0
            official_accuracy = stats['official_correct'] / total_samples * 100 if total_samples > 0 else 0
            severity_accuracy = stats['severity_correct'] / total_samples * 100 if total_samples > 0 else 0

            # 计算各阶段平均百分制得分（0-100）
            avg_reasoning_percentage = stats['total_reasoning_percentage'] / total_samples if total_samples > 0 else 0
            avg_official_percentage = stats['total_official_percentage'] / total_samples if total_samples > 0 else 0
            avg_action_percentage = stats['total_action_percentage'] / total_samples if total_samples > 0 else 0
            avg_severity_percentage = stats['total_severity_percentage'] / total_samples if total_samples > 0 else 0

            # 计算加权平均总分（0-100）
            total_score_sum = sum(r['total_score'] for r in results)
            overall_avg_score = total_score_sum / total_samples if total_samples > 0 else 0

            # 打印统计结果
            print("\n" + "=" * 70)
            print("✅ 评测完成！")
            print(f"📊 统计结果:")
            print(f"   总样本数: {total_samples}")
            print(f"   三字段全对: {stats['all_correct']} ({all_correct_rate:.2f}%)")
            print(f"   有字段错误: {stats['has_wrong']}")
            print(f"   评测失败: {stats['failed']}")
            print(f"\n   各字段正确率:")
            print(f"   - Official正确: {stats['official_correct']} ({official_accuracy:.2f}%)")
            print(f"   - Action正确: {stats['action_correct']} ({action_accuracy:.2f}%)")
            print(f"   - Severity正确: {stats['severity_correct']} ({severity_accuracy:.2f}%)")
            print(f"\n   四阶段平均得分（百分制）:")
            print(f"   - Reasoning平均分: {avg_reasoning_percentage:.2f}/100")
            print(f"   - Official平均分: {avg_official_percentage:.2f}/100")
            print(f"   - Action平均分: {avg_action_percentage:.2f}/100")
            print(f"   - Severity平均分: {avg_severity_percentage:.2f}/100")
            print(f"\n   加权计算公式:")
            print(f"   总分 = Reasoning({avg_reasoning_percentage:.2f}) × 40% + "
                f"Official({avg_official_percentage:.2f}) × 30% + "
                f"Action({avg_action_percentage:.2f}) × 20% + "
                f"Severity({avg_severity_percentage:.2f}) × 10%")
            print(f"\n   ⭐ 加权平均总分: {overall_avg_score:.2f}/100")
            print(f"\n   结果文件: {output_file}")
            print("=" * 70)

            # 保存统计信息
            summary = {
                'total_samples': total_samples,
                'all_correct': stats['all_correct'],
                'has_wrong': stats['has_wrong'],
                'failed': stats['failed'],
                'all_correct_rate': all_correct_rate,
                'scoring_method': 'percentage_weighted',  # 评分方法：百分制+加权
                'weights': WEIGHTS,  # 权重配置
                'accuracies': {
                    'action_accuracy': action_accuracy,
                    'official_accuracy': official_accuracy,
                    'severity_accuracy': severity_accuracy
                },
                'counts': {
                    'action_correct': stats['action_correct'],
                    'official_correct': stats['official_correct'],
                    'severity_correct': stats['severity_correct']
                },
                'average_percentage_scores': {
                    'reasoning_avg': avg_reasoning_percentage,
                    'official_avg': avg_official_percentage,
                    'action_avg': avg_action_percentage,
                    'severity_avg': avg_severity_percentage
                },
                'overall_avg_score': overall_avg_score
            }

            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"📈 统计摘要已保存到: {summary_file}")

    except Exception as e:
        logger.error(f"评测过程中发生错误: {str(e)}", exc_info=True)
        print(f"❌ 评测失败: {str(e)}")

def main():
    """主函数"""
    global API_KEY, MODEL_NAME

    parser = argparse.ArgumentParser(
        description="DNS流量分析模型评测程序 (百分制+加权评分版本)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  # 基本用法
  python %(prog)s -s standard.json -m model_output.json -o results.json

  # 完整参数
  python %(prog)s --standard standard.json --model model_output.json --output results.json --workers 16

  # Tiny模式测试
  python %(prog)s -s standard.json -m model_output.json -o results.json --tiny 10

评分规则（百分制+加权）：
  1. 每个维度先计算百分制分数（0-100）：
     - Reasoning（推理过程）: LLM评估 evidence(0-20) + analysis(0-20) → 转换为百分制
     - Official（威胁类型）: 精确匹配 → 100%% 或 0%%
     - Action（响应动作）: 精确匹配 → 100%% 或 0%%
     - Severity（严重程度）: 完全匹配100%%，偏差1级20%%，偏差2级+0%%

  2. 然后按权重加权求和得到总分（0-100）：
     总分 = Reasoning × 40%% + Official × 30%% + Action × 20%% + Severity × 10%%

评分示例：
  假设某样本：
  - Reasoning: 34/40分 → 85%%
  - Official: 30/30分 → 100%%
  - Action: 0/20分 → 0%%
  - Severity: 2/10分 → 20%%

  总分 = 85×0.4 + 100×0.3 + 0×0.2 + 20×0.1 = 66.0/100

评分特点：
  - 四个维度完全独立评估
  - 所有样本都调用LLM评估推理过程
  - 更加准确和公正
        """
    )

    parser.add_argument('-s', '--standard',
                        required=True,
                        help='标准答案JSON文件路径')

    parser.add_argument('-m', '--model',
                        required=True,
                        help='被测模型输出JSON文件路径')

    parser.add_argument('-o', '--output',
                        default='evaluation_results.json',
                        help='评测结果输出文件路径 (默认: evaluation_results.json)')

    parser.add_argument('--workers',
                        type=int,
                        default=MAX_WORKERS,
                        help=f'最大并发数 (默认: {MAX_WORKERS})')

    parser.add_argument('--tiny',
                        type=int,
                        default=None,
                        help='Tiny模式：随机抽取指定数量的样本进行评测（例如：--tiny 10）')

    parser.add_argument('--api-key',
                        default=API_KEY,
                        help='DashScope API key (默认读取 DASHSCOPE_API_KEY)')

    parser.add_argument('--judge-model',
                        default=MODEL_NAME,
                        help='DashScope judge model name (默认: qwen3-max)')

    args = parser.parse_args()
    API_KEY = args.api_key
    MODEL_NAME = args.judge_model

    # 检查文件
    if not os.path.exists(args.standard):
        print(f"❌ 标准答案文件不存在: {args.standard}")
        return

    if not os.path.exists(args.model):
        print(f"❌ 模型输出文件不存在: {args.model}")
        return

    # 开始评测
    evaluate_model_output(
        args.standard,
        args.model,
        args.output,
        max_workers=args.workers,
        tiny=args.tiny
    )

if __name__ == "__main__":
    main()
