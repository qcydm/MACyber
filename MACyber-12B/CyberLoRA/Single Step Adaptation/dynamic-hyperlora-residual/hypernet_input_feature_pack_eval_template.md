# HyperNet 输入特征增强实验记录模板

> 适用场景：比较以下版本时统一记录
> - `mean-only`
> - `mean + last + max -> sample_feature_dim`
> - `mean + last + max -> 192`（如果后续做）

---

## 1. 实验基本信息

- 日期：
- 实验名（`experiment.name`）：
- 配置文件：
- checkpoint：
- 数据设置：
  - real / synthetic：
  - `ood_source_jsons` 是否配置：
- 说明：

### 可选：直接先跑一遍汇总脚本
```bash
python scripts/summarize_feature_pack_eval.py \
  --eval-summary outputs/eval/<exp>/eval_summary.json \
  --train-metrics outputs/<exp>/train_metrics/epoch_metrics.jsonl
```

### 多版本横向对比（推荐）
```bash
python scripts/compare_feature_pack_experiments.py \
  --run mean_only outputs/eval/<exp_mean>/eval_summary.json outputs/<exp_mean>/train_metrics/epoch_metrics.jsonl \
  --run feature_pack_64 outputs/eval/<exp_new>/eval_summary.json outputs/<exp_new>/train_metrics/epoch_metrics.jsonl
```

---

## 2. 先看这 6 个核心指标

> 这 6 个最值得先看，能最快回答“这版到底值不值”。

| 版本 | exact_match | action_accuracy | severity_accuracy | parse_ok 占比 | val_loss | test_ood exact_match / ood_gap |
|---|---:|---:|---:|---:|---:|---:|
| mean-only |  |  |  |  |  |  |
| mean+last+max -> 现有 `sample_feature_dim` |  |  |  |  |  |  |
| mean+last+max -> 192（可选） |  |  |  |  |  |  |

### 这些数去哪找
- 评测总表：`outputs/eval/<exp>/eval_summary.json`
- CSV 总表：`outputs/eval/<exp>/eval_table.csv`
- 训练曲线：`outputs/<exp>/train_metrics/epoch_metrics.jsonl`
- 解析状态：`outputs/eval/<exp>/sample_traces/*.jsonl`

---

## 3. 建议顺序：先看结果，再看是不是假象

### A. 结果有没有真变好
从 `eval_summary.json` / `eval_table.csv` 抄：
- `exact_match`
- `action_accuracy`
- `severity_accuracy`
- `official_accuracy`

### B. 有没有被解析问题污染
从 `sample_traces/*.jsonl` 统计：
- `parse_ok`
- `parse_status`

建议记录：

| 版本 | ok | json_decode_failed_or_empty | json_fence_present_but_format_mismatch | bare_json_decode_failed_or_empty | missing_json_block |
|---|---:|---:|---:|---:|---:|
| mean-only |  |  |  |  |  |
| mean+last+max -> 现有 `sample_feature_dim` |  |  |  |  |  |
| mean+last+max -> 192（可选） |  |  |  |  |  |

> 如果 `exact_match` 变差，但 `parse_status` 里的格式失败变多，先别急着怪 feature packing。

---

## 4. 训练稳定性记录

从 `outputs/<exp>/train_metrics/epoch_metrics.jsonl` / `step_metrics.jsonl` 记录：

| 版本 | train_loss(最终) | val_loss(最终) | 曲线是否更稳 | 备注 |
|---|---:|---:|---|---|
| mean-only |  |  |  |  |
| mean+last+max -> 现有 `sample_feature_dim` |  |  |  |  |
| mean+last+max -> 192（可选） |  |  |  |  |

### 重点看什么
- 同样训练步数下，`val_loss` 有没有更低
- 曲线是不是更平稳
- 有没有明显抖动 / 停滞 / 非有限 loss

---

## 5. HyperNet 反应强不强

从 `eval_summary.json` 记录：
- `effective_rank_mean`
- `effective_rank_max`
- `adapt_latency_avg_ms`
- `adapt_latency_p95_ms`

| 版本 | effective_rank_mean | effective_rank_max | adapt_latency_avg_ms | adapt_latency_p95_ms | 备注 |
|---|---:|---:|---:|---:|---|
| mean-only |  |  |  |  |  |
| mean+last+max -> 现有 `sample_feature_dim` |  |  |  |  |  |
| mean+last+max -> 192（可选） |  |  |  |  |  |

### 怎么理解
- 指标涨了，`effective_rank` / alpha 也更有变化：说明 richer conditioning 大概率真在起作用
- 指标涨一点点，但 `effective_rank` 明显更活跃：说明可能开始碰到 64 维瓶颈

---

## 6. 是否值得继续做 192 维

### 先继续保留当前版本的信号
- `exact_match` 有明显提升
- `parse_ok` 没恶化
- `val_loss` 更稳
- `test_ood` 不掉

### 值得试 `64 * 3 = 192` 的信号
- `exact_match` 有提升，但幅度有限
- `effective_rank_mean / max` 看起来更有反应
- 你直觉上觉得信息已经进去了，但最后收益还没完全释放

---

## 7. 最后一句实验结论（每轮都写）

- 这轮结论：
- 我认为当前版本：保留 / 回退 / 继续迭代
- 是否建议下一步试 192 维：是 / 否
- 原因（最多 3 条）：
  1. 
  2. 
  3. 
