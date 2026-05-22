from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def default_metric_payload() -> dict[str, Any]:
    return {
        "action_accuracy": 0.0,
        "severity_accuracy": 0.0,
        "official_accuracy": 0.0,
        "exact_match": 0.0,
        "adapt_latency_avg_ms": 0.0,
        "adapt_latency_p95_ms": 0.0,
        "peak_vram_mb": 0.0,
        "effective_rank_mean": 0.0,
        "effective_rank_max": 0.0,
        "ood_gap": 0.0,
    }


def compute_generative_metrics(
    gt_actions: Sequence[str],
    gt_severities: Sequence[str],
    gt_officials: Sequence[str],
    pred_actions: Sequence[str],
    pred_severities: Sequence[str],
    pred_officials: Sequence[str],
) -> dict[str, float]:
    """Compute per-field accuracy and overall exact match for generative evaluation.

    All input sequences must have the same length.
    Empty strings in ground truth are skipped for that field's accuracy.
    Exact match requires all three fields to match simultaneously.
    """
    n = len(gt_actions)
    if not (n == len(gt_severities) == len(gt_officials) == len(pred_actions) == len(pred_severities) == len(pred_officials)):
        raise ValueError("All input sequences must have the same length")

    if n == 0:
        return {
            "action_accuracy": 0.0,
            "severity_accuracy": 0.0,
            "official_accuracy": 0.0,
            "exact_match": 0.0,
        }

    action_correct = 0
    action_total = 0
    severity_correct = 0
    severity_total = 0
    official_correct = 0
    official_total = 0
    exact_correct = 0

    for gt_a, gt_s, gt_o, pr_a, pr_s, pr_o in zip(
        gt_actions, gt_severities, gt_officials,
        pred_actions, pred_severities, pred_officials,
    ):
        if gt_a:
            action_total += 1
            if gt_a == pr_a:
                action_correct += 1
        if gt_s:
            severity_total += 1
            if gt_s == pr_s:
                severity_correct += 1
        if gt_o:
            official_total += 1
            if gt_o == pr_o:
                official_correct += 1
        # Exact match: all three fields agree (only counted when all GT fields are non-empty)
        if gt_a and gt_s and gt_o and gt_a == pr_a and gt_s == pr_s and gt_o == pr_o:
            exact_correct += 1

    exact_total = sum(1 for gt_a, gt_s, gt_o in zip(gt_actions, gt_severities, gt_officials) if gt_a and gt_s and gt_o)

    return {
        "action_accuracy": float(action_correct) / float(max(action_total, 1)),
        "severity_accuracy": float(severity_correct) / float(max(severity_total, 1)),
        "official_accuracy": float(official_correct) / float(max(official_total, 1)),
        "exact_match": float(exact_correct) / float(max(exact_total, 1)),
    }


def compute_binary_metrics(labels: Sequence[int], preds: Sequence[int]) -> dict[str, float]:
    if len(labels) != len(preds):
        raise ValueError("labels and preds must have the same length")
    total = len(labels)
    if total == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for label, pred in zip(labels, preds):
        if label == 1 and pred == 1:
            tp += 1
        elif label == 0 and pred == 0:
            tn += 1
        elif label == 0 and pred == 1:
            fp += 1
        elif label == 1 and pred == 0:
            fn += 1
        else:
            raise ValueError(f"label/pred must be binary 0/1, got label={label} pred={pred}")

    accuracy = float(tp + tn) / float(total)
    precision = float(tp) / float(max(tp + fp, 1))
    recall = float(tp) / float(max(tp + fn, 1))
    denom = precision + recall
    f1 = 0.0 if denom == 0.0 else 2.0 * precision * recall / denom
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_latency_stats(latencies_ms: Sequence[float]) -> dict[str, float]:
    if not latencies_ms:
        return {
            "adapt_latency_avg_ms": 0.0,
            "adapt_latency_p95_ms": 0.0,
        }

    values = sorted(float(v) for v in latencies_ms)
    avg = sum(values) / float(len(values))
    if len(values) == 1:
        p95 = values[0]
    else:
        p95_index = max(int(round(0.95 * (len(values) - 1))), 0)
        p95 = values[p95_index]
    return {
        "adapt_latency_avg_ms": avg,
        "adapt_latency_p95_ms": p95,
    }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def to_display_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, int):
        return str(value)
    if value is None:
        return ""
    return str(value)


def render_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    headers = list(columns)
    if not headers:
        return ""

    if not rows:
        return "No rows to display."

    widths: dict[str, int] = {col: len(col) for col in headers}
    for row in rows:
        for col in headers:
            text = to_display_value(row.get(col))
            widths[col] = max(widths[col], len(text))

    header_text = " | ".join(col.ljust(widths[col]) for col in headers)
    separator = "-+-".join("-" * widths[col] for col in headers)
    body = []
    for row in rows:
        values = []
        for col in headers:
            value = row.get(col)
            text = to_display_value(value)
            if _is_number(value):
                values.append(text.rjust(widths[col]))
            else:
                values.append(text.ljust(widths[col]))
        body.append(" | ".join(values))
    return "\n".join([header_text, separator, *body])

