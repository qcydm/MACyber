from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _ordered_fieldnames(history: list[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    for row in history:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)
    return fields


def write_metrics_jsonl(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in history:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_metrics_csv(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _ordered_fieldnames(history)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def _plot_xy_lines(
    ax: Any,
    x_values: list[float],
    series: list[tuple[str, list[float]]],
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    for label, values in series:
        ax.plot(x_values, values, marker="o", label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()


def plot_metrics_png(
    path: Path,
    epoch_history: list[dict[str, Any]],
    step_history: list[dict[str, Any]] | None = None,
) -> bool:
    if not epoch_history:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    has_step = bool(step_history)
    if has_step:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    epochs = [float(row.get("epoch", idx + 1)) for idx, row in enumerate(epoch_history)]
    train_loss = [float(row.get("train_loss", 0.0)) for row in epoch_history]
    val_loss = [float(row.get("val_loss", 0.0)) for row in epoch_history]
    train_rank = [float(row.get("train_effective_rank", 0.0)) for row in epoch_history]
    val_rank = [float(row.get("val_effective_rank", 0.0)) for row in epoch_history]

    if has_step:
        axes_epoch_loss = axes[0][0]
        axes_epoch_rank = axes[0][1]
    else:
        axes_epoch_loss = axes[0]
        axes_epoch_rank = axes[1]

    _plot_xy_lines(
        ax=axes_epoch_loss,
        x_values=epochs,
        series=[("train_loss", train_loss), ("val_loss", val_loss)],
        title="Epoch Loss Curve",
        xlabel="Epoch",
        ylabel="Loss",
    )
    _plot_xy_lines(
        ax=axes_epoch_rank,
        x_values=epochs,
        series=[("train_effective_rank", train_rank), ("val_effective_rank", val_rank)],
        title="Epoch Effective Rank",
        xlabel="Epoch",
        ylabel="Rank",
    )

    if has_step:
        steps = [float(row.get("global_step", idx + 1)) for idx, row in enumerate(step_history or [])]
        step_total_loss = [float(row.get("step_total_loss", 0.0)) for row in (step_history or [])]
        step_task_loss = [float(row.get("step_task_loss", 0.0)) for row in (step_history or [])]
        step_lr = [float(row.get("lr", 0.0)) for row in (step_history or [])]
        step_rank = [float(row.get("step_effective_rank", 0.0)) for row in (step_history or [])]

        _plot_xy_lines(
            ax=axes[1][0],
            x_values=steps,
            series=[("step_total_loss", step_total_loss), ("step_task_loss", step_task_loss)],
            title="Step Loss Curve",
            xlabel="Global Step",
            ylabel="Loss",
        )
        _plot_xy_lines(
            ax=axes[1][1],
            x_values=steps,
            series=[("lr", step_lr), ("step_effective_rank", step_rank)],
            title="Step LR / Effective Rank",
            xlabel="Global Step",
            ylabel="Value",
        )

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def write_metrics_html(
    path: Path,
    epoch_history: list[dict[str, Any]],
    step_history: list[dict[str, Any]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    epoch_payload = json.dumps(epoch_history, ensure_ascii=False)
    step_payload = json.dumps(step_history or [], ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>DHR Training Metrics</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    h2 {{ margin-bottom: 8px; }}
    .chart {{ border: 1px solid #ddd; margin: 12px 0 24px; }}
    .meta {{ color: #666; font-size: 13px; }}
  </style>
</head>
<body>
  <h2>DHR Training Dashboard</h2>
  <p class="meta">Auto-generated from epoch metrics (open this file in browser).</p>
  <canvas id="epochLossChart" class="chart" width="980" height="320"></canvas>
  <canvas id="epochRankChart" class="chart" width="980" height="320"></canvas>
  <canvas id="stepLossChart" class="chart" width="980" height="320"></canvas>
  <canvas id="stepLrChart" class="chart" width="980" height="320"></canvas>
  <script>
    const epochData = {epoch_payload};
    const stepData = {step_payload};

    function drawLineChart(canvasId, data, xSelector, title, series) {{
      const canvas = document.getElementById(canvasId);
      const ctx = canvas.getContext("2d");
      const w = canvas.width;
      const h = canvas.height;
      const pad = 50;

      if (!data.length) {{
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = "#fff";
        ctx.fillRect(0, 0, w, h);
        ctx.strokeStyle = "#ddd";
        ctx.strokeRect(0.5, 0.5, w - 1, h - 1);
        ctx.fillStyle = "#666";
        ctx.font = "14px Arial";
        ctx.fillText(`${{title}} (no data)`, 16, 26);
        return;
      }}

      const xValues = data.map((d, i) => Number(xSelector(d, i)));
      const values = series.flatMap(s => data.map(d => Number(d[s.key] ?? 0)));
      const minY = Math.min(...values);
      const maxY = Math.max(...values);
      const ySpan = Math.max(maxY - minY, 1e-8);

      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#fff";
      ctx.fillRect(0, 0, w, h);
      ctx.strokeStyle = "#ddd";
      ctx.strokeRect(0.5, 0.5, w - 1, h - 1);
      ctx.fillStyle = "#111";
      ctx.font = "16px Arial";
      ctx.fillText(title, 16, 26);

      function xAt(i) {{
        if (xValues.length === 1) return pad;
        return pad + (i / (xValues.length - 1)) * (w - 2 * pad);
      }}
      function yAt(v) {{
        return h - pad - ((v - minY) / ySpan) * (h - 2 * pad);
      }}

      ctx.strokeStyle = "#efefef";
      ctx.beginPath();
      ctx.moveTo(pad, pad);
      ctx.lineTo(pad, h - pad);
      ctx.lineTo(w - pad, h - pad);
      ctx.stroke();

      ctx.fillStyle = "#666";
      ctx.font = "12px Arial";
      ctx.fillText(`min=${{minY.toFixed(4)}}`, pad + 4, h - pad + 16);
      ctx.fillText(`max=${{maxY.toFixed(4)}}`, pad + 4, pad - 8);
      ctx.fillText(`x=${{xValues[0]}}→${{xValues[xValues.length - 1]}}`, w - 170, h - pad + 16);

      series.forEach((s, sIdx) => {{
        ctx.strokeStyle = s.color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        data.forEach((d, i) => {{
          const v = Number(d[s.key] ?? 0);
          const x = xAt(i);
          const y = yAt(v);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }});
        ctx.stroke();

        ctx.fillStyle = s.color;
        ctx.fillRect(16 + sIdx * 220, 36, 12, 12);
        ctx.fillStyle = "#222";
        ctx.fillText(s.label, 34 + sIdx * 220, 46);
      }});
    }}

    drawLineChart("epochLossChart", epochData, (d, i) => d.epoch ?? (i + 1), "Epoch Loss Curve", [
      {{ key: "train_loss", label: "train_loss", color: "#2563eb" }},
      {{ key: "val_loss", label: "val_loss", color: "#dc2626" }},
    ]);
    drawLineChart("epochRankChart", epochData, (d, i) => d.epoch ?? (i + 1), "Epoch Effective Rank", [
      {{ key: "train_effective_rank", label: "train_effective_rank", color: "#0891b2" }},
      {{ key: "val_effective_rank", label: "val_effective_rank", color: "#7c3aed" }},
    ]);
    drawLineChart("stepLossChart", stepData, (d, i) => d.global_step ?? (i + 1), "Step Loss Curve", [
      {{ key: "step_total_loss", label: "step_total_loss", color: "#0ea5e9" }},
      {{ key: "step_task_loss", label: "step_task_loss", color: "#f97316" }},
    ]);
    drawLineChart("stepLrChart", stepData, (d, i) => d.global_step ?? (i + 1), "Step LR / Rank", [
      {{ key: "lr", label: "lr", color: "#22c55e" }},
      {{ key: "step_effective_rank", label: "step_effective_rank", color: "#a855f7" }},
    ]);
  </script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def save_training_artifacts(
    artifact_dir: Path,
    epoch_history: list[dict[str, Any]],
    step_history: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    epoch_jsonl_path = artifact_dir / "epoch_metrics.jsonl"
    epoch_csv_path = artifact_dir / "epoch_metrics.csv"
    step_jsonl_path = artifact_dir / "step_metrics.jsonl"
    step_csv_path = artifact_dir / "step_metrics.csv"
    plot_path = artifact_dir / "training_curves.png"
    html_path = artifact_dir / "training_curves.html"

    write_metrics_jsonl(epoch_jsonl_path, epoch_history)
    write_metrics_csv(epoch_csv_path, epoch_history)
    if step_history:
        write_metrics_jsonl(step_jsonl_path, step_history)
        write_metrics_csv(step_csv_path, step_history)
    write_metrics_html(
        path=html_path,
        epoch_history=epoch_history,
        step_history=step_history,
    )
    plot_ok = plot_metrics_png(
        path=plot_path,
        epoch_history=epoch_history,
        step_history=step_history,
    )

    artifacts = {
        "epoch_metrics_jsonl": str(epoch_jsonl_path.resolve()),
        "epoch_metrics_csv": str(epoch_csv_path.resolve()),
        "training_curves_html": str(html_path.resolve()),
    }
    if step_history:
        artifacts["step_metrics_jsonl"] = str(step_jsonl_path.resolve())
        artifacts["step_metrics_csv"] = str(step_csv_path.resolve())
    if plot_ok:
        artifacts["training_curves_png"] = str(plot_path.resolve())
    return artifacts
