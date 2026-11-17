"""默认 metric parser（Phase 4 收编）。

- parse_lmms_eval_metrics：逐字搬移自 autoresearch_common.SafeExecutor._parse_metrics
  （accuracy:/acc:/exact_match 正则 + lmms-eval markdown 表格行解析 + "| Value |" 格式）。
  作为 TaskProfile.metric_parser 为 None 时的默认实现（见 base.TaskProfile.get_metric_parser）。
- parse_basic_metrics：accuracy/acc/score 三个文本正则的共享实现，供
  autoresearch_explorer/history.py 复用（原与其余 parser 重复的字面循环）。

本模块自包含，不 import explorer / coder / common。
"""
import logging
import re
from typing import Dict


def parse_lmms_eval_metrics(output: str, logger: logging.Logger = None) -> Dict[str, float]:
    """从输出日志中解析关键指标（默认 lmms-eval 解析器）"""
    metrics = {}

    # 匹配常见的准确率/分数模式 (不区分大小写)
    patterns = [
        (r'accuracy[:\s]+([0-9]*\.?[0-9]+)', 'accuracy'),
        (r'acc[:\s]+([0-9]*\.?[0-9]+)', 'accuracy'),
        (r'score[:\s]+([0-9]*\.?[0-9]+)', 'score'),
        (r'f1[:\s]+([0-9]*\.?[0-9]+)', 'f1'),
        (r'precision[:\s]+([0-9]*\.?[0-9]+)', 'precision'),
        (r'recall[:\s]+([0-9]*\.?[0-9]+)', 'recall'),
    ]

    for pattern, key in patterns:
        matches = re.findall(pattern, output, re.IGNORECASE)
        if matches:
            try:
                # 取最后一个匹配 (通常是最总结果)
                value = float(matches[-1])
                # 处理百分比格式
                if value > 1.0:
                    value = value / 100.0
                metrics[key] = value
            except ValueError:
                continue

    # 【新增】解析 lmms-eval 表格格式
    # 格式示例: |scienceqa|none  |     0|exact_match|↑  |0.8158|±  | 0.006|     0.006|
    table_patterns = [
        # exact_match 指标
        (r'\|\s*[^|]+\|\s*[^|]+\|\s*[^|]+\|\s*exact_match\s*\|\s*↑?\s*\|\s*([0-9]*\.?[0-9]+)\s*\|', 'accuracy'),
        # accuracy 指标
        (r'\|\s*[^|]+\|\s*[^|]+\|\s*[^|]+\|\s*accuracy\s*\|\s*↑?\s*\|\s*([0-9]*\.?[0-9]+)\s*\|', 'accuracy'),
        # 通用的数值提取：在任何指标列后寻找数值
        (r'\|\s*\w+\s*\|[^|]*\|\s*\d+\s*\|\s*\w+\s*\|\s*[↑↓]?\s*\|\s*([0-9]+\.[0-9]+)\s*\|', 'accuracy'),
    ]

    for pattern, key in table_patterns:
        matches = re.findall(pattern, output, re.IGNORECASE)
        if matches:
            try:
                value = float(matches[-1])
                # 处理百分比格式
                if value > 1.0:
                    value = value / 100.0
                # 如果已存在 accuracy，取较大值
                if key in metrics:
                    metrics[key] = max(metrics[key], value)
                else:
                    metrics[key] = value
                if logger:
                    logger.info(f"📊 从表格解析到 {key}: {value:.4f}")
            except ValueError:
                continue

    # 【新增】解析 "Value | 0.8158" 格式的输出
    value_pattern = r'\|\s*Value\s*\|\s*([0-9]*\.?[0-9]+)\s*\|'
    value_matches = re.findall(value_pattern, output)
    if value_matches:
        try:
            value = float(value_matches[-1])
            if value > 1.0:
                value = value / 100.0
            if 'accuracy' in metrics:
                metrics['accuracy'] = max(metrics['accuracy'], value)
            else:
                metrics['accuracy'] = value
        except ValueError:
            pass

    return metrics


def parse_basic_metrics(output: str) -> Dict[str, float]:
    """accuracy/acc/score 三个文本正则的共享实现（history.py 复用）"""
    metrics = {}

    # 匹配常见的准确率/分数模式
    patterns = [
        (r'accuracy[:\s]+([0-9]*\.?[0-9]+)', 'accuracy'),
        (r'acc[:\s]+([0-9]*\.?[0-9]+)', 'accuracy'),
        (r'score[:\s]+([0-9]*\.?[0-9]+)', 'score'),
    ]
    for pattern, key in patterns:
        matches = re.findall(pattern, output, re.IGNORECASE)
        if matches:
            try:
                value = float(matches[-1])
                if value > 1.0:
                    value = value / 100.0
                metrics[key] = value
            except ValueError:
                continue

    return metrics
