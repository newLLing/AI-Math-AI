"""
Evolution Gate: 人类主动发起的演进决策存储（异步、非阻塞）。

设计原则（与阻塞式门控相反）：
- autoresearch 主循环**永不等待**人类；每轮 depth 设计产出 K 个演进草案后，
  按 default_policy 立即自动选择一个继续执行。
- 人类想决策时：停止 autoresearch -> 用 CLI (autoresearch_evolution_gate.py)
  查看候选 / 调整参数 (pin/ban/policy) -> 重新启动。决策在后续 depth 迭代生效。

存储文件: <state_dir>/evolution_decisions.json
schema:
{
  "pinned_moves": ["DUALIZE"],        // 下轮 depth 强制优先使用的演进算子（FIFO，用后消费）
  "banned_moves": ["GENERALIZE"],     // 禁用的演进算子（人类裁决的"证伪"）
  "default_policy": "first",          // 无人类决策时的自动选择: "first" | "random"
  "updated_at": "...",                // 最近修改时间
  "history": [{"ts": ..., "action": "pin", "moves": [...]}, ...]   // 人类决策审计
}
"""
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

DECISION_FILENAME = "evolution_decisions.json"

_DEFAULTS: Dict[str, Any] = {
    "pinned_moves": [],
    "banned_moves": [],
    "default_policy": "first",
    "updated_at": "",
    "history": [],
}


def decision_file_path(state_dir: str) -> str:
    return os.path.join(state_dir, DECISION_FILENAME)


def load_decisions(state_dir: str) -> Dict[str, Any]:
    """读取决策存储；不存在或损坏时返回默认值（不抛异常，保证主循环健壮）。"""
    path = decision_file_path(state_dir)
    data = dict(_DEFAULTS)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                data.update({k: v for k, v in loaded.items() if k in _DEFAULTS})
        except Exception:
            pass
    data["pinned_moves"] = [str(m).upper() for m in data.get("pinned_moves", [])]
    data["banned_moves"] = [str(m).upper() for m in data.get("banned_moves", [])]
    if data.get("default_policy") not in ("first", "random"):
        data["default_policy"] = "first"
    return data


def save_decisions(state_dir: str, data: Dict[str, Any], action: str = "",
                   moves: Optional[List[str]] = None) -> None:
    """原子写回决策存储，并追加审计记录。"""
    os.makedirs(state_dir, exist_ok=True)
    data["updated_at"] = datetime.now().isoformat()
    if action:
        history = data.setdefault("history", [])
        history.append({
            "ts": data["updated_at"],
            "action": action,
            "moves": moves or [],
        })
        data["history"] = history[-200:]  # 限制审计长度
    path = decision_file_path(state_dir)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def consume_pinned(state_dir: str, used_moves: List[str]) -> None:
    """从 pinned_moves 中移除已被本轮使用的算子（FIFO 消费）。"""
    if not used_moves:
        return
    data = load_decisions(state_dir)
    used = {m.upper() for m in used_moves}
    remaining = [m for m in data.get("pinned_moves", []) if m not in used]
    if remaining != data.get("pinned_moves", []):
        data["pinned_moves"] = remaining
        save_decisions(state_dir, data, action="consume_pinned", moves=sorted(used))
