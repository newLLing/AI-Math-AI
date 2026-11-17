"""
AutoResearch 主程序入口
协调 Explorer Agent 和 Coder Agent 完成自动研究迭代
"""

import os
import sys

# 【关键修复】强制离线模式，避免 Hugging Face Hub 429 限流导致迭代失败
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import re
import time
import signal
import argparse
import traceback
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional

# 导入共享模块
from autoresearch_common import (
    AutoResearchConfig,
    RobustLLMClient,
    StateManager,
    SafeExecutor,
    CodeExtractor,
    setup_logging,
    backup_original_code,
    FingerprintExecutor,
    derive_elite_archive_path,
    AsyncEvalManager,
    AsyncEvalTask,
)

# 导入 Agent 模块
from autoresearch_explorer import ExplorerAgent
from autoresearch_explorer.exploration_tree import ExplorationTree
from autoresearch_explorer.tree_guided_controller import TreeGuidedEvolutionController
from autoresearch_coder import CoderAgent


# ==========================================
# 主引擎类
# ==========================================

class AutoResearchEngine:
    """AutoResearch 主引擎 - 协调各 Agent 完成自动研究"""
    
    # 实验已知的两个关键 baseline（优先从配置读取，回退到当前任务 profile 的 baselines）
    @property
    def FAKE_SUCCESS_TOLERANCE(self) -> float:
        from autoresearch_tasks import current_task
        return self._get_baseline('fake_success_tolerance', current_task().baselines.fake_success_tolerance)

    def _get_baseline(self, attr_name: str, default_val: float) -> float:
        """从 config 读取 baseline，未配置则使用默认值"""
        return float(getattr(self.config, attr_name, default_val) or default_val)

    @property
    def ORIGINAL_BASELINE(self) -> float:
        from autoresearch_tasks import current_task
        return self._get_baseline('original_baseline', current_task().baselines.original_baseline)

    @property
    def MMTOK_BASELINE(self) -> float:
        from autoresearch_tasks import current_task
        return self._get_baseline('mmtok_baseline', current_task().baselines.mmtok_baseline)

    @property
    def UNIFORM_STEP_BASELINE(self) -> float:
        from autoresearch_tasks import current_task
        return self._get_baseline('uniform_step_baseline', current_task().baselines.uniform_step_baseline)

    def _apply_task_profile_overrides(self):
        """把 config 的运行级覆盖写入 task profile（单进程单任务，安全）。

        必须在创建任何 agent 之前调用（IdeaHistoryTracker 等在初始化时
        固化 profile 基线，见 history.py:28-34）。

        - mmtok_baseline 为唯一基线口径：empirical_accuracy_baseline /
          history_baseline_fallback / mmtok_baseline / breakthrough_protect_baseline
          全部对齐，修复 UCT 价值归一化（objective_feedback 默认 0.82/0.08）
          与方向潜力分（history 默认 0.7038）的口径分裂。
        - empirical_accuracy_scale：UCT 归一化 scale。
        - combo_min_trials：Phase A 每 (combo, archetype) 对的固定探索轮数
          （20-30，combo_allocator.min_trials() 显式值优先）。
        """
        from autoresearch_tasks import current_task
        profile = current_task()
        mmtok = getattr(self.config, 'mmtok_baseline', None)
        if mmtok:
            b = profile.baselines
            b.mmtok_baseline = float(mmtok)
            b.empirical_accuracy_baseline = float(mmtok)
            b.history_baseline_fallback = float(mmtok)
            b.breakthrough_protect_baseline = float(mmtok)
            self.logger.info(
                f"📏 [Engine] 基线口径统一: profile baselines 已对齐 "
                f"mmtok_baseline={float(mmtok):.4f}"
                f"（含 UCT 归一化 / 方向潜力分 / history 统计）"
            )
        scale = getattr(self.config, 'empirical_accuracy_scale', None)
        if scale:
            profile.baselines.empirical_accuracy_scale = float(scale)
        # fusion_threshold：cross 的 Fusion Round 门槛（cross.py 读取 profile baselines）。
        # profile 默认 0.86 是 LLaVA 0.8x 时代残留——OCRBench 平台期 ~0.49 时
        # qualified 父本恒 < 2，cross 融合永不触发；允许 config 按任务覆盖
        ft = getattr(self.config, 'fusion_threshold', None)
        if ft:
            profile.baselines.fusion_threshold = float(ft)
            self.logger.info(f"📏 [Engine] fusion_threshold 覆盖为 {float(ft):.4f}")
        mt = int(getattr(self.config, 'combo_min_trials', 0) or 0)
        if mt > 0:
            profile.combo_objective_min_trials = mt
            self.logger.info(
                f"📏 [Engine] Phase A 每 (combo, archetype) 对固定探索 {mt} 轮"
            )

    def __init__(self, config: AutoResearchConfig, debug_idea_path: str = None, start_from: int = None):
        self.config = config
        # Phase 4：任务 profile 接线——必须在创建任何 agent / 调用任何 current_task() 之前完成。
        # 空 task_profile 时按 model_family 映射到默认 profile（向后兼容现有配置）。
        from autoresearch_tasks import set_current_task, resolve_task_name
        self.task_profile = set_current_task(
            resolve_task_name(getattr(config, 'task_profile', ''), getattr(config, 'model_family', 'qwen')))
        self.logger = setup_logging(config.log_dir, 0)
        self.logger.info(f"🧩 [Engine] 当前任务 profile: {self.task_profile.name}")
        # 统一基线口径 + Phase A 显式 min_trials（必须在创建 agents 之前：
        # history_tracker 等在初始化时固化 profile 基线）
        self._apply_task_profile_overrides()
        self.state_manager = StateManager(config.state_file, self.logger, config=config)
        self.llm_client = RobustLLMClient(config, self.logger)
        self.executor = SafeExecutor(config, self.logger)
        self._interrupted = False

        # [AsyncEval] 异步评测流水线（生效条件：async_eval_mode 且非 smoke 且非并行分支；
        # 不满足时 manager 为 None，所有路径逐字同原行为）
        if self._async_eval_enabled():
            self.async_eval_manager = AsyncEvalManager(config, self.logger)
            self.logger.info(
                f"📤 [Engine] 异步评测模式已启用（max_async_evals="
                f"{getattr(config, 'max_async_evals', 2)}）"
            )
        else:
            self.async_eval_manager = None

        # [AsyncEval] look-ahead 缓存：槽位等待期间预生产的下一轮
        # {"iteration", "strategy", "idea", "code", "topic"}；消费或作废后置 None。
        self._lookahead = None
        # 每次槽位等待只尝试预生产一次（记录目标轮号）：预生产失败/被禁用
        # 时避免在等待循环里反复触发 LLM 调用形成热循环
        self._lookahead_attempted_for = None

        os.makedirs(config.log_dir, exist_ok=True)
        
        # v2.1: 根据配置动态设置深度探索门槛（支持按模型家族差异化）
        deepen_threshold = getattr(self.config, 'deepen_threshold_base', None)
        if deepen_threshold is not None:
            # 注：历史上此处还会写 ResearchThresholds.DEEPEN_THRESHOLD_BASE 与
            # tgc.DEFAULT_DEPTH_THRESHOLD_RAW 两个全局变量，但二者全库零读者
            # （阈值实际由 TreeGuidedEvolutionController._compute_adaptive_threshold
            # 从 config 直接读取），属死写，已删除；仅保留此 info 日志。
            self.logger.info(
                f"📊 [Engine] 深度探索门槛已设置为 {float(deepen_threshold):.4f} "
                f"(model_family={getattr(self.config, 'model_family', 'qwen')})"
            )
        
        # 启动时先清理历史状态中的虚假成功记录（必须在重建历史之前）
        self._sanitize_state_baselines()
        
        # 断点续跑：若指定了 start_from，先截断 state 到该轮次之前
        if start_from is not None and start_from > 1:
            self._truncate_state_for_restart(start_from)
        
        # 初始化 Agents
        self.explorer = ExplorerAgent(config, self.llm_client, self.logger)
        self.coder = CoderAgent(config, self.llm_client, self.logger, should_stop=lambda: self._interrupted)
        
        # 从历史状态重建 Explorer 的历史记录（断点续跑时至关重要）
        self._reconstruct_explorer_history(max_iteration=start_from)
        
        # FIX: 如果 deepen_templates 仍为空，从 experiment_results.md 构建的种子文件加载
        # 解决 Qwen OCRBench 等低分 benchmark 上 deepen_threshold 过高导致模板池为空的问题
        if not getattr(self.explorer.history_tracker, 'deepen_templates', None):
            seed_path = os.path.join(config.log_dir, "deepen_templates_seed.json")
            if os.path.exists(seed_path):
                try:
                    with open(seed_path, 'r', encoding='utf-8') as f:
                        seed_data = json.load(f)
                    templates = seed_data.get("templates", [])
                    if templates:
                        self.explorer.history_tracker.deepen_templates = templates
                        self.logger.info(
                            f"📚 [Engine] 从 {seed_path} 加载 {len(templates)} 个种子模板 "
                            f"作为 deepen_templates (最高 acc={max(t['accuracy'] for t in templates):.4f})"
                        )
                except Exception as e:
                    self.logger.warning(f"⚠️ [Engine] 加载 deepen_templates_seed 失败: {e}")
        
        # 保存上一轮错误日志，用于 CoderAgent 基于日志修复
        self.last_error_log = None
        
        # 调试模式：直接使用现成的 idea 文件，跳过搜索
        self.debug_idea_path = debug_idea_path
        
        # 初始化指纹预跑器（F包）
        if getattr(config, 'enable_fingerprint', True):
            self.fingerprint_executor = FingerprintExecutor(self.logger)
        else:
            self.fingerprint_executor = None

        # v12.1: 运行时 token 数探针（防动态超选 hack，如 iter_461 CAWR 事件）
        if getattr(config, 'enable_token_count_probe', True):
            from autoresearch_common import TokenCountProbe
            self.token_count_probe = TokenCountProbe(
                self.logger,
                retain_ratio=float(getattr(config, 'probe_retain_ratio', 0.05)),
                timeout=int(getattr(config, 'probe_timeout_seconds', 180)),
            )
        else:
            self.token_count_probe = None
        
        # === v2: 初始化探索树和树引导控制器 ===
        tree_file = os.path.join(config.log_dir, "exploration_tree.json")
        self.exploration_tree = ExplorationTree(tree_file, self.logger)
        # 尝试加载已有树
        loaded_controller_state = self.exploration_tree.load()
        if loaded_controller_state is not None:
            self.logger.info(f"🌲 [Engine] 探索树已加载: {len(self.exploration_tree.nodes)} 节点")
            # FIX: 如果 tree 落后于 state，触发回填（防止中断后 tree 缺失新迭代）
            state_iterations = len(self.state_manager.state.get("completed_iterations", []))
            tree_iteration = self.exploration_tree.current_iteration
            if tree_iteration < state_iterations:
                self.logger.warning(
                    f"🌲 [Engine] Tree ({tree_iteration}) lags behind state ({state_iterations}), "
                    f"triggering backfill"
                )
                self._backfill_exploration_tree()
                # 重新加载 controller state（backfill 后可能更新了）
                loaded_controller_state = self.tree_controller.get_state() if hasattr(self, 'tree_controller') else None
        else:
            # 尝试从 state 回填
            self._backfill_exploration_tree()
        # v4.0: Controller state health check
        if loaded_controller_state:
            total = loaded_controller_state.get("breadth_count", 0) + loaded_controller_state.get("depth_count", 0) + loaded_controller_state.get("cross_count", 0)
            if total > 20 and loaded_controller_state.get("breadth_count", 0) / total < 0.1:
                self.logger.warning(
                    f"🚨 [Engine] Controller state unhealthy: breadth={loaded_controller_state.get('breadth_count')}, "
                    f"depth={loaded_controller_state.get('depth_count')}, cross={loaded_controller_state.get('cross_count')}. "
                    f"Resetting counters to restore exploration balance."
                )
                loaded_controller_state = None

        model_family = getattr(self.config, 'model_family', 'qwen')
        self.tree_controller = TreeGuidedEvolutionController(
            self.exploration_tree, self.logger,
            config={
                "original_baseline": getattr(self.config, 'original_baseline', None),
                "mmtok_baseline": getattr(self.config, 'mmtok_baseline', None),
                "deepen_threshold_base": getattr(self.config, 'deepen_threshold_base', None),
                "force_mode": getattr(self.config, 'force_mode', None),
                "force_breadth_only": getattr(self.config, 'force_breadth_only', True),
                "model_family": model_family,
                # v12.0: learning 模式的 depth 预算
                "depth_budget": getattr(self.config, 'depth_budget', 0.2),
                # 消融臂：策略选择去分数化（均匀随机 leaf/mode，绕过 UCT）
                "random_strategy_selection": getattr(self.config, 'random_strategy_selection', False),
            },
            controller_state=loaded_controller_state,
        )
        # v6.0 FIX: 注入 history_tracker 到 Controller，启用 lineage awareness 和 depth cooling
        self.tree_controller.history_tracker = self.explorer.history_tracker
        # 将 tree、controller 注入 ExplorerAgent
        self.explorer.exploration_tree = self.exploration_tree
        self.explorer.tree_controller = self.tree_controller

        # Phase I1: 张量组合自适应探索（默认关闭）。开启时创建两阶段分配器挂到
        # 树控制器；树状态恢复后需显式 load_state 恢复分配统计。
        self.combo_allocator = None
        if getattr(config, 'enable_tensor_combo_exploration', False):
            from autoresearch_explorer.combo_allocator import ComboAllocator
            self.combo_allocator = ComboAllocator(total_budget=config.max_iterations)
            if loaded_controller_state and loaded_controller_state.get("combo_allocator"):
                self.combo_allocator.load_state(loaded_controller_state["combo_allocator"])
            self.tree_controller.combo_allocator = self.combo_allocator
            _mt = self.combo_allocator.min_trials()
            _n_pairs = len(self.combo_allocator._pairs())
            self.logger.info(
                f"🧩 [Engine I1] 张量组合两阶段探索已启用"
                f"（每对至少 {_mt} 轮探索，共 {_n_pairs} 对）")
            self.logger.info(
                f"🧩 [Engine I1] Phase A 预算估算: {_n_pairs} 对 × {_mt} 轮 = "
                f"{_n_pairs * _mt} trials（并行分支会加速消耗），"
                f"总预算 {self.config.max_iterations} 轮")

        # === v12.0: Research Notebook（depth-as-learning）===
        # depth 模式重建为学习闭环的知识库；breadth 保持纯搜索不被收窄。
        self.notebook = None
        if getattr(self.config, 'enable_learning_depth', False):
            try:
                from autoresearch_explorer.research_notebook import ResearchNotebook
                orch = getattr(self.explorer, '_orchestrator', None)
                nb_store = getattr(orch, 'state_store', None) if orch else None
                if nb_store is not None:
                    self.notebook = ResearchNotebook(
                        nb_store, self.llm_client, self.logger, self.config)
                    # 经 ExplorerAgent.__setattr__ 代理注入 orchestrator 与各 subagent
                    self.explorer.notebook = self.notebook
                    self.tree_controller.notebook = self.notebook
                    # history_tracker 直接引用（deepen_templates 入池闸门用）
                    self.explorer.history_tracker.notebook = self.notebook
                    self.logger.info("📓 [Engine v12.0] Research Notebook 已接线（depth-as-learning）")
                else:
                    self.logger.warning("⚠️ [Engine v12.0] orchestrator.state_store 不可用，notebook 未接线")
            except Exception as nb_init_err:
                self.logger.warning(f"⚠️ [Engine v12.0] Notebook 初始化失败（不影响主流程）: {nb_init_err}")
                self.notebook = None
        
        # v8.1: Seed exploration tree from elite archive + set elite_domains for cross-domain prioritization
        # v9.0 FIX: 当 disable_high_score_template_injection=True 时跳过，避免高分模板冷启动污染
        if not getattr(self.config, 'disable_high_score_template_injection', False):
            try:
                elite_path = derive_elite_archive_path(
                    self.config, os.path.join(os.path.dirname(__file__), "autoresearch_explorer"))
                if os.path.exists(elite_path):
                    with open(elite_path, 'r', encoding='utf-8') as f:
                        elite_data = json.load(f)
                    self.exploration_tree.seed_from_elite(elite_data.get('algorithms', []))
                    from autoresearch_explorer.math_domain_registry import DIRECTION_TO_CELLS
                    elite_domains = set()
                    for alg in elite_data.get('algorithms', []):
                        cells = DIRECTION_TO_CELLS.get(alg.get('direction', ''), [])
                        for cell in cells:
                            elite_domains.add(cell[0])
                    self.tree_controller._elite_domains = elite_domains
                    self.logger.info(f"🌲 [Engine v8.1] Elite domains seeded: {sorted(elite_domains)}")
            except Exception as e:
                self.logger.warning(f"⚠️ [Engine v8.1] Elite tree seeding failed: {e}")
        
        # v7.5: 从 elite_distilled.json 加载 top_performers 为 deepen_templates
        # 解决 LLaVA 配置 log_dir 指向空目录导致的彻底冷启动问题
        # v9.0 FIX: 当 disable_high_score_template_injection=True 时跳过，保持冷启动纯净
        if not getattr(self.config, 'disable_high_score_template_injection', False):
            distilled_path = getattr(self.config, 'distilled_history_path', None)
            if distilled_path and os.path.exists(distilled_path):
                try:
                    with open(distilled_path, 'r', encoding='utf-8') as f:
                        distilled = json.load(f)
                    # 加载 top_performers 为 deepen_templates（v7.5 核心修复）
                    top_performers = distilled.get('top_performers', [])
                    if top_performers:
                        templates = []
                        for perf in top_performers:
                            templates.append({
                                "iteration": perf.get("iteration", 0),
                                "accuracy": perf.get("accuracy", 0.0),
                                "direction": perf.get("direction", "unknown"),
                                "operators": perf.get("operators", []),
                            })
                        self.explorer.history_tracker.deepen_templates = templates
                        self.logger.info(
                            f"📚 [Engine] 从 elite_distilled 加载 {len(templates)} 个 top_performers "
                            f"作为 deepen_templates (最高 acc={max(t['accuracy'] for t in templates):.4f})"
                        )
                    # 同时加载方向潜力分作为参考
                    direction_stats = distilled.get('direction_stats', {})
                    for direction, stats in direction_stats.items():
                        pot = stats.get('potential', 0.5)
                        self.explorer.history_tracker.direction_potential[direction] = pot
                    if direction_stats:
                        self.logger.info(
                            f"📚 [Engine] 加载方向潜力分参考: "
                            f"{len(direction_stats)} 个方向"
                        )
                except Exception as e:
                    self.logger.warning(f"⚠️ 加载 elite_distilled 失败: {e}")
        
        # v3: 加载外部基准最佳代码（已禁用 —— 不同 benchmark 间不共享基准）
        model_family = getattr(self.config, 'model_family', 'qwen')
        if model_family != 'qwen':
            self.logger.info("📚 [Engine] 外部基准代码加载仅支持 Qwen，LLaVA 跳过")
        else:
            self.logger.info("📚 [Engine] 外部基准代码加载已禁用（不同 benchmark 结果不可迁移）")
        
        self.logger.info("=" * 60)
        self.logger.info("🤖 AutoResearch Engine 启动 (v2: UCT Tree-Driven)")
        self.logger.info(f"📁 日志目录: {os.path.abspath(config.log_dir)}")
        self.logger.info(f"🔄 最大迭代次数: {config.max_iterations}")
        self.logger.info(f"🤖 Explorer LLM: Qwen ({config.qwen_model_name})")
        if getattr(config, 'coder_use_kimi', False):
            self.logger.info(f"💻 Coder LLM: Kimi ({config.kimi_model_name})")
        else:
            coder_model = getattr(config, 'coder_qwen_model_name', None) or config.qwen_model_name
            self.logger.info(f"💻 Coder LLM: Qwen ({coder_model})")
        self.logger.info(f"💡 任务: 数学理论召回 + UCT 树驱动探索 + 代码生成")
        self.logger.info("🔍 搜索来源 (数学理论优先):")
        self.logger.info("   - OpenAlex 学术论文")
        self.logger.info("   - OpenReview 理论 track (ICML/NeurIPS/ICLR)")
        self.logger.info("🌲 探索树已启用 (UCT-driven, 8 domain + 时间感知)")
        # 配置合理性检查
        if self.MMTOK_BASELINE >= self.ORIGINAL_BASELINE:
            self.logger.warning(
                f"🚨 [Engine] Baseline 配置异常: MMTOK_BASELINE ({self.MMTOK_BASELINE:.4f}) "
                f">= ORIGINAL_BASELINE ({self.ORIGINAL_BASELINE:.4f})，fake success 检测可能失效"
            )
        deepen_threshold = getattr(self.config, 'deepen_threshold_base', None)
        if deepen_threshold is not None and deepen_threshold <= self.MMTOK_BASELINE:
            self.logger.warning(
                f"🚨 [Engine] deepen_threshold_base ({deepen_threshold:.4f}) <= MMTOK_BASELINE "
                f"({self.MMTOK_BASELINE:.4f})，深度探索门槛形同虚设，精英档案质量将下降"
            )
        force_mode = getattr(getattr(self, 'tree_controller', None), 'force_mode', None) if hasattr(self, 'tree_controller') else None
        if force_mode == "depth" and getattr(self.config, 'disable_high_score_template_injection', False):
            self.logger.warning(
                "🚨 [Engine] force_mode='depth' 与 disable_high_score_template_injection=True 互斥，"
                "depth 模式将无法执行，将强制降级为 breadth"
            )
        
        self.logger.info("=" * 60)
        # 多 GPU 分支并行评测：为每个分支准备 PYTHONPATH 影子包环境
        self._setup_branch_envs()

    def _resolve_mmtok_pkg_dir(self) -> str:
        """定位 mmtok 包目录：优先 import mmtok；失败时从 target_patch_file 向上查找。
        （从 _setup_branch_envs 原样抽取，供分支环境与 [AsyncEval] 快照环境共用。）
        """
        try:
            import mmtok  # 延迟导入防循环
            return os.path.dirname(os.path.abspath(mmtok.__file__))
        except ImportError:
            # 回退：mmtok 未安装到当前解释器时，从 target_patch_file 向上查找 mmtok 包目录
            mmtok_pkg_dir = None
            d = os.path.dirname(os.path.abspath(self.config.target_patch_file))
            while True:
                if os.path.basename(d) == "mmtok" and os.path.isfile(os.path.join(d, "__init__.py")):
                    mmtok_pkg_dir = d
                    break
                parent = os.path.dirname(d)
                if parent == d:
                    break
                d = parent
            if mmtok_pkg_dir is None:
                raise RuntimeError(
                    "无法定位 mmtok 包目录（import mmtok 失败且 "
                    "target_patch_file 不在 mmtok 包内）")
            return mmtok_pkg_dir

    def _setup_branch_envs(self):
        """parallel_branches > 0 时，为每个分支拷贝一份 mmtok 代码包到
        <log_dir>/branch_envs/branch_{i}/ 下，评测时 PYTHONPATH 前置该目录即可
        让评测子进程加载分支专属的 autoresearch_selector.py，实现分支间文件级隔离。
        总是刷新拷贝，保证分支代码与当前 mmtok 包一致。
        """
        self._branch_envs: List[str] = []
        self._branch_patch_files: List[str] = []
        n = int(getattr(self.config, 'parallel_branches', 0) or 0)
        if n <= 0:
            return
        try:
            import shutil
            mmtok_pkg_dir = self._resolve_mmtok_pkg_dir()
            rel = os.path.relpath(
                os.path.abspath(self.config.target_patch_file),
                os.path.dirname(mmtok_pkg_dir))
            for i in range(n):
                branch_env = os.path.join(self.config.log_dir, "branch_envs", f"branch_{i}")
                os.makedirs(branch_env, exist_ok=True)
                shutil.copytree(
                    mmtok_pkg_dir,
                    os.path.join(branch_env, "mmtok"),
                    ignore=shutil.ignore_patterns(
                        '__pycache__', 'lmms_eval_results', '*.pyc'),
                    dirs_exist_ok=True)
                self._branch_envs.append(branch_env)
                self._branch_patch_files.append(os.path.join(branch_env, rel))
            self.logger.info(
                f"🌿 [Engine] 分支并行环境已就绪: {n} 个分支 "
                f"({os.path.join(self.config.log_dir, 'branch_envs')})，"
                f"分支目标文件相对路径: {rel}")
        except Exception as e:
            self.logger.warning(f"⚠️ [Engine] 分支并行环境准备失败，禁用分支并行: {e}")
            self._branch_envs = []
            self._branch_patch_files = []

    def _get_expected_retain_ratio(self) -> str:
        """从 eval_cmd 推导期望的 retain_ratio；env 键名与默认值均从任务 profile 读取"""
        from autoresearch_tasks import current_task
        profile = current_task()
        eval_cmd = getattr(self.config, 'eval_cmd', '')
        for ratio_key in profile.retain_ratio_env_keys:
            if ratio_key in eval_cmd:
                try:
                    m = re.search(rf'{re.escape(ratio_key)}(\d+\.\d+|\d+)', eval_cmd)
                    if m:
                        return m.group(1)
                except Exception:
                    pass
        return str(profile.fixed_constraints.get('expected_retain_ratio', '0.1'))
    
    def _truncate_state_for_restart(self, target_iter: int):
        """将 state 截断到指定轮次之前，确保断点续跑时状态干净"""
        original_completed = self.state_manager.state.get("completed_iterations", [])
        removed = len([c for c in original_completed if c.get("iteration", 0) >= target_iter])
        self.state_manager.state["last_iteration"] = target_iter - 1
        self.state_manager.state["completed_iterations"] = [
            c for c in original_completed
            if c.get("iteration", 0) < target_iter
        ]
        # 清除来自被截断轮次的 best_result
        best = self.state_manager.state.get("best_result")
        if best and best.get("iteration", 0) >= target_iter:
            self.state_manager.state["best_result"] = None
        # 清除旧反馈，防止污染新探索
        self.state_manager.state["history_feedback"] = ""
        self.state_manager.save()
        self.logger.info(f"🔄 [Engine] 状态已预截断到 iteration {target_iter - 1} (移除 {removed} 条历史记录)")
        
        # v6.1 FIX: 同步截断 exploration_tree.json，避免未来节点误导 UCB 决策
        tree_file = os.path.join(self.config.log_dir, "exploration_tree.json")
        if os.path.exists(tree_file):
            try:
                with open(tree_file, 'r', encoding='utf-8') as f:
                    tree_data = json.load(f)
                nodes = tree_data.get("nodes", {})
                pruned_nodes = {
                    k: v for k, v in nodes.items()
                    if v.get("discovery_iteration", 0) < target_iter
                }
                removed_nodes = len(nodes) - len(pruned_nodes)
                tree_data["nodes"] = pruned_nodes
                tree_data["current_iteration"] = target_iter - 1
                # 截断 controller state
                cs = tree_data.get("controller_state", {})
                if cs:
                    cs["decision_log"] = [
                        d for d in cs.get("decision_log", [])
                        if d.get("iteration", 0) < target_iter
                    ]
                    b = d = x = 0
                    for log in cs["decision_log"]:
                        m = log.get("strategy", {}).get("mode", "breadth")
                        if m == "depth": d += 1
                        elif m == "cross": x += 1
                        else: b += 1
                    cs["breadth_count"] = b
                    cs["depth_count"] = d
                    cs["cross_count"] = x
                with open(tree_file, 'w', encoding='utf-8') as f:
                    json.dump(tree_data, f, indent=2, ensure_ascii=False)
                self.logger.info(
                    f"🌲 [Engine] 探索树已截断到 iteration {target_iter - 1} "
                    f"(移除 {removed_nodes} 个未来节点)"
                )
            except Exception as e:
                self.logger.warning(f"⚠️ [Engine] 探索树截断失败: {e}")
    
    def _cleanup_incomplete_iteration_dirs(self, start_iter: int):
        """v7.6 FIX: 启动时检测并清理未完成的迭代目录
        
        若存在 iter_dir 但 state 中没有该迭代的完成记录，说明上一轮在中途中断，
        将其重命名为 .interrupted. 后缀，避免日志追加污染，同时保留旧日志供查看。
        """
        import shutil
        completed_iters = {
            c.get("iteration", 0) for c in self.state_manager.state.get("completed_iterations", [])
        }
        cleaned = 0
        for iter_num in range(start_iter, self.config.max_iterations + 1):
            iter_dir = os.path.join(self.config.log_dir, f"iter_{iter_num:03d}")
            if not os.path.exists(iter_dir):
                continue
            if iter_num in completed_iters:
                continue  # 已完成的迭代，保留
            # 未完成的迭代目录：重命名
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup_dir = f"{iter_dir}.interrupted.{timestamp}"
            try:
                shutil.move(iter_dir, backup_dir)
                self.logger.info(
                    f"🧹 [Engine] 检测到未完成迭代目录 {iter_dir}，已重命名为 {backup_dir}"
                )
                cleaned += 1
            except Exception as e:
                self.logger.warning(f"⚠️ [Engine] 清理未完成目录失败: {e}")
        if cleaned:
            self.logger.info(f"🧹 [Engine] 共清理 {cleaned} 个未完成迭代目录")
        # 清理完成后，移除 interrupted_at_iteration 标记并持久化
        if "interrupted_at_iteration" in self.state_manager.state:
            del self.state_manager.state["interrupted_at_iteration"]
            self.state_manager.save()
    
    def _reconstruct_explorer_history(self, max_iteration: Optional[int] = None):
        """从 StateManager 和日志目录重建 Explorer 的历史记录，确保断点续跑时决策一致"""
        completed = self.state_manager.state.get("completed_iterations", [])
        
        # v7.0 FIX: Fallback to UnifiedStateStore if state.json is empty/corrupted
        if not completed:
            orch = getattr(self.explorer, '_orchestrator', None)
            if orch and hasattr(orch, 'state_store'):
                try:
                    db_records = orch.state_store.get_experiments(limit=10000)
                    if db_records:
                        completed = [self._experiment_record_to_state_entry(r) for r in db_records]
                        self.logger.info(
                            f"📜 [Engine] 状态文件为空，从 SQLite 回退加载 {len(completed)} 条记录"
                        )
                except Exception as e:
                    self.logger.warning(f"⚠️ [Engine] SQLite 回退加载失败: {e}")
        
        if completed:
            self.explorer.history_tracker.load_from_state_entries(completed, log_dir=self.config.log_dir)
            self.explorer.history_tracker.reconstruct_from_log_dir(self.config.log_dir, max_iteration=max_iteration)
            ht = self.explorer.history_tracker
            top_potentials = sorted(ht.direction_potential.items(), key=lambda x: x[1], reverse=True)[:5]
            potential_str = ", ".join([f"{d}={p:.2f}" for d, p in top_potentials])
            self.logger.info(
                f"📜 [Engine] 已重建历史: {len(ht.idea_history)} 条记录, "
                f"已探索方向={ht.explored_directions}, "
                f"Top5潜力分=[{potential_str}]"
            )
        
        # 【修复】回填 DomainRegistry 覆盖率，避免进程重启后归零
        if completed and hasattr(self.explorer.history_tracker, 'domain_registry'):
            # 优先从持久化文件恢复（最准确）
            self.explorer.history_tracker.load_domain_registry()
            # 再用当前 history 做补充（处理未持久化的新记录）
            for entry in self.explorer.history_tracker.idea_history:
                cells = set(tuple(c) for c in entry.get("structural_cells", []))
                if not cells and entry.get("direction"):
                    # 如果 state 中未保存 cells，从 direction 推断
                    from autoresearch_explorer.math_domain_registry import DIRECTION_TO_CELLS
                    cells = set(DIRECTION_TO_CELLS.get(entry["direction"], []))
                if cells:
                    acc = float(entry.get("metrics", {}).get("accuracy") or 0.0)
                    self.explorer.history_tracker.domain_registry.record_attempt(
                        cells,
                        success=entry.get("success", False) and not entry.get("is_fake_success", False) and not entry.get("is_tampered_ratio", False),
                        accuracy=acc
                    )
        
        # 同时初始化 topic generator 的 tracker
        # FIX: 同时恢复 _generated_topics 和 topic_pool_manager.used_topics，
        # 防止断点续跑后重复生成相同 topic 并错误触发扩展
        restored_topics = 0
        for entry in completed:
            topic = entry.get("topic")
            if topic:
                if hasattr(self.explorer, 'topic_generator'):
                    self.explorer.topic_generator.record_result(
                        topic=topic,
                        success=entry.get("success", False),
                        accuracy=float(entry.get("metrics", {}).get("accuracy") or 0.0)
                    )
                    self.explorer.topic_generator._generated_topics.add(topic)
                    restored_topics += 1
                if hasattr(self.explorer, 'topic_pool_manager') and self.explorer.topic_pool_manager:
                    self.explorer.topic_pool_manager.record_used(topic)
        if restored_topics > 0:
            self.logger.info(
                f"📜 [Engine] 已恢复 {restored_topics} 个 topic 到 TopicGenerator "
                f"(generated_pool={len(self.explorer.topic_generator._generated_topics)}, "
                f"used_pool={len(self.explorer.topic_pool_manager.used_topics) if hasattr(self.explorer, 'topic_pool_manager') else 'N/A'})"
            )
        
        # FIX: 恢复 DirectionCoverageEnforcer 的方向使用记录，
        # 避免重启后误以为所有方向都未使用过而强制探索已用方向
        if completed and hasattr(self.explorer, 'direction_enforcer') and self.explorer.direction_enforcer:
            restored_dirs = 0
            for entry in completed:
                direction = entry.get("direction", "")
                if direction:
                    self.explorer.direction_enforcer.update(direction)
                    restored_dirs += 1
            if restored_dirs > 0:
                self.logger.info(
                    f"📜 [Engine] 已恢复 {restored_dirs} 个方向使用记录到 DirectionEnforcer "
                    f"(used={len(self.explorer.direction_enforcer.direction_usage)}/"
                    f"{len(self.explorer.direction_enforcer.ALL_DIRECTIONS)})"
                )
        
        # v2: 不再加载旧蒸馏历史作为策略输入（避免锁定旧方向和 deepen_threshold）
        # 探索完全由 UCT 树驱动

    def _experiment_record_to_state_entry(self, record) -> Dict[str, Any]:
        """Convert ExperimentRecord (from SQLite) to state entry dict for history_tracker."""
        import json
        strategy = {}
        try:
            strategy = json.loads(record.strategy_json or "{}")
        except Exception:
            pass
        return {
            "iteration": record.iteration,
            "direction": record.direction or "general",
            "approach_type": "Unknown",
            "idea_summary": record.idea_text[:500] if record.idea_text else "",
            "success": record.success,
            "is_fake_success": record.is_fake_success,
            "is_tampered_ratio": record.is_tampered_ratio,
            "metrics": {
                "accuracy": record.accuracy,
                "eval_time": strategy.get("eval_time", 0.0),
            },
            "feedback": "",
            "timestamp": record.completed_at or record.created_at,
            "structural_cells": [],
            "parent_iteration": record.parent_iteration,
            "operators": record.operators,
            "code_flow_sequence": record.code_flow,
            "hyperparams": record.hyperparams,
        }

    def _backfill_exploration_tree(self):
        """从 completed_iterations 回填探索树（当 tree.json 缺失时）"""
        if not hasattr(self, 'exploration_tree') or not self.exploration_tree:
            return
        completed = self.state_manager.state.get("completed_iterations", [])
        if not completed:
            return
        
        def _approximate_feedback_vector(entry: dict) -> list:
            """从 state entry 近似构造 3-D feedback vector（无完整 code 时）
            
            3D vector: [empirical_accuracy, theoretical_formalization, computational_efficiency]
            与 objective_feedback.py 的 FEEDBACK_WEIGHTS_V2 对齐。
            
            v6.1 FIX: 使用模型自适应 baseline，避免低 baseline 模型（如 LLaVA ~0.70）
            的反馈被全部 clamp 到 0，导致 UCT 完全失效。
            """
            acc = float(entry.get("metrics", {}).get("accuracy", 0.0))
            # 模型自适应 baseline：以 MMTOK_BASELINE 为锚点
            base = self.MMTOK_BASELINE
            scale = max(0.01, 0.90 - base)
            empirical = max(0.0, min(1.0, (acc - base) / scale))
            # 回填时无法准确计算 theory_score / eff_score
            # 用中性值填充，让 UCT 主要依赖 accuracy 决策
            return [empirical, 0.5, 0.5]
        
        self.exploration_tree.backfill_from_state(
            completed,
            infer_domain_func=lambda direction, cells: self.explorer._infer_domain(direction, cells),
            infer_theory_func=lambda topic, direction: self.explorer._infer_theory_cluster(topic, direction),
            approximate_feedback_func=_approximate_feedback_vector,
        )
        self.exploration_tree.save()
        self.logger.info("🌲 [Engine] 探索树回填完成并已保存")

    def _sanitize_state_baselines(self):
        """清理状态中的虚假成功记录：若 best_result 得分等于原始模型 baseline 或均匀步长指纹，则视为无效"""
        has_retain_ratio = getattr(self.config, 'has_retain_ratio', True)
        best = self.state_manager.state.get("best_result")
        if best:
            best_acc = float(best.get("metrics", {}).get("accuracy") or 0)
            # v7.1 FIX: Only flag fake-success against ORIGINAL_BASELINE when the model
            # actually has a distinguishable retain_ratio (has_retain_ratio=True).
            # For LLaVA (has_retain_ratio=False), ORIGINAL_BASELINE == MMTOK_BASELINE,
            # so a result matching baseline is legitimate, not fake.
            if has_retain_ratio and abs(best_acc - self.ORIGINAL_BASELINE) < self.FAKE_SUCCESS_TOLERANCE:
                self.logger.warning(
                    f"⚠️ 检测到历史最佳结果为虚假成功 (accuracy={best_acc:.4f}="
                    f"原始模型 baseline)。已清除该记录，避免后续迭代被误导。"
                )
                self.state_manager.state["best_result"] = None
            if abs(best_acc - self.UNIFORM_STEP_BASELINE) < self.FAKE_SUCCESS_TOLERANCE:
                self.logger.warning(
                    f"⚠️ 检测到历史最佳结果为均匀步长回退 (accuracy={best_acc:.4f}="
                    f"均匀步长指纹)。已清除该记录，避免后续迭代被误导。"
                )
                self.state_manager.state["best_result"] = None
        
        # 同时从 completed_iterations 中标记这些为虚假成功
        for entry in self.state_manager.state.get("completed_iterations", []):
            acc = float(entry.get("metrics", {}).get("accuracy") or 0)
            if has_retain_ratio and abs(acc - self.ORIGINAL_BASELINE) < self.FAKE_SUCCESS_TOLERANCE:
                entry["is_fake_success"] = True
            if abs(acc - self.UNIFORM_STEP_BASELINE) < self.FAKE_SUCCESS_TOLERANCE:
                entry["is_uniform_fallback"] = True
                entry["is_fake_success"] = True
        fake_count = sum(1 for e in self.state_manager.state.get("completed_iterations", []) if e.get("is_fake_success"))
        uniform_count = sum(1 for e in self.state_manager.state.get("completed_iterations", []) if e.get("is_uniform_fallback"))
        if fake_count or uniform_count:
            self.logger.info(
                f"🧹 [Engine] 虚假成功清理完成: 标记 {fake_count} 条 fake_success, "
                f"{uniform_count} 条 uniform_fallback"
            )
        self.state_manager.save()
    
    def _update_results_md(self):
        """自动维护实验结果记录文档 experiment_results.md"""
        from autoresearch_tasks import current_task
        comps = self.state_manager.state.get("completed_iterations", [])
        if not comps:
            return
        
        lines = []
        lines.append("# AutoResearch Iteration Results")
        lines.append("")
        lines.append("| Iter | Accuracy | Success | Fake | Uniform | Tampered | Direction | Approach Type | Combo | Topic | Idea Summary |")
        lines.append("|------|----------|---------|------|---------|----------|-----------|---------------|-------|-------|--------------|")
        
        for c in comps:
            iter_num = c.get("iteration", "?")
            # 分支并行：同一迭代的多个分支各占一行，Iter 列显示 "迭代-分支"
            if c.get("branch") is not None:
                iter_num = f"{iter_num}-{c['branch']}"
            # FIX: 当 accuracy 键存在但值为 None 时，.get() 返回 None 而非默认值。
            # 使用显式检查确保 None 值显示为 N/A。
            _m = c.get("metrics", {})
            _acc = _m.get(current_task().primary_metric, _m.get("accuracy"))
            acc = "N/A" if _acc is None else _acc
            # [SmokeMode] 冒烟模式记录：Accuracy 列渲染 smoke✓/smoke✗（无 acc 数值）
            if c.get("smoke_passed") is not None:
                acc = "smoke✓" if c.get("smoke_passed") else "smoke✗"
            success = "✓" if c.get("success") else "✗"
            fake = "✓" if c.get("is_fake_success") else ""
            uniform = "✓" if c.get("is_uniform_fallback") else ""
            tampered = "✓" if c.get("is_tampered_ratio") else ""
            direction = c.get("direction", "")
            approach = c.get("approach_type", "")
            # Phase I2: 张量组合列（组合探索关闭/未记录时显示 "-"）
            combo = c.get("tensor_combo") or "-"
            topic = c.get("topic", "")
            summary = c.get("idea_summary", "")
            summary_short = summary.replace("\n", " ")[:80] + ("..." if len(summary) > 80 else "")
            lines.append(
                f"| {iter_num} | {acc} | {success} | {fake} | {uniform} | {tampered} | {direction} | {approach} | {combo} | {topic} | {summary_short} |"
            )
        
        lines.append("")
        lines.append(f"**Total experiment records:** {len(comps)}")
        best = self.state_manager.state.get("best_result")
        if best:
            lines.append(
                f"**Best legitimate result:** Iteration {best.get('iteration')} with accuracy {best.get('metrics',{}).get('accuracy')}"
            )

        # v12.0: Learning Progress 小节（depth-as-learning 的可观测度量）
        _nb = getattr(self, 'notebook', None)
        if _nb is not None:
            try:
                learning_md = _nb.learning_summary()
                if learning_md:
                    lines.append(learning_md)
            except Exception:
                pass
        
        md_path = os.path.join(self.config.log_dir, "experiment_results.md")
        try:
            with open(md_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except Exception as e:
            self.logger.warning(f"⚠️ 更新实验结果记录失败: {e}")

    def _append_elite_algorithm(self, result: Dict[str, Any], iteration: int):
        """v4.1: Append high-performing algorithm to model_family-isolated elite archive."""
        # v4.1: Use model_family-specific archive path to prevent cross-model pollution
        elite_path = derive_elite_archive_path(self.config, os.path.dirname(__file__))
        # Adaptive threshold: baseline + 0.015, mimicking the TreeController logic
        from autoresearch_tasks import current_task
        base = getattr(self.config, 'original_baseline', None) or getattr(
            self.config, 'mmtok_baseline', current_task().baselines.mmtok_baseline)
        elite_threshold = getattr(self.config, 'deepen_threshold_base', float(base) + 0.015)
        try:
            data = {"threshold": elite_threshold, "sources": [], "total_count": 0, "algorithms": []}
            if os.path.exists(elite_path):
                with open(elite_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            acc = float(result.get("metrics", {}).get("accuracy") or 0.0)
            entry = {
                "iteration": iteration,
                "source": "current_run",
                "accuracy": acc,
                "direction": result.get("direction", "unknown"),
                "operators": result.get("operators", []),
                "core_intuition": "",
                "key_insights": []
            }
            # deduplicate by iteration
            existing_iters = {a.get("iteration") for a in data.get("algorithms", [])}
            # v12.0: n=1 高分离群未通过复现验证前，不进入 elite archive
            _nb = getattr(self, 'notebook', None)
            _verified = True
            if _nb is not None and getattr(self.config, 'notebook_require_verified_for_elite', True):
                try:
                    _verified = _nb.is_verified(iteration)
                except Exception:
                    _verified = True
            if not _verified:
                self.logger.info(
                    f"🚫 [NotebookGate] iter={iteration} acc={acc:.4f} 未复现验证，暂缓加入 elite archive"
                )
            elif iteration not in existing_iters and acc >= elite_threshold:
                data.setdefault("algorithms", []).append(entry)
                data["total_count"] = len(data["algorithms"])
                with open(elite_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                self.logger.info(f"🏆 Elite archive updated: iter={iteration} acc={acc:.4f} (threshold={elite_threshold:.4f}, total={data['total_count']})")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to update elite archive: {e}")
    
    def _reclassify_direction_by_code(self, code: str, fallback_direction: str) -> str:
        """基于实际代码重新分类方向，修正 Idea 文本中的误标
        
        v7.1 FIX: topk/argsort 是所有子集选择算法的通用最终选择操作，
        不能单独作为 ranking 方向的判定依据。改为要求额外的 ranking 核心逻辑关键词。
        
        Phase G3a: 规则表迁移至 TaskProfile.direction_reclassify_rules
        （数据逐字不变，格式见 mmtok_common.DIRECTION_RECLASSIFY_RULES 注释）；
        空表 = 直接返回 fallback_direction（跳过重分类）。
        """
        if not code:
            return fallback_direction
        from autoresearch_tasks import current_task
        rules_table = current_task().direction_reclassify_rules
        if not rules_table or not rules_table.get("rules"):
            return fallback_direction
        code_lower = code.lower()
        haystacks = {
            "code": code,
            "code_lower": code_lower,
            "code_no_mask": code_lower.replace('attention_mask', ''),
        }

        def _group_match(group: Dict[str, Any]) -> bool:
            h = haystacks[group.get("haystack", "code_lower")]
            if "all" in group:
                return all(p in h for p in group["all"])
            return any(p in h for p in group.get("any", []))

        signals: Dict[str, bool] = {}
        for name, spec in rules_table.get("signals", []):
            if "any_groups" in spec:
                matched = any(_group_match(g) for g in spec["any_groups"])
            else:
                matched = _group_match(spec)
            if matched and any(signals.get(s) for s in spec.get("unless_signals", [])):
                matched = False
            signals[name] = matched

        # v7.1: ranking 必须包含核心 ranking 逻辑关键词，而不仅仅是最终的 topk/argsort
        # 保留旧检测作为弱信号：仅当存在 ranking 相关变量/函数名时才认定
        ranking_spec = rules_table.get("ranking", {})
        has_ranking_core = any(r in code_lower for r in ranking_spec.get("core", []))
        has_ranking_weak = any(w in code_lower for w in ranking_spec.get("weak", []))
        is_ranking = has_ranking_core or (
            has_ranking_weak
            and not any(signals.get(s) for s in ranking_spec.get("weak_unless_signals", []))
        )

        for rule in rules_table["rules"]:
            if not all(signals.get(s) for s in rule.get("if_signals", [])):
                continue
            if any(signals.get(s) for s in rule.get("unless_signals", [])):
                continue
            if rule.get("if_any_signals") and not any(
                    signals.get(s) for s in rule["if_any_signals"]):
                continue
            if rule.get("if_ranking") and not is_ranking:
                continue
            if rule.get("if_any_patterns"):
                h = haystacks[rule.get("haystack", "code_lower")]
                if not any(p in h for p in rule["if_any_patterns"]):
                    continue
            if "return_first_of" in rule:
                for sig, direction in rule["return_first_of"]:
                    if signals.get(sig):
                        return direction
                continue
            return rule["return"]
        return fallback_direction
    
    def _classify_agent_failure(self, strategy: Dict[str, Any]) -> str:
        """v7.3: Classify why the Agent failed to generate ideas.
        
        First tries to read the precise failure mode cached by the Orchestrator
        from the last SubAgentResult. Falls back to heuristic classification.
        """
        # Priority 1: precise failure mode from Orchestrator
        if (hasattr(self.explorer, '_orchestrator') 
                and self.explorer._orchestrator 
                and self.explorer._orchestrator.last_failure_mode):
            return self.explorer._orchestrator.last_failure_mode
        
        # Fallback: heuristic classification based on strategy and history
        target_domain = strategy.get("target_domain", "")
        if target_domain and target_domain != "general":
            domain_attempts = sum(
                1 for h in self.explorer.history_tracker.idea_history[-20:]
                if h.get("direction") == target_domain
            )
            if domain_attempts >= 5:
                return "recall_insufficient"
        
        recent_duplicates = sum(
            1 for h in self.explorer.history_tracker.idea_history[-10:]
            if "duplicate" in (h.get("failure_reason") or "").lower()
        )
        if recent_duplicates >= 3:
            return "duplicate_rejected"
        
        return "empty_generation"

    def _ensure_backup(self):
        """确保原始代码已备份"""
        backup_marker = os.path.join(
            os.path.dirname(self.config.target_patch_file), 
            ".autoresearch_backups", 
            ".original_backed_up"
        )
        if not os.path.exists(backup_marker):
            backup_original_code(self.config.target_patch_file, self.logger)
            # 创建标记文件表示已备份
            os.makedirs(os.path.dirname(backup_marker), exist_ok=True)
            with open(backup_marker, 'w') as f:
                f.write(datetime.now().isoformat())
    
    def _auto_fix_missing_imports(self, code: str) -> str:
        """
        自动注入常见的缺失 import（确定性修复，不调用 LLM）。
        在保存代码到文件之前调用，避免运行时 NameError。
        """
        import re

        # 提取已有 import 别名
        imported_aliases = set()
        for m in re.finditer(r'import\s+[\w.]+(?:\s+as\s+(\w+))?', code):
            if m.group(1):
                imported_aliases.add(m.group(1))
        for m in re.finditer(r'from\s+[\w.]+\s+import\s+([^#\n]+)', code):
            imports = m.group(1)
            for item in imports.split(','):
                item = item.strip()
                as_m = re.search(r'\w+\s+as\s+(\w+)', item)
                if as_m:
                    imported_aliases.add(as_m.group(1))

        # (使用模式, import 语句, 别名)
        missing_import_checks = [
            (r'\bF\.[a-zA-Z_]', 'import torch.nn.functional as F', 'F'),
            (r'\bnp\.[a-zA-Z_]', 'import numpy as np', 'np'),
        ]

        for usage_pattern, import_stmt, alias in missing_import_checks:
            if re.search(usage_pattern, code) and alias not in imported_aliases:
                # 找到最后一个 import 语句，在其后插入
                last_import_match = None
                for m in re.finditer(r'^(?:import|from)\s+[^\n]+\n', code, re.MULTILINE):
                    last_import_match = m

                if last_import_match:
                    insert_pos = last_import_match.end()
                    code = code[:insert_pos] + import_stmt + '\n' + code[insert_pos:]
                else:
                    # 无现有 import，在文件第一个非注释行之前插入
                    lines = code.split('\n')
                    insert_line = 0
                    in_docstring = False
                    for i, line in enumerate(lines):
                        stripped = line.strip()
                        if not stripped:
                            continue
                        if stripped.startswith('"""') or stripped.startswith("'''"):
                            if stripped.count('"""') == 1 or stripped.count("'''") == 1:
                                in_docstring = not in_docstring
                            continue
                        if in_docstring:
                            continue
                        if stripped.startswith('#'):
                            continue
                        insert_line = i
                        break
                    lines.insert(insert_line, import_stmt)
                    code = '\n'.join(lines)

                self.logger.info(f"🔧 [AutoFix] 自动注入缺失 import: {import_stmt}")

        return code

    def _mechanism_gate(self, code: str) -> Tuple[bool, str]:
        """评测前机制簇闸门：命中饱和/超配额机制簇的代码不值得花 GPU 全量评测。

        在代码级（而非 idea 文本级）提取算子指纹——覆盖率和准确度都更高。
        命中时返回 (False, 反馈文本)，调用方按静态门禁处理：Coder repair 重试，
        重试耗尽则本轮在评测前失败，节省一次全量评测（~5min GPU）。
        对 breadth/depth/cross 所有产物统一生效。
        """
        try:
            # A2 消融臂：关闭机制级多样性时闸门不生效
            if getattr(self.config, 'disable_diversity_mechanisms', False):
                return True, ""
            ht = getattr(self.explorer, 'history_tracker', None)
            if ht is None:
                return True, ""
            ops = ht._extract_operators(code)
            key = ht.mechanism_cluster_key(ops)
            if not key:
                return True, ""
            # 单元素簇（('GEOMETRY',)/('RANKING',)/('ATOM_norm',) 等）指纹太粗：
            # 大量不同机制共享同一标签，硬拦截会误伤真正的新机制。
            # 单元素簇只做软警告（anti-deadlock guidance），硬拦截只用于
            # ≥2 算子的精确指纹簇（GREEDY_ORTH / ATOM_norm+ATOM_projection 等）。
            if len(key) < 2:
                return True, ""
            ops_str = "+".join(key)
            sat = ht.get_saturated_mechanisms().get(key)
            if sat:
                return False, (
                    f"Mechanism saturation gate: the operator combination [{ops_str}] is a "
                    f"SATURATED mechanism family ({sat['type']}, {sat['runs']} runs, "
                    f"best={sat['best_acc']:.4f}). Any re-packaging of it reproduces the same "
                    f"selection behavior and the same score. You MUST implement a fundamentally "
                    f"different selection operator (different gain function / geometry / update "
                    f"rule) — not another variant of this family."
                )
            if key in ht.get_over_quota_mechanisms():
                return False, (
                    f"Mechanism quota gate: the operator combination [{ops_str}] exceeds its "
                    f"sliding-window share quota — recent iterations are dominated by this "
                    f"mechanism family. Implement a fundamentally different selection operator."
                )
            return True, ""
        except Exception as e:
            self.logger.warning(f"⚠️ [MechanismGate] 异常（非致命，放行）: {e}")
            return True, ""

    def _selection_clone_gate(self, iteration: int) -> Optional[str]:
        """行为级查重（hard reject）：探针采集的选择指纹与历史一致 = 功能克隆。

        代码文本不同但在多个结构化合成输入（随机/聚簇/grid=None）上选中索引
        完全相同的算法，是同一机制的换皮实现（0.493/0.308 同分指纹事件），
        评测它纯属浪费 GPU。命中返回反馈文本（调用方按静态门禁打回修复）；
        未命中则登记 pending 指纹供轮末落历史，返回 None。
        """
        # A2 消融臂：关闭机制级多样性时闸门不生效
        if getattr(self.config, 'disable_diversity_mechanisms', False):
            return None
        probe = getattr(self, 'token_count_probe', None)
        sig = getattr(probe, 'last_signature', None) if probe else None
        if not sig:
            return None
        ht = getattr(self.explorer, 'history_tracker', None)
        if ht is None:
            return None
        dup = ht.check_selection_clone(sig)
        if dup:
            return (
                f"Behavioral clone detected: this implementation produces token selections "
                f"IDENTICAL to iter_{dup['iteration']} (acc={dup['acc']:.4f}) on every probe "
                f"input, despite different code. Re-implementing the same selection behavior "
                f"wastes a full evaluation. Change the actual selection behavior — a different "
                f"gain function, geometry, or update rule — not merely the packaging."
            )
        ht.pending_signatures[iteration] = sig
        return None

    def _static_smoke_test(self, code: str) -> Tuple[bool, str]:
        """
        静态 smoke test：在不运行完整模型的情况下，检查代码是否包含已知的均匀步长回退模式，
        以及是否包含足够的算法操作（避免生成的是空壳代码）。
        """
        # Phase F3：tensor 契约 / 注入锚点 / 后处理检查组统一从当前任务 patch_spec 读取
        from autoresearch_tasks import current_task
        profile = current_task()
        patch_spec = profile.patch_spec
        model_family = getattr(self.config, 'model_family', 'qwen')
        main_tensor = getattr(self.config, 'main_tensor_name', patch_spec.main_tensor_name)
        has_retain_ratio = getattr(self.config, 'has_retain_ratio', True)
        patch_mode = getattr(self.config, 'patch_mode', 'model')

        # 构建同时匹配各主 tensor 候选名的正则前缀
        tensor_alt = "(?:" + "|".join(re.escape(t) for t in patch_spec.smoke_tensor_names) + ")"
        # G2：target token 变量名从 patch_spec 读取（默认值逐字 == 现字面量）
        tvt_var = re.escape(patch_spec.target_tokens_var)

        # 0. 接口完整性：模型文件 import 的 selector 类必须在生成代码中存在。
        # iter_1677 事故：Coder 整文件重写时丢掉 AutoResearchSelector，评测子进程
        # import 阶段才炸，白烧一次评测 + 两轮修复。类名从 patch_spec.probe_entries
        # 读取（数据驱动，qwen/llava 通用）。
        missing_classes = [
            e["class_name"] for e in patch_spec.probe_entries
            if e.get("class_name")
            and not re.search(rf'class\s+{re.escape(e["class_name"])}\b', code)
        ]
        if missing_classes:
            return False, (
                f"Static smoke test failed: required selector class(es) {missing_classes} "
                f"are missing. The model file imports them from this module — dropping them "
                f"crashes the eval at import time. Keep the original file's full public "
                f"interface and modify ONLY the ALGORITHM CORE region."
            )

        # 1. 检查已知的均匀步长指纹模式
        # 修复：不再粗暴禁止所有 torch.arange(..., step=...)，只拦截明确的均匀回退计算
        uniform_patterns = [
            # 明确的均匀步长计算：step = max(1, N // target_vision_tokens) 是典型的崩溃回退模式
            rf'step\s*=\s*max\s*\(\s*1\s*,\s*{tensor_alt}\.shape\[0\]\s*//\s*{tvt_var}\s*\)',
            rf'step\s*=\s*max\s*\(\s*1\s*,\s*N\s*//\s*{tvt_var}\s*\)',
            # 只有当 torch.arange 的结果直接作为 selected_indices 且 step 明显是均匀采样时才拦截
            rf'selected_indices\s*=\s*torch\.arange\s*\(\s*0\s*,\s*(?:{tensor_alt}\.shape\[0\]|N)\s*,\s*step\s*=\s*max\s*\(',
        ]
        for pattern in uniform_patterns:
            if re.search(pattern, code):
                return False, (
                    "Static smoke test failed: uniform step fallback pattern detected in code. "
                    f"This will cause the algorithm to silently fall back to the {self.UNIFORM_STEP_BASELINE:.4f} fingerprint score after crash. "
                    "Please remove uniform step logic such as `step = max(1, N // target_vision_tokens)` and implement a real algorithm."
                )
        
        # 1.5 检查保留全部 token 的回退模式
        retain_all_patterns = [
            rf'selected_indices\s*=\s*torch\.arange\s*\(\s*{tensor_alt}\.shape\[0\]\s*\)',
            rf'selected_indices\s*=\s*torch\.arange\s*\(\s*0\s*,\s*{tensor_alt}\.shape\[0\]\s*\)',
        ]
        for pattern in retain_all_patterns:
            if re.search(pattern, code):
                return False, (
                    "Static smoke test failed: retain-all-tokens fallback pattern detected in code. "
                    f"This will cause a {self.ORIGINAL_BASELINE:.4f} fake success (original model baseline). "
                    "Please replace it with torch.randperm(N)[:target_vision_tokens] or a real algorithm."
                )
        
        # 1.55 检查 Qwen 后处理完整性（防止 no-op：selected_indices 被计算但从未使用）
        # 【selector 模式】跳过 model 特定的后处理检查
        if_anchor = patch_spec.inject_if_anchor
        if patch_mode != "selector" and model_family == 'qwen' and if_anchor in code:
            # 提取注入锚点块的内容（到下一个同缩进或更低缩进的结构为止）
            select_pixel_match = re.search(
                re.escape(if_anchor) + r'(.*?)(?=\n        [^\s#]|\n    [^\s#]|\Z)',
                code, re.DOTALL
            )
            if select_pixel_match:
                block = select_pixel_match.group(1)
                # 检查是否计算了 selected_indices
                has_selected_indices = 'selected_indices' in block
                # 后处理完整性检查（token 替换 / position_ids / attention_mask，
                # 判定子串与正则为任务数据，见 patch_spec.smoke_postprocess_checks）
                missing = []
                for check in patch_spec.smoke_postprocess_checks:
                    ok = any(s in block for s in check["substrings"]) or any(
                        re.search(p, block) is not None for p in check["regexes"])
                    if not ok:
                        missing.append(check["label"])
                if has_selected_indices and missing:
                    return False, (
                        "Static smoke test failed: incomplete post-processing in Qwen forward. "
                        "Code computes selected_indices but lacks critical steps to actually apply the selection: "
                        f"missing {'; '.join(missing)}. "
                        "Without these steps, the algorithm is a no-op and will retain all tokens, producing a fake success."
                    )
        
        # 1.6 检查保留比例是否被偷偷修改 — 从 eval_cmd 推导期望 retain_ratio，避免硬编码 0.1 与实际配置矛盾
        # 【selector 模式】跳过 TOKEN_RETAIN_RATIO 检查，因为 selector 不直接处理 retain ratio
        expected_retain_ratio = self._get_expected_retain_ratio()
        
        if patch_mode != "selector" and has_retain_ratio:
            # G2：env 键名从 profile.retain_ratio_env_keys 读取（去掉尾部 "="，逐字含现 TOKEN_RETAIN_RATIO）
            env_keys_alt = "(?:" + "|".join(
                re.escape(k.rstrip("=")) for k in profile.retain_ratio_env_keys) + ")"
            ratio_match = re.search(
                rf'{env_keys_alt}\s*,\s*"([^"]+)"\s*\)',
                code,
            )
            if ratio_match:
                detected_default = ratio_match.group(1)
                if detected_default != expected_retain_ratio:
                    return False, (
                        f"Static smoke test failed: TOKEN_RETAIN_RATIO default value in code is {detected_default}, "
                        f"but the experiment mandates it must be {expected_retain_ratio}. Secretly modifying the retention ratio is cheating, "
                        f"and will lead to unfairly high accuracy (retaining more tokens). Please strictly set the default value to \"{expected_retain_ratio}\"."
                    )
        
        # 1.7 检查硬编码保留比例（绕过 TOKEN_RETAIN_RATIO 的变相作弊）
        # 【selector 模式】放宽检查，因为 selector 中 n 和 k_max 已由框架计算
        hardcoded_ratio_patterns = [
            # 删除过于宽泛的 k\s*= 模式，避免误杀合法代码（如 num_heads = N // 64）
            rf'{tvt_var}\s*=\s*[Nn]\s*//\s*\d+',
            rf'{re.escape(main_tensor)}\.shape\[0\]\s*//\s*\d+',
            rf'{tvt_var}\s*=\s*int\([^)]*shape\[0\]\s*//\s*\d+[^)]*\)',
        ]
        for pattern in hardcoded_ratio_patterns:
            if re.search(pattern, code):
                if patch_mode != "selector" and has_retain_ratio:
                    return False, (
                        "Static smoke test failed: hard-coded retention ratio detected in code. "
                        f"The experiment mandates retaining exactly {expected_retain_ratio} of tokens via TOKEN_RETAIN_RATIO. "
                        "Hard-coding a different ratio (e.g. N // 2) is cheating and will be rejected."
                    )
                elif patch_mode != "selector":
                    return False, (
                        "Static smoke test failed: hard-coded retention ratio detected in code. "
                        f"The experiment mandates using the configured target_vision_tokens from self._mmtok_core.target_vision_tokens. "
                        "Hard-coding a different ratio (e.g. N // 2) is cheating and will be rejected."
                    )
        
        # 2.5 检查常见 PyTorch/NumPy 缩写是否有对应 import（预防 NameError）
        imported_aliases = set()
        for m in re.finditer(r'import\s+[\w.]+(?:\s+as\s+(\w+))?', code):
            if m.group(1):
                imported_aliases.add(m.group(1))
        for m in re.finditer(r'from\s+[\w.]+\s+import\s+([^#\n]+)', code):
            imports = m.group(1)
            for item in imports.split(','):
                item = item.strip()
                as_m = re.search(r'\w+\s+as\s+(\w+)', item)
                if as_m:
                    imported_aliases.add(as_m.group(1))
        # 检查 F (torch.nn.functional)
        if re.search(r'\bF\.[a-zA-Z_]', code) and 'F' not in imported_aliases:
            return False, (
                "Static smoke test failed: code uses `F.xxx` (e.g., F.conv2d, F.normalize) "
                "but `import torch.nn.functional as F` is missing. "
                "Please add the missing import statement."
            )

        # 2. 检查代码是否包含足够的算法操作（避免空壳代码）
        # 修复：扩充基础操作列表，避免轻量级有效算法被误判为空壳
        algorithmic_markers = [
            'torch.norm', 'torch.matmul', 'torch.mm', 'torch.topk',
            'torch.cdist', 'torch.cosine_similarity', 'F.cosine_similarity',
            'torch.sort', 'torch.argsort', 'torch.softmax', 'torch.log_softmax',
            'torch.mean', 'torch.std', 'torch.var', 'torch.cov',
            'torch.fft', 'torch.einsum',
            'F.normalize', 'torch.clamp', 'torch.where',
            # 新增：更多基础 PyTorch 操作，避免轻量级算法被误杀
            'torch.index_select', 'torch.scatter', 'torch.gather',
            'torch.stack', 'torch.cat', 'torch.unsqueeze', 'torch.flatten',
            'torch.sum', 'torch.max', 'torch.min', 'torch.abs',
            'torch.exp', 'torch.log', 'torch.sqrt', 'torch.pow', 'torch.cumsum',
            'torch.diff', 'torch.unique', 'torch.bincount', 'torch.histc',
            'torch.masked_select', 'torch.nonzero', 'torch.repeat_interleave',
        ]
        has_algorithmic = any(marker in code for marker in algorithmic_markers)
        
        # 如果代码只有非常简单的索引操作（如直接 arange 或 range），视为可疑
        simple_only = re.search(r'selected_indices\s*=\s*torch\.arange\s*\(', code)
        if simple_only and not has_algorithmic:
            return False, (
                "Static smoke test failed: no meaningful algorithmic operations detected in code. "
                "selected_indices appears to be a simple arange without implementing the mathematical logic described in the algorithm design."
            )
        
        return True, "Static smoke test passed"

    # ------------------------------------------------------------------
    # [SmokeMode] smoke_only_mode 专用：GPU 显存闸 + 真实 GPU 冒烟。
    # 以下三个方法仅在 config.smoke_only_mode=True 时被调用；
    # 默认配置下主流程逐字不变。
    # ------------------------------------------------------------------
    def _smoke_wait_for_gpu(self) -> Optional[str]:
        """[SmokeMode] 显存闸：轮询等待一张空闲显存 >= smoke_min_free_gpu_gb 的卡。

        语义仿 SafeExecutor._wait_for_gpu：gpu_wait_seconds<=0 表示无限等待直到有卡
        满足门槛；>0 时等待超过该时长后兜底使用当前最空闲的卡。GPU 探测整体失败
        （nvidia-smi 与 PyTorch 均不可用）或被中断时返回 None。
        """
        min_free_mib = float(getattr(self.config, 'smoke_min_free_gpu_gb', 30) or 0) * 1024
        wait_budget = int(getattr(self.config, 'gpu_wait_seconds', 600))
        poll = max(5, int(getattr(self.config, 'gpu_poll_interval', 60)))
        waited = 0
        while True:
            if self._interrupted:
                self.logger.info("🛑 [SmokeMode] GPU 显存检查: 检测到中断请求，停止等待")
                return None
            gpus = SafeExecutor._query_free_gpus()
            if not gpus:
                self.logger.warning(
                    "⚠️ [SmokeMode] GPU 显存检查: 探测失败（nvidia-smi 与 PyTorch 均不可用）"
                )
                return None
            best_idx, best_free = gpus[0]
            if best_free >= min_free_mib:
                self.logger.info(
                    f"🎮 [SmokeMode] GPU 显存检查: 选择 GPU {best_idx}"
                    f"（剩余 {best_free/1024:.1f} GiB ≥ 需求 {min_free_mib/1024:.0f} GiB）"
                )
                return str(best_idx)
            # gpu_wait_seconds <= 0 表示无限等待，直到有卡满足门槛
            if wait_budget > 0 and waited >= wait_budget:
                self.logger.warning(
                    f"⚠️ [SmokeMode] GPU 显存检查: 等待 {waited}s 后仍无 GPU 满足 "
                    f"{min_free_mib/1024:.0f} GiB（最空闲 GPU {best_idx} 仅余 "
                    f"{best_free/1024:.1f} GiB），兜底使用 GPU {best_idx}"
                )
                return str(best_idx)
            wait_hint = (f"已等待 {waited}s" if wait_budget > 0
                         else f"已等待 {waited}s（无限等待直到有卡可用）")
            self.logger.info(
                f"⏳ [SmokeMode] GPU 显存检查: 所有 GPU 剩余显存均不足 "
                f"{min_free_mib/1024:.0f} GiB（最空闲 GPU {best_idx} 余 "
                f"{best_free/1024:.1f} GiB），{poll}s 后重试（{wait_hint}）..."
            )
            time.sleep(poll)
            waited += poll

    def _gpu_smoke_check(self, target_path: str) -> Tuple[bool, str]:
        """[SmokeMode] GPU 显存闸 + 真实 GPU 冒烟。

        冒烟内容保持 TokenCountProbe 现有语义：真实实例化 patch 后的 selector、
        用合成输入跑一次 forward、校验输出 token 数与维度；区别是在显存闸选定的
        空闲 GPU 上运行（strict 模式：任何异常判失败），跑完释放显存。
        """
        probe = getattr(self, 'token_count_probe', None)
        if probe is None:
            # 与 pre-eval 探针缺失语义一致：探针未启用时冒烟不拦截
            return True, "token_count_probe 未启用，冒烟跳过"
        gpu = self._smoke_wait_for_gpu()
        if gpu is None:
            return False, "GPU 显存闸失败：无可用 GPU（探测失败或被中断）"
        try:
            ok, msg = probe.verify_on_gpu(target_path, f"cuda:{gpu}")
        except Exception as e:
            return False, f"GPU 冒烟异常: {type(e).__name__}: {e}"
        return ok, f"[GPU {gpu}] {msg}"

    def _update_idea_history_maybe_skip_tree(self, idea, result, iteration,
                                             code=None, strategy=None):
        """[SmokeMode] 冒烟模式下 update_idea_history 跳过探索树 backprop。

        树 backprop 在 ExplorerAgent.update_idea_history 内部按
        result["metrics"]["accuracy"] 计算 feedback_vector（explorer 目录不可修改）。
        smoke 模式没有真实 acc：注入中性 acc 会污染 state store / best performers，
        传 acc=0.0 会对 UCT 叶节点施加惩罚性 backprop。最安全的方式是临时摘除
        explorer.exploration_tree，使其内部 `if self.exploration_tree:` 块（含
        tree_controller.record_iteration）整体跳过；探索信号改由机制覆盖统计承担。
        非 smoke 模式与原调用逐字一致。
        """
        kwargs = dict(
            code=code,
            static_smoke_test=self._static_smoke_test,
            fallback_detector=getattr(self.coder, 'fallback_detector', None),
            api_validator=getattr(self.coder, 'api_validator', None),
            strategy=strategy,
        )
        if not getattr(self.config, 'smoke_only_mode', False):
            self.explorer.update_idea_history(idea, result, iteration, **kwargs)
            return
        tree_bak = getattr(self.explorer, 'exploration_tree', None)
        self.explorer.exploration_tree = None
        try:
            self.explorer.update_idea_history(idea, result, iteration, **kwargs)
        finally:
            self.explorer.exploration_tree = tree_bak

    # ------------------------------------------------------------------
    # [AsyncEval] 异步评测流水线：快照环境 + 交出 + 槽位等待 + 主线程回收。
    # 以下方法仅在 _async_eval_enabled() 为真时被调用（async_eval_mode 且非
    # smoke 且非并行分支）；不满足时主流程逐字同原行为。
    # ------------------------------------------------------------------
    def _async_eval_enabled(self) -> bool:
        """异步评测流水线的生效条件（smoke 路径、并行分支路径完全不变）。"""
        return (bool(getattr(self.config, 'async_eval_mode', False))
                and not bool(getattr(self.config, 'smoke_only_mode', False))
                and int(getattr(self.config, 'parallel_branches', 0) or 0) <= 0)

    def _create_async_eval_env(self, iteration: int, code: str) -> str:
        """[AsyncEval] 为 iter=N 的评测创建独立快照环境，返回 PYTHONPATH 前缀目录。

        做法与 _setup_branch_envs 一致：拷贝 mmtok 包到
        <log_dir>/async_envs/iter_N/，再把本轮生成代码显式写入快照内
        target_patch_file 的对应位置。硬性正确性要求：lmms_eval 子进程 import
        selector 的时刻可能晚于下一轮 coder 覆盖 target_patch_file，在途评测
        必须与被覆盖的实时文件完全隔离。
        """
        import shutil
        mmtok_pkg_dir = self._resolve_mmtok_pkg_dir()
        rel = os.path.relpath(
            os.path.abspath(self.config.target_patch_file),
            os.path.dirname(mmtok_pkg_dir))
        env_dir = os.path.join(self.config.log_dir, "async_envs", f"iter_{iteration:03d}")
        if os.path.exists(env_dir):
            shutil.rmtree(env_dir)
        os.makedirs(env_dir, exist_ok=True)
        shutil.copytree(
            mmtok_pkg_dir,
            os.path.join(env_dir, "mmtok"),
            ignore=shutil.ignore_patterns(
                '__pycache__', 'lmms_eval_results', '*.pyc'),
            dirs_exist_ok=True)
        snapshot_patch = os.path.join(env_dir, rel)
        os.makedirs(os.path.dirname(snapshot_patch), exist_ok=True)
        with open(snapshot_patch, 'w', encoding='utf-8') as f:
            f.write(code)
        return env_dir

    def _cleanup_async_eval_env(self, iteration: int):
        """[AsyncEval] 回收后删除快照环境（代码备份在 iter_dir/2_modified_code_*.py，
        快照留着只会随迭代数线性膨胀）。"""
        import shutil
        env_dir = os.path.join(self.config.log_dir, "async_envs", f"iter_{iteration:03d}")
        try:
            if os.path.exists(env_dir):
                shutil.rmtree(env_dir)
        except Exception as e:
            self.logger.warning(f"⚠️ [AsyncEval] 清理快照环境失败 {env_dir}: {e}")

    def _async_hard_timeout_seconds(self) -> float:
        """[AsyncEval] 单个在途评测的硬超时（秒）。0=自动：
        等 GPU 预算（gpu_wait_seconds，<=0 按 3600 计）+ 评测与重试预算
        （timeout_seconds * (eval_retry_attempts + 1)）+ 900s 余量。"""
        cfg = self.config
        explicit = int(getattr(cfg, 'async_eval_hard_timeout_seconds', 0) or 0)
        if explicit > 0:
            return float(explicit)
        gpu_wait = int(getattr(cfg, 'gpu_wait_seconds', 600) or 0)
        if gpu_wait <= 0:
            gpu_wait = 3600
        timeout_s = int(getattr(cfg, 'timeout_seconds', 1200) or 1200)
        retries = int(getattr(cfg, 'eval_retry_attempts', 2) or 0)
        return float(gpu_wait + timeout_s * (retries + 1) + 900)

    def _adaptive_async_cap(self, max_async: int) -> int:
        """自适应并发槽位：min(max_async_evals, 合格卡数)，下限 1。

        合格卡 = 空闲 ≥ min_free_gpu_gb 且未被引擎评测认领。外部任务挤占时
        临时降槽——此时强发只会 OOM，不算牺牲吞吐；卡空闲后自动恢复。
        探测失败回退 max_async（保守，不改变现状）。
        """
        try:
            min_free_mib = float(getattr(self.config, 'min_free_gpu_gb', 0.0) or 0.0) * 1024
            eligible = SafeExecutor.eligible_gpu_count(min_free_mib)
            return max(1, min(max_async, eligible))
        except Exception:
            return max_async

    def _async_wait_slot(self, iteration: int):
        """[AsyncEval] 在途数达到 max_async_evals 时主循环阻塞等槽位（poll + drain）。

        等待期间若 enable_lookahead_production，主线程顺带预生产下一轮
        iteration+1 的 idea+code（look-ahead，只生产一次）：槽位释放后
        下一轮 run_iteration 直接快照提交，消除评测间的 LLM 串行空窗。
        """
        mgr = self.async_eval_manager
        max_async = max(1, int(getattr(self.config, 'max_async_evals', 2) or 2))
        poll = 10
        waited = 0
        while True:
            cap = self._adaptive_async_cap(max_async)
            if mgr.inflight_count() < cap:
                break
            if self._interrupted:
                # run_iteration 的 except 只捕 Exception；KeyboardInterrupt 直接
                # 终止主循环（与 Ctrl+C 语义一致，状态已由 signal handler 保存）
                raise KeyboardInterrupt("等异步评测槽位时被中断")
            if (self._lookahead is None
                    and self._lookahead_attempted_for != iteration + 1):
                # 先回收已完成评测（记账更新历史），让预生产吃到最新历史；
                # 每次等待只尝试预生产一次（attempted 标记防热循环）
                self._lookahead_attempted_for = iteration + 1
                self._drain_async_evals()
                if mgr.inflight_count() >= self._adaptive_async_cap(max_async):
                    self._produce_lookahead(iteration + 1)
                    continue  # 预生产耗时较长，结束后立刻重查槽位
            if waited == 0 or waited % 60 == 0:
                self.logger.info(
                    f"⏳ [AsyncEval] iter={iteration} 在途 "
                    f"{mgr.inflight_count()}/{cap} 已满（自适应槽位，max={max_async}），等待..."
                    f"（已等待 {waited // 60}m）"
                )
            time.sleep(poll)
            waited += poll
            self._drain_async_evals()

    def _drain_async_evals(self):
        """[AsyncEval] 主线程回收点：drain 已完成评测，逐条用真实 acc 补全记账。

        记账走 _finalize_iteration_result 完整路径（baseline 校验 / fake/uniform/
        tamper 审计 / best_code.py 更新 / update_idea_history 含树 backprop /
        state_manager.update_iteration / results.md 重生成）；finalize 的
        iteration 参数用该评测自己的轮号，与当前主循环轮号无关（finalize 内部
        不依赖"当前轮==本轮"：iteration 全部显式传参，仅 last_used_topic 是
        引擎全局态，此处先换回该评测所属轮的 topic 再调）。
        """
        mgr = getattr(self, 'async_eval_manager', None)
        if mgr is None:
            return
        drained = mgr.drain()
        # 硬超时收割：在途过久的评测放弃等待，按失败记账（worker 自身有界——
        # 子进程受 timeout_seconds、等 GPU 受 gpu_wait_seconds 约束，会自行退出；
        # 其迟到结果由 drain() 丢弃）。防止个别卡死 worker 无限拖住主流程。
        hard_timeout = self._async_hard_timeout_seconds()
        for task in mgr.reap_timed_out(hard_timeout):
            self.logger.error(
                f"⏱️ [AsyncEval] iter={task.iteration} 在途超过硬超时 "
                f"{hard_timeout:.0f}s，按失败收割记账（迟到结果将被丢弃）"
            )
            drained.append((
                task,
                f"[AsyncEval] hard timeout: eval exceeded {hard_timeout:.0f}s "
                f"in-flight budget, reaped as failure.",
                False,
                {"error": "async_hard_timeout"},
            ))
        for task, eval_log, is_success, metrics in drained:
            iteration = task.iteration
            try:
                # 完整评测日志落盘（worker 只负责跑，文件写盘由主线程做）
                log_path = os.path.join(task.iter_dir, "3_execution_sub0_r0.log")
                with open(log_path, 'w', encoding='utf-8') as f:
                    f.write(eval_log)
                if not isinstance(metrics, dict):
                    metrics = {}
                acc = self.task_profile.extract_primary_metric(metrics)
                self.logger.info(
                    f"📥 [AsyncEval] iter={iteration} 评测完成 acc={acc:.4f} "
                    f"success={is_success}，开始回收记账"
                )
                if is_success:
                    inner_feedback, error_type = None, "static"
                else:
                    inner_feedback = self._extract_execution_error(eval_log)
                    error_type = "execution"
                result = {
                    "iteration": iteration,
                    "success": False,
                    "metrics": {},
                    "feedback": "",
                    "timestamp": datetime.now().isoformat(),
                    "logic_check_passed": True,
                    "logic_issues": [],
                }
                # finalize 内 result["topic"] / topic_generator / candidate_pool
                # 均读 self.explorer.last_used_topic（当前轮的全局态），必须换回
                # 该评测所属轮交出时的 topic
                topic_bak = getattr(self.explorer, 'last_used_topic', None)
                self.explorer.last_used_topic = task.topic
                try:
                    self._finalize_iteration_result(
                        iteration, result, task.strategy, task.current_mode, task.idea,
                        is_success, metrics, task.code, eval_log,
                        True, [], inner_feedback, error_type)
                finally:
                    self.explorer.last_used_topic = topic_bak
                self.logger.info(f"✅ [AsyncEval] 回收记账完成 iter={iteration}")
            except Exception as drain_err:
                self.logger.error(
                    f"❌ [AsyncEval] iter={iteration} 回收记账异常: {drain_err}\n"
                    f"{traceback.format_exc()}"
                )
                # 记账异常也不能丢轮：记一条失败记录，避免 last_iteration 跳过该轮
                try:
                    self.state_manager.update_iteration(iteration, {
                        "iteration": iteration,
                        "success": False,
                        "metrics": {},
                        "feedback": (f"[Iteration {iteration}: Async Drain Exception] "
                                     f"{drain_err}"),
                        "idea": task.idea,
                        "topic": task.topic,
                    })
                    self._update_results_md()
                except Exception:
                    pass
            finally:
                self._cleanup_async_eval_env(iteration)
        if drained:
            # 修正 last_iteration：update_iteration 无条件把它设为被回收轮号，
            # 评测乱序完成时会回退/跳跃。统一重算：仍有在途 → 最小在途轮 - 1
            # （重启后从该轮整体重做）；无在途 → 已完成记录的最大轮号。
            if mgr.inflight_count() > 0:
                self.state_manager.state["last_iteration"] = mgr.inflight_iterations()[0] - 1
            else:
                done = {c.get("iteration", 0)
                        for c in self.state_manager.state.get("completed_iterations", [])}
                if done:
                    self.state_manager.state["last_iteration"] = max(done)
            self.state_manager.save()

    def _async_wait_all(self):
        """[AsyncEval] 等待全部在途评测完成并 drain 记账（选等待而非 kill：
        评测已跑了大半，kill 会浪费已消耗的 GPU 时间）。被中断时退出等待。"""
        if not self._async_eval_enabled():
            return
        mgr = getattr(self, 'async_eval_manager', None)
        if mgr is None or mgr.inflight_count() == 0:
            return
        self.logger.info(
            f"⏳ [AsyncEval] 等待 {mgr.inflight_count()} 个在途评测完成..."
        )
        while mgr.inflight_count() > 0:
            if self._interrupted:
                self.logger.warning("🛑 [AsyncEval] 等待在途评测期间被中断，退出")
                break
            time.sleep(10)
            self._drain_async_evals()
        self._drain_async_evals()
        if mgr.inflight_count() == 0:
            self.logger.info("✅ [AsyncEval] 全部在途评测已回收记账")

    def _produce_lookahead(self, next_iteration: int):
        """[LookAhead] 槽位等待期间预生产下一轮 idea+code（主线程执行，无额外线程）。

        成功时填满 self._lookahead = {"iteration", "strategy", "idea", "code",
        "topic"}；code 为 None 表示只产出了 idea（消费轮走现产代码路径）。
        任何失败仅记日志——预生产是优化，绝不影响正常流程。

        正确性要点：
        - generate_idea 会覆盖 explorer.last_used_topic，结束（含异常）时恢复，
          保证当前轮随后提交评测时 AsyncEvalTask.topic 仍取到本轮 topic；
        - 预生产 coder 会覆盖 target_patch_file，但在途评测全部运行在
          _create_async_eval_env 的快照上，与实时文件完全隔离；
        - strategy 在此预先决策（UCT/Combo 计数已消耗），消费轮 run() 直接复用，
          不会二次决策。
        """
        if self._lookahead is not None:
            return
        if not getattr(self.config, 'enable_lookahead_production', True):
            return
        if not self._async_eval_enabled():
            return
        if next_iteration > int(getattr(self.config, 'max_iterations', 0) or 0):
            return
        iter_dir = os.path.join(self.config.log_dir, f"iter_{next_iteration:03d}")
        os.makedirs(iter_dir, exist_ok=True)
        topic_bak = getattr(self.explorer, 'last_used_topic', None)
        try:
            strategy = self._decide_strategy(next_iteration)
            pkg = {"iteration": next_iteration, "strategy": strategy,
                   "idea": None, "code": None, "topic": None}
            # strategy 已决策就必须缓存（计数不可二次消耗），即使后续产出失败
            self._lookahead = pkg
            self.logger.info(
                f"🔮 [LookAhead] 槽位等待期间预生产 iter={next_iteration} "
                f"（mode={strategy.get('mode', 'breadth')}）...")
            idea = self.explorer.generate_idea(
                next_iteration,
                self.state_manager.state["history_feedback"],
                iter_dir,
                strategy=strategy)
            if not idea:
                self.logger.warning(
                    f"⚠️ [LookAhead] iter={next_iteration} idea 预生成失败，"
                    f"消费轮将现产")
                return
            pkg["idea"] = idea
            # 与 run_iteration 的正常路径一致：idea 落盘（消费轮的 Resume
            # 检测会读到同一文件，行为对齐）
            with open(os.path.join(iter_dir, "1_algorithm_idea.md"),
                      'w', encoding='utf-8') as f:
                f.write(idea)
            code = self._produce_validated_code(next_iteration, idea, iter_dir, strategy)
            if not code:
                self.logger.warning(
                    f"⚠️ [LookAhead] iter={next_iteration} code 预生成未通过静态门禁，"
                    f"消费轮将现产代码")
                return
            pkg["code"] = code
            pkg["topic"] = getattr(self.explorer, 'last_used_topic', None)
            self.logger.info(
                f"✅ [LookAhead] iter={next_iteration} idea+code 已就绪，等待 GPU 槽位")
        except Exception as e:
            self.logger.warning(
                f"⚠️ [LookAhead] 预生产异常（忽略，不影响正常流程）: "
                f"{type(e).__name__}: {e}")
        finally:
            self.explorer.last_used_topic = topic_bak

    def _produce_validated_code(self, iteration: int, idea: str, iter_dir: str,
                                strategy: Dict[str, Any]) -> Optional[str]:
        """[LookAhead] 预生产用：coder 生成 + 全部 pre-eval 静态门禁（不含任何评测）。

        与 _execute_single_idea 内层 repair 循环的静态门禁部分同构（有意不复用：
        该函数的 repair 循环还耦合了评测失败修复逻辑，拆开回归风险高）。
        返回通过门禁的代码；未通过（含 repair 用尽）返回 None。
        """
        max_inner_repairs = getattr(self.config, 'max_inner_repairs', 2)
        inner_feedback = None
        error_type = "static"
        for repair_idx in range(max_inner_repairs + 1):
            if self._interrupted:
                return None
            feedback_for_coder = None if repair_idx == 0 else inner_feedback
            error_type_for_coder = "static" if repair_idx == 0 else error_type
            code = self.coder.generate_code(
                idea, iter_dir,
                error_log=feedback_for_coder,
                repair_mode=(repair_idx > 0),
                strategy=strategy,
                error_type=error_type_for_coder)
            if code is None:
                self.logger.warning("⚠️ [LookAhead] CoderAgent 未能生成任何代码")
                inner_feedback = ("Code generation failed: all strategies failed to "
                                  "generate valid code.")
                error_type = "static"
                continue

            code = self._auto_fix_missing_imports(code)

            # 原子写 target_patch_file + 迭代目录备份（同 _execute_single_idea）
            target_path = self.config.target_patch_file
            temp_target_path = target_path + ".tmp"
            with open(temp_target_path, 'w', encoding='utf-8') as f:
                f.write(code)
            os.replace(temp_target_path, target_path)
            backup_path = os.path.join(iter_dir, f"2_modified_code_sub0_r{repair_idx}.py")
            temp_backup_path = backup_path + ".tmp"
            with open(temp_backup_path, 'w', encoding='utf-8') as f:
                f.write(code)
            os.replace(temp_backup_path, backup_path)

            smoke_ok, smoke_msg = self._static_smoke_test(code)
            if not smoke_ok:
                self.logger.warning(f"⚠️ [LookAhead] 静态 smoke test 未通过: {smoke_msg}")
                inner_feedback = f"Static smoke test failed: {smoke_msg}"
                error_type = "static"
                continue

            # 机制簇闸门：饱和/超配额机制族直接打回修复，不浪费全量评测
            mech_ok, mech_msg = self._mechanism_gate(code)
            if not mech_ok:
                self.logger.warning(f"🚫 [LookAhead] 机制簇闸门拦截: {mech_msg[:120]}")
                inner_feedback = mech_msg
                error_type = "static"
                continue

            if self.fingerprint_executor:
                fp_ok, fp_msg = self.fingerprint_executor.quick_run(
                    self.config.target_patch_file)
                if not fp_ok and ("UNIFORM" in fp_msg or "RETAIN_ALL" in fp_msg):
                    self.logger.warning(f"⚠️ [LookAhead] 指纹预跑拦截陷阱: {fp_msg}")
                    inner_feedback = (
                        f"Fingerprint pre-run interception: {fp_msg}. Please rewrite "
                        f"the core algorithm; uniform step and retain-all fallbacks "
                        f"are forbidden.")
                    error_type = "static"
                    continue

            if getattr(self, 'token_count_probe', None):
                try:
                    tc_ok, tc_msg = self.token_count_probe.verify(
                        self.config.target_patch_file)
                    if not tc_ok:
                        self.logger.warning(f"⚠️ [LookAhead] Token 数探针拦截: {tc_msg}")
                        inner_feedback = f"Token count probe interception: {tc_msg}"
                        error_type = "static"
                        continue
                    # 行为级查重：选择指纹命中历史 = 功能克隆，打回修复
                    clone_msg = self._selection_clone_gate(iteration)
                    if clone_msg:
                        self.logger.warning(f"🚫 [LookAhead] 行为克隆拦截: iter={iteration}")
                        inner_feedback = clone_msg
                        error_type = "static"
                        continue
                except Exception as tc_err:
                    self.logger.warning(
                        f"⚠️ [LookAhead] Token 数探针异常（非致命）: {tc_err}")

            return code
        return None

    def _consume_lookahead(self, iteration: int, pkg: Dict[str, Any], iter_dir: str,
                           result: Dict[str, Any],
                           strategy: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """[LookAhead] 快速路径提交：原子写代码 → 快照环境 → 等槽位 → 提交异步评测。

        成功返回 pending result（与正常 pending 收尾同构）；提交环节失败返回
        None，调用方落入正常现产路径（预生产的 idea 文件会被 Resume 检测复用）。
        """
        code = pkg["code"]
        try:
            # 预生产时已写入 target_patch_file 并通过静态门禁；此处幂等重写一次，
            # 保证快照内容与本函数持有的 code 严格一致
            target_path = self.config.target_patch_file
            temp_target_path = target_path + ".tmp"
            with open(temp_target_path, 'w', encoding='utf-8') as f:
                f.write(code)
            os.replace(temp_target_path, target_path)
            pythonpath_prefix = self._create_async_eval_env(iteration, code)
            self._async_wait_slot(iteration)  # 槽位等待（内含 drain）
            orig_pp = os.environ.get("PYTHONPATH", "")
            pythonpath = pythonpath_prefix + (os.pathsep + orig_pp if orig_pp else "")
            task = AsyncEvalTask(
                iteration=iteration,
                iter_dir=iter_dir,
                pythonpath=pythonpath,
                code=code,
                idea=pkg["idea"],
                strategy=strategy,
                current_mode=strategy.get("mode", "breadth") if strategy else "breadth",
                # 预生产时快照的 topic（explorer.last_used_topic 已在预生产
                # 结束后恢复为上一轮值，此处不能读实时值）
                topic=pkg.get("topic"),
            )
            self.async_eval_manager.submit(task)
            self.logger.info(
                f"📤 [LookAhead] iter={iteration} 使用预生产 idea+code 直接提交评测"
                f"（在途={self.async_eval_manager.inflight_count()}/"
                f"{max(1, int(getattr(self.config, 'max_async_evals', 2) or 2))}）"
            )
            result["eval_pending"] = True
            result["idea"] = pkg["idea"]
            result["topic"] = pkg.get("topic")
            self.logger.info(
                f"⏭️ [AsyncEval] iter={iteration} 以 pending 收尾，立即进入下一轮"
            )
            return result
        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.logger.warning(
                f"⚠️ [LookAhead] 快速路径提交失败，落入现产路径: "
                f"{type(e).__name__}: {e}")
            return None

    def _execute_single_idea(self, iteration: int, idea: str, iter_dir: str, sub_idx: int, strategy: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], Optional[str], str, bool, List, str, str]:
        """执行单个 idea 的内层 execution-repair loop。不修改 self.last_error_log。"""
        max_inner_repairs = getattr(self.config, 'max_inner_repairs', 2)
        inner_success = False
        inner_metrics = {}
        inner_feedback = None
        error_type = "static"
        best_code_in_iter = None
        logic_passed = False
        logic_issues = []
        eval_log = ""

        for repair_idx in range(max_inner_repairs + 1):
            if self._interrupted:
                self.logger.info("🛑 检测到中断请求，终止内层 repair 循环")
                break
            self.logger.info(f"\n{'='*40}")
            self.logger.info(f"🔧 [Inner Loop] 第 {repair_idx + 1}/{max_inner_repairs + 1} 轮代码生成/修复")
            self.logger.info(f"{'='*40}")

            # 首轮生成不携带上一轮错误上下文，避免污染新 idea
            if repair_idx == 0:
                feedback_for_coder = None
                error_type_for_coder = "static"
            else:
                feedback_for_coder = inner_feedback
                error_type_for_coder = error_type

            code = self.coder.generate_code(
                idea, iter_dir,
                error_log=feedback_for_coder,
                repair_mode=(repair_idx > 0),
                strategy=strategy,
                error_type=error_type_for_coder
            )

            if code is None:
                self.logger.error("❌ CoderAgent 未能生成任何代码")
                inner_feedback = "Code generation failed: all strategies failed to generate valid code. Please check LLM API availability or prompt wording."
                error_type = "static"
                continue

            # FIX: always keep the last attempted code so tree update can proceed even on failure
            best_code_in_iter = code

            # 自动修复常见缺失 import（确定性，不调用 LLM）
            code = self._auto_fix_missing_imports(code)

            logic_passed = True
            logic_issues = []

            # 6. 原子性保存代码到目标文件
            target_path = self.config.target_patch_file
            temp_target_path = target_path + ".tmp"
            with open(temp_target_path, 'w', encoding='utf-8') as f:
                f.write(code)
            os.replace(temp_target_path, target_path)

            backup_path = os.path.join(iter_dir, f"2_modified_code_sub{sub_idx}_r{repair_idx}.py")
            temp_backup_path = backup_path + ".tmp"
            with open(temp_backup_path, 'w', encoding='utf-8') as f:
                f.write(code)
            os.replace(temp_backup_path, backup_path)

            # 6.5 静态 smoke test：检查均匀步长回退指纹
            smoke_ok, smoke_msg = self._static_smoke_test(code)
            if not smoke_ok:
                self.logger.warning(f"⚠️ 静态 smoke test 未通过: {smoke_msg}")
                inner_feedback = f"Static smoke test failed: {smoke_msg}"
                error_type = "static"
                continue

            # 机制簇闸门：饱和/超配额机制族直接打回修复，不浪费全量评测
            mech_ok, mech_msg = self._mechanism_gate(code)
            if not mech_ok:
                self.logger.warning(f"🚫 机制簇闸门拦截: {mech_msg[:120]}")
                inner_feedback = mech_msg
                error_type = "static"
                continue

            # 6.6 指纹预跑（F包）
            if self.fingerprint_executor:
                fp_ok, fp_msg = self.fingerprint_executor.quick_run(self.config.target_patch_file)
                if not fp_ok and ("UNIFORM" in fp_msg or "RETAIN_ALL" in fp_msg):
                    self.logger.warning(f"⚠️ 指纹预跑拦截陷阱: {fp_msg}")
                    inner_feedback = f"Fingerprint pre-run interception: {fp_msg}. Please rewrite the core algorithm; uniform step and retain-all fallbacks are forbidden."
                    error_type = "static"
                    continue
                elif not fp_ok:
                    self.logger.warning(f"⚠️ 指纹预跑非致命异常（可能为 mock 环境不足）: {fp_msg}，继续正式评测")

            # 6.7 v12.1: 运行时 token 数探针（防动态超选 hack）
            # iter_461 教训：算法可在运行时静默保留远超 target 的 token 虚增 accuracy，
            # 静态 tampered-ratio 审计抓不到。评测前用合成输入实测保留数。
            if getattr(self, 'token_count_probe', None):
                try:
                    tc_ok, tc_msg = self.token_count_probe.verify(self.config.target_patch_file)
                    if not tc_ok:
                        self.logger.warning(f"⚠️ Token 数探针拦截: {tc_msg}")
                        if "OVER_SELECTION" in tc_msg:
                            inner_feedback = (
                                f"Token count probe interception: {tc_msg} "
                                f"Your algorithm MUST select exactly target_vision_tokens tokens. "
                                f"Common cause: clustering/merging loops that terminate with more "
                                f"clusters than requested (e.g., lazy-heap exhaustion), each "
                                f"contributing one representative. Ensure the final selection is "
                                f"hard-truncated to target_vision_tokens."
                            )
                        else:
                            # 维度不匹配等其他拦截：直接透传探针诊断作为修复反馈
                            inner_feedback = f"Token count probe interception: {tc_msg}"
                        error_type = "static"
                        continue
                    else:
                        self.logger.info(f"🔢 Token 数探针通过: {tc_msg}")
                        # 行为级查重：选择指纹命中历史 = 功能克隆，打回修复
                        clone_msg = self._selection_clone_gate(iteration)
                        if clone_msg:
                            self.logger.warning(f"🚫 行为克隆拦截: iter={iteration}")
                            inner_feedback = clone_msg
                            error_type = "static"
                            continue
                except Exception as tc_err:
                    self.logger.warning(f"⚠️ Token 数探针异常（非致命，继续评测）: {tc_err}")

            # 6.8 [SmokeMode] GPU 显存闸 + 真实 GPU 冒烟：通过即跳过正式评测进下一轮。
            # 冒烟失败按现有 pre-eval 失败路径处理（进修复循环）。
            if getattr(self.config, 'smoke_only_mode', False):
                smoke_ok, smoke_msg = self._gpu_smoke_check(self.config.target_patch_file)
                smoke_log_path = os.path.join(iter_dir, f"3_smoke_sub{sub_idx}_r{repair_idx}.log")
                with open(smoke_log_path, 'w', encoding='utf-8') as f:
                    f.write(smoke_msg)
                if not smoke_ok:
                    self.logger.warning(f"⚠️ [SmokeMode] GPU 冒烟未通过: {smoke_msg[:500]}")
                    inner_feedback = f"GPU smoke test failed: {smoke_msg}"
                    error_type = "static"
                    continue
                self.logger.info("✅ [SmokeMode] 冒烟通过，跳过正式评测")
                inner_success = True
                # inner_metrics 必须是 dict（finalize 依赖）；smoke_passed 标记驱动
                # _finalize_iteration_result 的 smoke 记账分支
                inner_metrics = {"smoke_passed": True}
                best_code_in_iter = code
                eval_log = f"[SmokeMode] GPU smoke passed, formal evaluation skipped. {smoke_msg}"
                # 沿用 3_execution 文件名保存，保持迭代目录结构一致
                log_path = os.path.join(iter_dir, f"3_execution_sub{sub_idx}_r{repair_idx}.log")
                with open(log_path, 'w', encoding='utf-8') as f:
                    f.write(eval_log)
                break

            # 6.9 [AsyncEval] 异步评测：快照环境 + 交出后台，本轮以 pending 立即返回。
            # 快照隔离是硬性正确性要求（在途评测子进程 import selector 的时刻可能
            # 晚于下一轮 coder 覆盖 target_patch_file）；交出失败回退同步评测。
            if self._async_eval_enabled():
                try:
                    pythonpath_prefix = self._create_async_eval_env(iteration, code)
                    self._async_wait_slot(iteration)  # 槽位等待（内含 drain）
                    orig_pp = os.environ.get("PYTHONPATH", "")
                    pythonpath = pythonpath_prefix + (os.pathsep + orig_pp if orig_pp else "")
                    task = AsyncEvalTask(
                        iteration=iteration,
                        iter_dir=iter_dir,
                        pythonpath=pythonpath,
                        code=code,
                        idea=idea,
                        strategy=strategy,
                        current_mode=strategy.get("mode", "breadth") if strategy else "breadth",
                        topic=getattr(self.explorer, 'last_used_topic', None),
                    )
                    self.async_eval_manager.submit(task)
                    self.logger.info(
                        f"📤 [AsyncEval] iter={iteration} 评测已交出"
                        f"（在途={self.async_eval_manager.inflight_count()}/"
                        f"{max(1, int(getattr(self.config, 'max_async_evals', 2) or 2))}）"
                    )
                    # eval_pending 标记：run_iteration 据此以 pending 收尾（不记账）
                    return False, {"eval_pending": True}, code, "", True, [], None, "async_pending"
                except KeyboardInterrupt:
                    raise
                except Exception as async_err:
                    self.logger.warning(
                        f"⚠️ [AsyncEval] 异步交出失败，回退同步评测: "
                        f"{type(async_err).__name__}: {async_err}"
                    )

            # 7. 运行评测
            eval_log, is_success, metrics = self.executor.run_evaluation(iteration)

            # 8. 保存日志
            log_path = os.path.join(iter_dir, f"3_execution_sub{sub_idx}_r{repair_idx}.log")
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(eval_log)

            if is_success:
                # 【新增】accuracy=0 意味着算法在（部分）样本上崩溃/未生效（如 selected_indices 未绑定）。
                # 本质是代码 bug，给一次针对性修复机会（仅 repair_idx==0，避免死循环）。
                # 主指标键取当前任务 profile（默认 accuracy，取值不变；缺键回落 accuracy）
                from autoresearch_tasks import current_task
                parsed_acc = metrics.get(current_task().primary_metric, metrics.get('accuracy', -1))
                if (getattr(self.config, 'enable_zero_metric_crash_detection', True)
                        and parsed_acc == 0.0 and repair_idx == 0):
                    err_excerpt = self._extract_execution_error(eval_log)[:1500]
                    self.logger.warning("⚠️ accuracy=0，判定为算法崩溃/未生效，给予一次修复机会...")
                    inner_feedback = (
                        "The evaluation finished but accuracy=0.0000, which means the algorithm CRASHED or "
                        "produced invalid selected_indices on (some) samples instead of falling back silently. "
                        f"Error excerpt from the evaluation log:\n{err_excerpt}\n"
                        "Fix the boundary/branch logic so that selected_indices is ALWAYS defined and valid "
                        "(torch.long, correct device, ascending order) on every code path."
                    )
                    error_type = "execution"
                    continue
                inner_success = True
                inner_metrics = metrics
                best_code_in_iter = code
                self.logger.info(f"✅ 内层循环第 {repair_idx + 1} 轮执行成功！")
                break

            # FIX: 评测超时 —— 给一次针对性的“降复杂度”修复机会（repair_idx==0），再次超时才放弃。
            # 历史统计中超时是失败最大头且全部直接放弃；feedback 必须明确说明“是算法太慢”，
            # 否则下一轮修复不知道要修什么。
            if metrics.get("error") == "timeout":
                if repair_idx == 0:
                    self.logger.warning(f"⏱️ 评测超时，给予一次降复杂度修复机会...")
                    inner_feedback = (
                        "Evaluation TIMEOUT: the previous code exceeded the wall-clock limit, i.e. the algorithm is TOO SLOW, "
                        "not incorrect. You MUST reduce its runtime complexity: remove ALL per-iteration GPU kernel calls "
                        "(torch.linalg.*, torch.cdist, .item(), .tolist()) from Python for/while loops — vectorize them or "
                        "precompute outside the loop; avoid materializing [N, N, D] intermediates; for large N, project "
                        "features to <=256 dims before pairwise computations. If the algorithm cannot be made fast enough, "
                        "simplify it while preserving its core idea. Keep changes inside the ALGORITHM CORE block."
                    )
                    error_type = "timeout"
                    continue  # 进入下一轮内层修复
                self.logger.warning(f"⏱️ 评测再次超时，终止本轮内层循环，进入下一轮 iteration")
                inner_feedback = "Evaluation timeout: exceeded configured time limit. Skipping to next iteration."
                error_type = "timeout"
                break

            inner_feedback = self._extract_execution_error(eval_log)
            error_type = "execution"

            # 【新增】环境性 OOM（其他进程抢占显存）：不进 LLM 修复，换卡重跑一次评测；
            # 仍失败则按环境错误终止内层循环（LLM 修复无法解决资源争抢，白烧 token）。
            if self._is_contention_oom(inner_feedback):
                self.logger.warning(
                    "🎮 检测到环境性 OOM（显存被其他进程占用），更换 GPU 重跑一次评测，不进入 LLM 修复"
                )
                retry_log, retry_success, retry_metrics = self.executor.run_evaluation(iteration)
                retry_log_path = os.path.join(iter_dir, f"3_execution_sub{sub_idx}_r{repair_idx}_retry_gpu.log")
                try:
                    with open(retry_log_path, 'w', encoding='utf-8') as f:
                        f.write(retry_log)
                except Exception:
                    pass
                if retry_success:
                    inner_success = True
                    inner_metrics = retry_metrics
                    best_code_in_iter = code
                    self.logger.info("✅ 更换 GPU 重跑成功！")
                    break
                self.logger.error("🚨 更换 GPU 重跑仍失败，按环境错误终止本轮内层循环")
                inner_feedback = self._extract_execution_error(retry_log)
                error_type = "environment"
                break

            # 【新增】检测环境依赖类错误，避免 Coder Agent 做无效修复
            if self._is_environment_error(inner_feedback):
                self.logger.error(
                    f"🚨 检测到环境依赖错误（ImportError / ModuleNotFoundError / 版本不兼容），"
                    f"代码修复无法解决此问题，终止本轮内层循环。请手动检查环境配置。\n"
                    f"错误摘要: {inner_feedback[:600]}"
                )
                break

            self.logger.warning(f"⚠️ 第 {repair_idx + 1} 轮执行失败，准备进入下一轮修复...")

        return inner_success, inner_metrics, best_code_in_iter, eval_log, logic_passed, logic_issues, inner_feedback, error_type

    def _parallel_eligible_gpus(self) -> List[int]:
        """返回剩余显存 ≥ parallel_branch_min_free_gb 的 GPU 索引列表（按剩余显存降序）。

        探测失败或 parallel_branches 关闭时返回空列表（调用方据此回落单 idea 路径）。
        """
        if int(getattr(self.config, 'parallel_branches', 0) or 0) <= 0:
            return []
        threshold_mib = float(getattr(self.config, 'parallel_branch_min_free_gb', 30.0) or 0.0) * 1024
        try:
            free_gpus = SafeExecutor._query_free_gpus()
        except Exception as e:
            self.logger.warning(f"⚠️ [Parallel] GPU 探测失败: {e}")
            return []
        return [idx for idx, free in free_gpus if free >= threshold_mib]

    def _produce_branch_code(self, iteration: int, idea: str, iter_dir: str, sub_idx: int,
                             strategy: Dict[str, Any]) -> Optional[str]:
        """为分支 sub_idx 生产候选代码（仅 pre-eval 阶段：生成 + 静态检查 + 原子写分支文件）。

        复用 _execute_single_idea 的候选代码生产逻辑，区别：
        - 目标文件为分支影子包中的 patch 文件（self._branch_patch_files[sub_idx]）；
        - 不含评测步骤；inner repair 循环仅覆盖静态错误的修复。
        全部 repair 轮失败时返回 None（该分支跳过）。
        """
        max_inner_repairs = getattr(self.config, 'max_inner_repairs', 2)
        inner_feedback = None
        error_type = "static"
        target_path = self._branch_patch_files[sub_idx]

        for repair_idx in range(max_inner_repairs + 1):
            if self._interrupted:
                self.logger.info("🛑 [Parallel] 检测到中断请求，终止分支代码生产")
                return None
            self.logger.info(f"\n{'='*40}")
            self.logger.info(f"🌿 [Parallel] 分支 {sub_idx} 第 {repair_idx + 1}/{max_inner_repairs + 1} 轮代码生成/修复")
            self.logger.info(f"{'='*40}")

            if repair_idx == 0:
                feedback_for_coder = None
                error_type_for_coder = "static"
            else:
                feedback_for_coder = inner_feedback
                error_type_for_coder = error_type

            code = self.coder.generate_code(
                idea, iter_dir,
                error_log=feedback_for_coder,
                repair_mode=(repair_idx > 0),
                strategy=strategy,
                error_type=error_type_for_coder
            )

            if code is None:
                self.logger.error(f"❌ [Parallel] 分支 {sub_idx} CoderAgent 未能生成任何代码")
                inner_feedback = "Code generation failed: all strategies failed to generate valid code. Please check LLM API availability or prompt wording."
                error_type = "static"
                continue

            # 自动修复常见缺失 import（确定性，不调用 LLM）
            code = self._auto_fix_missing_imports(code)

            # 原子性保存代码到分支目标文件
            temp_target_path = target_path + ".tmp"
            with open(temp_target_path, 'w', encoding='utf-8') as f:
                f.write(code)
            os.replace(temp_target_path, target_path)

            backup_path = os.path.join(iter_dir, f"2_modified_code_sub{sub_idx}_r{repair_idx}.py")
            temp_backup_path = backup_path + ".tmp"
            with open(temp_backup_path, 'w', encoding='utf-8') as f:
                f.write(code)
            os.replace(temp_backup_path, backup_path)

            # 静态 smoke test：检查均匀步长回退指纹
            smoke_ok, smoke_msg = self._static_smoke_test(code)
            if not smoke_ok:
                self.logger.warning(f"⚠️ [Parallel] 分支 {sub_idx} 静态 smoke test 未通过: {smoke_msg}")
                inner_feedback = f"Static smoke test failed: {smoke_msg}"
                error_type = "static"
                continue

            # 机制簇闸门：饱和/超配额机制族直接打回修复，不浪费全量评测
            mech_ok, mech_msg = self._mechanism_gate(code)
            if not mech_ok:
                self.logger.warning(f"🚫 [Parallel] 分支 {sub_idx} 机制簇闸门拦截: {mech_msg[:120]}")
                inner_feedback = mech_msg
                error_type = "static"
                continue

            # 指纹预跑（分支文件路径）
            if self.fingerprint_executor:
                fp_ok, fp_msg = self.fingerprint_executor.quick_run(target_path)
                if not fp_ok and ("UNIFORM" in fp_msg or "RETAIN_ALL" in fp_msg):
                    self.logger.warning(f"⚠️ [Parallel] 分支 {sub_idx} 指纹预跑拦截陷阱: {fp_msg}")
                    inner_feedback = f"Fingerprint pre-run interception: {fp_msg}. Please rewrite the core algorithm; uniform step and retain-all fallbacks are forbidden."
                    error_type = "static"
                    continue
                elif not fp_ok:
                    self.logger.warning(f"⚠️ [Parallel] 分支 {sub_idx} 指纹预跑非致命异常（可能为 mock 环境不足）: {fp_msg}，继续正式评测")

            # 运行时 token 数探针（分支文件路径，防动态超选 hack）
            if getattr(self, 'token_count_probe', None):
                try:
                    tc_ok, tc_msg = self.token_count_probe.verify(target_path)
                    if not tc_ok:
                        self.logger.warning(f"⚠️ [Parallel] 分支 {sub_idx} Token 数探针拦截: {tc_msg}")
                        if "OVER_SELECTION" in tc_msg:
                            inner_feedback = (
                                f"Token count probe interception: {tc_msg} "
                                f"Your algorithm MUST select exactly target_vision_tokens tokens. "
                                f"Common cause: clustering/merging loops that terminate with more "
                                f"clusters than requested (e.g., lazy-heap exhaustion), each "
                                f"contributing one representative. Ensure the final selection is "
                                f"hard-truncated to target_vision_tokens."
                            )
                        else:
                            # 维度不匹配等其他拦截：直接透传探针诊断作为修复反馈
                            inner_feedback = f"Token count probe interception: {tc_msg}"
                        error_type = "static"
                        continue
                    else:
                        self.logger.info(f"🔢 [Parallel] 分支 {sub_idx} Token 数探针通过: {tc_msg}")
                        # 行为级查重：选择指纹命中历史 = 功能克隆，打回修复
                        clone_msg = self._selection_clone_gate(iteration)
                        if clone_msg:
                            self.logger.warning(
                                f"🚫 [Parallel] 分支 {sub_idx} 行为克隆拦截: iter={iteration}")
                            inner_feedback = clone_msg
                            error_type = "static"
                            continue
                except Exception as tc_err:
                    self.logger.warning(f"⚠️ [Parallel] 分支 {sub_idx} Token 数探针异常（非致命，继续评测）: {tc_err}")

            # [SmokeMode] 分支 GPU 显存闸 + 冒烟（串行；两分支可共用一张满足条件的卡，
            # 无需绑不同卡）。冒烟失败按 pre-eval 失败路径进修复循环。
            if getattr(self.config, 'smoke_only_mode', False):
                smoke_ok, smoke_msg = self._gpu_smoke_check(target_path)
                if not hasattr(self, '_branch_smoke_info'):
                    self._branch_smoke_info = {}
                if not smoke_ok:
                    self.logger.warning(f"⚠️ [SmokeMode] 分支 {sub_idx} 冒烟结果: 失败 - {smoke_msg[:300]}")
                    inner_feedback = f"GPU smoke test failed: {smoke_msg}"
                    error_type = "static"
                    continue
                self._branch_smoke_info[sub_idx] = smoke_msg
                self.logger.info(f"🌿 [SmokeMode] 分支 {sub_idx} 冒烟结果: 通过 - {smoke_msg[:200]}")

            return code

        return None

    # ------------------------------------------------------------------
    # Phase 3: 组合深耕（combo deepen）—— 高 success-rate (combo, archetype) 对
    # 标记为 objective_deepen 迭代：全程 breadth 模式，无父本；由 BreadthAgent
    # 对该对的优化目标本身做深度推理演进（见 agents/breadth.py）。
    # ------------------------------------------------------------------
    def _maybe_mark_objective_deepen(self, strategy: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Phase 3: 组合深耕标记 —— strategy 的 (tensor_combo, combo_archetype)
        命中 ComboAllocator 深耕集合时，复制 strategy 并打 objective_deepen=True
        （mode 保持 breadth，不钉父本、无 parent_iteration）。
        enable_combo_deepen 关闭 / 组合探索关闭 / 非深耕对时原样返回。
        """
        if not strategy:
            return strategy
        if not getattr(self.config, 'enable_combo_deepen', False):
            return strategy
        alloc = getattr(self, 'combo_allocator', None)
        if alloc is None:
            return strategy
        combo = strategy.get("tensor_combo")
        if not combo:
            return strategy
        arch = strategy.get("combo_archetype") or ""
        if not alloc.is_depth_pair(combo, arch):
            return strategy
        new_strategy = dict(strategy)
        new_strategy["objective_deepen"] = True
        self.logger.info(
            f"🔬 [Engine] Phase3 组合深耕: {combo}/{arch} 标记 objective_deepen，"
            f"本轮对该方向的优化目标做深度推理（breadth 模式，无父本）"
        )
        return new_strategy

    def _record_combo_reward(self, strategy: Optional[Dict[str, Any]], acc: float):
        """combo reward 统一记账。正常路径由 _finalize_iteration_result 记录；
        本方法用于未走到 finalize 的提前返回路径（agent 失败/异常/分支跳过），
        防止 ComboAllocator pending 泄漏与 trial 口径不一致（失败 == acc 0.0 未超基线）。
        """
        alloc = getattr(self, 'combo_allocator', None)
        if alloc is None or not (strategy or {}).get("tensor_combo"):
            return
        alloc.record_reward(
            strategy["tensor_combo"], float(acc or 0.0), self.MMTOK_BASELINE,
            arch=(strategy or {}).get("combo_archetype") or "")

    @staticmethod
    def _is_synthetic_topic(topic) -> bool:
        """合成 topic 标签（objective_deepen 轮次 / 历史 THEORY_DEEPEN 伪 topic），
        不得进入 good_topics 池（会被当召回检索查询复用，污染召回）。"""
        t = str(topic or "")
        return t.startswith("objective_deepen:") or t.startswith("THEORY_DEEPEN")

    def _execute_iteration_parallel_branches(self, iteration: int, strategy: Dict[str, Any],
                                             iter_dir: str, result: Dict[str, Any],
                                             gpus: List[int]) -> Dict[str, Any]:
        """多 GPU 分支并行评测：B 个分支各自生成 idea + 代码，并行评测，取 acc 最高分支收尾。

        - 生成阶段串行（Explorer/Coder 有大量共享状态，不并行）；inner repair 仅限 pre-eval 静态阶段；
        - 评测阶段用 ThreadPoolExecutor 并行，每分支绑定一张合格 GPU + 独立 PYTHONPATH 影子包；
        - 记录阶段串行：所有分支记录 combo reward（最佳分支由 _finalize_iteration_result 现有路径记录，
          避免双计），acc 最高分支走现有三 baseline 校验 / best_result 更新 / update_idea_history /
          _update_results_md（accuracy=0 崩溃修复与 tampered 审计也仅对最佳分支生效）。
        """
        B = len(gpus)
        self.logger.info(f"🌿 [Parallel] 启用 {B} 个 GPU 分支并行评测 (GPUs: {gpus})")
        result["parallel_branches_used"] = B
        self._branch_smoke_info = {}  # [SmokeMode] 每轮重置分支冒烟信息（非 smoke 模式闲置）

        # 1. 分支策略：branch 0 用 decide_strategy 的 strategy 原样；
        #    branch b>0 重新抽张量组合（抽不中则去掉 tensor_combo/combo_archetype 键）
        strategies: List[Optional[Dict[str, Any]]] = []
        for b in range(B):
            if b == 0:
                strategy_b = strategy
            else:
                strategy_b = dict(strategy) if strategy else {}
                if getattr(self, 'combo_allocator', None) is not None:
                    combo = self.combo_allocator.select_combo(iteration)
                    if combo:
                        strategy_b["tensor_combo"] = combo
                        strategy_b["combo_archetype"] = self.combo_allocator.current_archetype(combo)
                    else:
                        strategy_b.pop("tensor_combo", None)
                        strategy_b.pop("combo_archetype", None)
            # Phase 3: 组合深耕标记（与 run 循环接线点一致；非深耕对原样返回；
            # 全程 breadth 模式，无父本）
            strategy_b = self._maybe_mark_objective_deepen(strategy_b)
            strategies.append(strategy_b)

        # 2. 串行生成：每分支独立 idea + 候选代码（写到分支文件）
        branch_ideas: List[Optional[str]] = [None] * B
        branch_codes: List[Optional[str]] = [None] * B
        for b in range(B):
            if self._interrupted:
                self.logger.info("🛑 [Parallel] 检测到中断请求，停止后续分支生成")
                break
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"🌿 [Parallel] 分支 {b}/{B} 生成 idea")
            self.logger.info(f"{'='*60}")
            # 深耕已改为 breadth 内 objective_deepen 生成路径，deepen_mode 恒为 False
            self.coder.deepen_mode = False
            self.explorer.history_tracker.deepen_mode = False
            idea_b = self.explorer.generate_idea(
                iteration,
                self.state_manager.state["history_feedback"],
                iter_dir,
                strategy=strategies[b]
            )
            if not idea_b:
                self.logger.warning(f"⚠️ [Parallel] 分支 {b} idea 生成失败，跳过该分支")
                self._record_combo_reward(strategies[b], 0.0)
                continue
            idea_path = os.path.join(iter_dir, f"1_algorithm_idea_sub{b}.md")
            with open(idea_path, 'w', encoding='utf-8') as f:
                f.write(idea_b)
            branch_ideas[b] = idea_b

            code_b = self._produce_branch_code(iteration, idea_b, iter_dir, b, strategies[b])
            if code_b is None:
                self.logger.warning(f"⚠️ [Parallel] 分支 {b} 代码生产失败，跳过该分支")
                self._record_combo_reward(strategies[b], 0.0)
                continue
            branch_codes[b] = code_b

        active = [b for b in range(B) if branch_codes[b] is not None]
        if not active:
            self.logger.error("❌ [Parallel] 所有分支均未能产出候选代码")
            # 所有已选组合记失败（acc=0），清除 pending，保持 trial 口径一致
            for _s in strategies:
                self._record_combo_reward(_s, 0.0)
            failure_mode = self._classify_agent_failure(strategy)
            result["agent_failure_mode"] = failure_mode
            result["feedback"] = (
                f"[Iteration {iteration}: Agent Failure] No branch produced valid candidate code. "
                f"Failure mode: {failure_mode}"
            )
            result["topic"] = getattr(self.explorer, 'last_used_topic', None)
            self.state_manager.update_iteration(iteration, result)
            self._update_results_md()
            return result

        # [SmokeMode] 冒烟模式：不再 fan-out 评测线程（_eval_one 不调用）。
        # 所有 active 分支已在 _produce_branch_code 内串行通过 GPU 冒烟，
        # 直接按 smoke 规则选主分支收尾；idea/代码双分支生成能力保持不变。
        if getattr(self.config, 'smoke_only_mode', False):
            return self._finish_parallel_smoke(
                iteration, strategy, iter_dir, result, gpus,
                strategies, branch_ideas, branch_codes, active)

        # 3. 并行评测：每分支绑定一张合格 GPU，PYTHONPATH 前置分支影子包
        from concurrent.futures import ThreadPoolExecutor
        orig_pythonpath = os.environ.get("PYTHONPATH", "")
        eval_results: Dict[int, Tuple[str, bool, Dict[str, Any]]] = {}

        def _eval_one(b: int):
            branch_env = self._branch_envs[b]
            pythonpath = branch_env + (os.pathsep + orig_pythonpath if orig_pythonpath else "")
            eval_log_b, success_b, metrics_b = self.executor.run_evaluation(
                iteration,
                cuda_dev=str(gpus[b]),
                eval_cmd_override=self.config.eval_cmd,
                extra_env={"PYTHONPATH": pythonpath})
            return b, eval_log_b, success_b, metrics_b

        self.logger.info(f"🚀 [Parallel] {len(active)} 个分支并行评测开始 (active branches: {active})")
        with ThreadPoolExecutor(max_workers=len(active)) as pool:
            for b, eval_log_b, success_b, metrics_b in pool.map(_eval_one, active):
                eval_results[b] = (eval_log_b, success_b, metrics_b)

        # 4. 保存评测日志（3_execution_sub{b}_r0.log）
        for b in active:
            eval_log_b, _, _ = eval_results[b]
            log_path = os.path.join(iter_dir, f"3_execution_sub{b}_r0.log")
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(eval_log_b)

        # 5. 串行记录：选 acc 最高的分支（acc 口径 extract_primary_metric）
        branch_results = []
        best_b, best_acc = None, None
        for b in active:
            _, success_b, metrics_b = eval_results[b]
            acc_b = self.task_profile.extract_primary_metric(metrics_b)
            branch_results.append({
                "branch": b,
                "combo": (strategies[b] or {}).get("tensor_combo"),
                "combo_archetype": (strategies[b] or {}).get("combo_archetype"),
                "acc": acc_b,
                "success": success_b,
            })
            self.logger.info(
                f"🌿 [Parallel] 分支 {b} 结果: success={success_b}, acc={acc_b:.4f}, "
                f"combo={(strategies[b] or {}).get('tensor_combo')}")
            if best_acc is None or acc_b > best_acc:
                best_b, best_acc = b, acc_b
        result["branch_results"] = branch_results

        # 组合 bandit 反馈：非最佳分支在此记录；最佳分支由 _finalize_iteration_result
        # 现有路径记录（strategy 含 tensor_combo 时），避免双计
        _combo_alloc = getattr(self, 'combo_allocator', None)
        if _combo_alloc is not None:
            for entry in branch_results:
                if entry["branch"] == best_b:
                    continue
                if entry["combo"]:
                    _combo_alloc.record_reward(
                        entry["combo"], entry["acc"], self.MMTOK_BASELINE,
                        arch=entry.get("combo_archetype") or "")

        self.logger.info(f"🏆 [Parallel] 最佳分支: {best_b} (acc={best_acc:.4f})，走标准收尾流程")

        # 6. 用最佳分支数据走现有收尾流程（三 baseline 校验 / best_result / history / results.md）
        eval_log_best, success_best, metrics_best = eval_results[best_b]
        if success_best:
            inner_feedback_best = None
            error_type_best = "static"
        else:
            inner_feedback_best = self._extract_execution_error(eval_log_best)
            error_type_best = "execution"
        strategy_best = strategies[best_b]
        current_mode_best = strategy_best.get("mode", "breadth") if strategy_best else "breadth"
        final_result = self._finalize_iteration_result(
            iteration, result, strategy_best, current_mode_best, branch_ideas[best_b],
            success_best, metrics_best, branch_codes[best_b], eval_log_best,
            True, [], inner_feedback_best, error_type_best)

        # 7. 每个分支都是独立实验：非最佳分支也逐一记录（一个算法一条记录）。
        # 记录内容：自己的 metrics/success/idea/combo + history + state；
        # tampered 审计与 accuracy=0 崩溃修复仍仅对最佳分支（finalize 内已处理）。
        has_rr = getattr(self.config, 'has_retain_ratio', True)
        for b in active:
            if b == best_b:
                continue
            eval_log_b, success_b, metrics_b = eval_results[b]
            acc_b = self.task_profile.extract_primary_metric(metrics_b)
            combo_b = (strategies[b] or {}).get("tensor_combo")
            result_b = {
                "iteration": iteration,
                "branch": b,
                "success": success_b,
                "metrics": metrics_b,
                "idea": branch_ideas[b],
                "topic": getattr(self.explorer, 'last_used_topic', None),
                "feedback": (f"[Iteration {iteration} branch {b}] "
                             f"combo={combo_b} acc={acc_b:.4f} success={success_b}"),
            }
            if combo_b:
                result_b["tensor_combo"] = combo_b
            result_b["is_fake_success"] = (
                has_rr and success_b
                and abs(acc_b - self.ORIGINAL_BASELINE) < self.FAKE_SUCCESS_TOLERANCE)
            result_b["is_uniform_fallback"] = (
                success_b
                and abs(acc_b - self.UNIFORM_STEP_BASELINE) < self.FAKE_SUCCESS_TOLERANCE)
            try:
                self.explorer.update_idea_history(
                    branch_ideas[b], result_b, iteration, code=branch_codes[b],
                    static_smoke_test=self._static_smoke_test,
                    fallback_detector=getattr(self.coder, 'fallback_detector', None),
                    api_validator=getattr(self.coder, 'api_validator', None),
                    strategy=strategies[b],
                )
            except Exception as hist_err:
                self.logger.warning(f"⚠️ [Parallel] 分支 {b} 历史记录失败（不影响主流程）: {hist_err}")
            self.state_manager.update_iteration(iteration, result_b)
            self.logger.info(f"📝 [Parallel] 分支 {b} 已记录为独立实验 (acc={acc_b:.4f})")
        self._update_results_md()
        return final_result

    def _finish_parallel_smoke(self, iteration: int, strategy: Dict[str, Any],
                               iter_dir: str, result: Dict[str, Any], gpus: List[int],
                               strategies: List[Optional[Dict[str, Any]]],
                               branch_ideas: List[Optional[str]],
                               branch_codes: List[Optional[str]],
                               active: List[int]) -> Dict[str, Any]:
        """[SmokeMode] 并行分支路径的冒烟收尾（不 fan-out 评测线程）。

        所有 active 分支已在 _produce_branch_code 内串行通过 GPU 冒烟（可共用一卡）。
        主分支选择：冒烟通过优先（active 分支全部通过）、其次 cell_mismatch 较少
        （冒烟日志含 cell_mismatch=N 标记时）、再其次分支编号最小；
        主分支走 _finalize_iteration_result 的 smoke 记账分支，其余分支照常单独记录。
        """
        smoke_info = getattr(self, '_branch_smoke_info', {}) or {}

        # 1. 保存冒烟日志（沿用 3_execution 文件名，保持迭代目录结构一致）
        for b in active:
            log_path = os.path.join(iter_dir, f"3_execution_sub{b}_r0.log")
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(f"[SmokeMode] branch {b} GPU smoke passed, formal evaluation skipped. "
                        f"{smoke_info.get(b, '')}")

        # 2. 主分支选择
        def _cell_mismatch(b: int) -> int:
            m = re.search(r'cell_mismatch=(\d+)', smoke_info.get(b, "") or "")
            return int(m.group(1)) if m else 0

        for b in active:
            self.logger.info(
                f"🌿 [SmokeMode] 分支 {b} 冒烟结果: 通过 (cell_mismatch={_cell_mismatch(b)}, "
                f"combo={(strategies[b] or {}).get('tensor_combo')})")
        best_b = min(active, key=lambda b: (_cell_mismatch(b), b))
        self.logger.info(f"🏆 [SmokeMode] 主分支: {best_b}，走冒烟记账收尾流程")

        branch_results = []
        for b in active:
            branch_results.append({
                "branch": b,
                "combo": (strategies[b] or {}).get("tensor_combo"),
                "combo_archetype": (strategies[b] or {}).get("combo_archetype"),
                "acc": None,  # smoke 模式无 acc
                "success": True,
                "smoke_passed": True,
            })
        result["branch_results"] = branch_results

        # 组合 bandit 反馈：非最佳分支在此记录（新配置组合探索已关闭，strategy 无
        # tensor_combo 时该块自然跳过；acc 缺省按 0.0，口径与失败记账一致）
        _combo_alloc = getattr(self, 'combo_allocator', None)
        if _combo_alloc is not None:
            for entry in branch_results:
                if entry["branch"] == best_b:
                    continue
                if entry["combo"]:
                    _combo_alloc.record_reward(
                        entry["combo"], 0.0, self.MMTOK_BASELINE,
                        arch=entry.get("combo_archetype") or "")

        # 3. 主分支走 _finalize_iteration_result 的 smoke 记账分支
        strategy_best = strategies[best_b]
        current_mode_best = strategy_best.get("mode", "breadth") if strategy_best else "breadth"
        final_result = self._finalize_iteration_result(
            iteration, result, strategy_best, current_mode_best, branch_ideas[best_b],
            True, {"smoke_passed": True}, branch_codes[best_b],
            f"[SmokeMode] branch {best_b} GPU smoke passed, formal evaluation skipped. "
            f"{smoke_info.get(best_b, '')}",
            True, [], None, "static")

        # 4. 非最佳分支照常单独记录（一个算法一条记录；树 backprop 由
        #    _update_idea_history_maybe_skip_tree 在 smoke 模式下跳过）
        for b in active:
            if b == best_b:
                continue
            combo_b = (strategies[b] or {}).get("tensor_combo")
            result_b = {
                "iteration": iteration,
                "branch": b,
                "success": True,
                "smoke_passed": True,
                "metrics": {"smoke_passed": True},
                "idea": branch_ideas[b],
                "topic": getattr(self.explorer, 'last_used_topic', None),
                "feedback": (f"[Iteration {iteration} branch {b}: Smoke Passed] "
                             f"combo={combo_b} GPU smoke passed; formal evaluation skipped."),
            }
            if combo_b:
                result_b["tensor_combo"] = combo_b
            try:
                self._update_idea_history_maybe_skip_tree(
                    branch_ideas[b], result_b, iteration, code=branch_codes[b],
                    strategy=strategies[b])
            except Exception as hist_err:
                self.logger.warning(f"⚠️ [SmokeMode] 分支 {b} 历史记录失败（不影响主流程）: {hist_err}")
            self.state_manager.update_iteration(iteration, result_b)
            self.logger.info(f"📝 [SmokeMode] 分支 {b} 已记录为独立实验（冒烟通过，acc=N/A）")
        self._update_results_md()
        return final_result

    def _finalize_iteration_result(self, iteration: int, result: Dict[str, Any],
                                   strategy: Optional[Dict[str, Any]], current_mode: str,
                                   best_idea: str, inner_success: bool,
                                   inner_metrics: Dict[str, Any], best_code_in_iter: Optional[str],
                                   eval_log: str, logic_passed: bool, logic_issues: List,
                                   inner_feedback: Optional[str], error_type: str) -> Dict[str, Any]:
        """迭代收尾：acc 提取、三 baseline 校验、best_result 更新、历史记录、状态持久化。

        从 run_iteration 单 idea 路径原样抽取（逐字不变），供
        _execute_iteration_parallel_branches 多分支路径对最佳分支复用。
        """
        acc = self.task_profile.extract_primary_metric(inner_metrics)
        has_retain_ratio = getattr(self.config, 'has_retain_ratio', True)
        is_fake_sub = has_retain_ratio and abs(acc - self.ORIGINAL_BASELINE) < self.FAKE_SUCCESS_TOLERANCE
        is_uniform_sub = abs(acc - self.UNIFORM_STEP_BASELINE) < self.FAKE_SUCCESS_TOLERANCE

        result["success"] = inner_success
        result["metrics"] = inner_metrics
        result["idea"] = best_idea
        result["logic_check_passed"] = logic_passed
        result["logic_issues"] = [
            {k: v for k, v in issue.items() if k != 'line_content'}
            for issue in logic_issues[:5]
        ]

        # === [SmokeMode] 冒烟模式判定（gate 本函数内全部冒烟专用分支）===
        _smoke_mode = bool(getattr(self.config, 'smoke_only_mode', False))

        # === 三 Baseline 校验（应用于最佳 sub-result） ===
        if _smoke_mode:
            # === [SmokeMode] 冒烟模式专用记账分支 ===
            # success = 冒烟通过与否；acc 记为 N/A（metrics 不含 accuracy 键，
            # experiment_results.md 渲染为 smoke✓/smoke✗）。
            # 跳过：baseline 校验、tampered 审计、Zero Accuracy 判定、
            # best_result/best_code.py 更新（smoke 模式不产生可比 acc，best 无意义）。
            smoke_passed = bool(isinstance(inner_metrics, dict) and inner_metrics.get("smoke_passed"))
            result["smoke_passed"] = smoke_passed
            if inner_success and smoke_passed:
                self.logger.info(
                    f"✅ [SmokeMode] 第 {iteration} 轮 GPU 冒烟通过，acc=N/A，"
                    f"跳过正式评测与全部 baseline / Zero-Accuracy / best 审计"
                )
                result["feedback"] = (
                    f"[Iteration {iteration}: Smoke Passed] GPU smoke test passed on a real GPU "
                    f"(selector instantiated with synthetic inputs, forward OK, token count/dims OK). "
                    f"Formal evaluation is skipped in smoke_only_mode; accuracy is N/A."
                )
                self.last_error_log = None
            else:
                # 冒烟反复失败耗尽修复：success=False，acc=N/A，同样跳过上述审计
                error_summary = (inner_feedback or "Unknown error")
                self.logger.error(f"❌ [SmokeMode] Iteration {iteration} 冒烟反复失败: {error_summary[:200]}")
                max_inner_repairs = getattr(self.config, 'max_inner_repairs', 2)
                result["feedback"] = (
                    f"[Iteration {iteration}: Smoke Failed] GPU smoke test still failed after "
                    f"{max_inner_repairs} inner-loop repairs. Latest error: {error_summary}\n"
                    f"Please analyze the failure cause and correct the implementation in the next round."
                )
                self.last_error_log = error_summary if error_type == "execution" else None
        elif inner_success:
            acc = self.task_profile.extract_primary_metric(inner_metrics)

            # --- 执行后保留比例审查（防绕过静态检查） ---
            code_to_audit = best_code_in_iter if best_code_in_iter else ""
            from autoresearch_tasks import current_task
            patch_spec = current_task().patch_spec
            has_retain_ratio = getattr(self.config, 'has_retain_ratio', True)
            patch_mode = getattr(self.config, 'patch_mode', 'model')
            main_tensor = getattr(self.config, 'main_tensor_name', patch_spec.main_tensor_name)
            # G2: enable_tamper_audit=False 时跳过整个审查（默认 True == 现行为）
            if not getattr(self.config, 'enable_tamper_audit', True):
                code_to_audit = ""
            tvt_var = re.escape(patch_spec.target_tokens_var)
            rr_var = patch_spec.retain_ratio_var
            core_target_expr = patch_spec.mmtok_core_target_expr
            hardcoded_patterns = [
                # 删除过于宽泛的 k\s*= 模式，避免误杀合法代码
                rf'{tvt_var}\s*=\s*[Nn]\s*//\s*\d+',
                rf'{tvt_var}\s*=\s*int\s*\(\s*[Nn]\s*//\s*\d+\s*\)',
                rf'{re.escape(main_tensor)}\.shape\[0\]\s*//\s*\d+',
            ]
            is_tampered = False
            for pattern in hardcoded_patterns:
                if re.search(pattern, code_to_audit):
                    is_tampered = True
                    break
            if not is_tampered and code_to_audit:
                tvt_assignments = list(re.finditer(rf'{tvt_var}\s*(?!=)=\s*([^#\n]+)', code_to_audit))
                if tvt_assignments:
                    has_valid_source = False
                    # FIX: 对于 selector 模式（如 LLaVA / Qwen selector），
                    # target_vision_tokens 由模型包装器计算并作为参数传入，
                    # 因此 selector 内部不需要重新从 token_retain_ratio 计算。
                    # 需要区分"所有赋值都是合法的"和"存在非法赋值"。
                    any_non_skipped_assignment = False
                    for m in tvt_assignments:
                        # FIX: 排除函数/方法参数默认值，如 `def foo(..., target_vision_tokens: int = 32, ...)`
                        line_start = code_to_audit.rfind('\n', 0, m.start()) + 1
                        line_prefix = code_to_audit[line_start:m.start()]
                        if 'def ' in line_prefix:
                            continue  # 函数参数默认值，跳过
                            
                        expr = m.group(1).strip()
                        # FIX: 处理跨行的函数调用/表达式（如 multiline getattr(...)）
                        if expr.count('(') > expr.count(')'):
                            rest = code_to_audit[m.end():]
                            depth = expr.count('(') - expr.count(')')
                            i = 0
                            while i < len(rest) and depth > 0:
                                c = rest[i]
                                if c == '(':
                                    depth += 1
                                elif c == ')':
                                    depth -= 1
                                expr += c
                                i += 1
                        # FIX: 清理函数调用参数中的尾随逗号和后续参数
                        # 如 `target_vision_tokens=tgt, image_grid_thw=grid` -> `tgt`
                        # 如 `target_vision_tokens=self.target_vision_tokens,` -> `self.target_vision_tokens`
                        expr = expr.split(',')[0].strip().rstrip(',').strip()
                            
                        # 同样排除类型注解前缀
                        if expr.startswith(':'):
                            continue
                            
                        # FIX: 排除纯变量引用赋值（如 self.target_vision_tokens = target_vision_tokens）
                        # 这类赋值只是将参数/变量传递给属性，不是硬编码比例
                        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_.]*$', expr):
                            continue
                            
                        # FIX: selector 模式下，从 mmtok_core 读取目标 token 数是合法的
                        # （模型包装器已经根据 retain_ratio / target_vision_tokens 设置好）。
                        # 支持直接属性访问和 getattr 回退写法。
                        if patch_mode == 'selector' and re.search(
                            rf'self\._?mmtok_core[,.]?\s*["\']?{tvt_var}', expr
                        ):
                            continue
                            
                        any_non_skipped_assignment = True
                        if has_retain_ratio and rr_var in expr:
                            has_valid_source = True
                            break
                        if not has_retain_ratio and core_target_expr in expr:
                            has_valid_source = True
                            break
                    # FIX: 对于 selector 模式（LLaVA / Qwen），
                    # 如果所有 target_vision_tokens 赋值都是合法的（被 skip），不应标记 tampered
                    if not has_valid_source:
                        if patch_mode == 'selector' and not any_non_skipped_assignment:
                            # selector 模式：所有赋值都是合法的函数参数传递/纯变量引用/从 core 读取
                            pass  # 不标记 tampered
                        else:
                            is_tampered = True

            if is_tampered:
                self.logger.warning(
                    f"⚠️ 第 {iteration} 轮执行后审查发现硬编码保留比例！accuracy={acc:.4f} 无效。"
                )
                inner_success = False
                result["success"] = False
                result["is_tampered_ratio"] = True
                result["metrics"] = inner_metrics
                if has_retain_ratio:
                    expected_retain_ratio = self._get_expected_retain_ratio()
                    result["feedback"] = (
                        f"[Iteration {iteration}: Tampered Ratio] Post-execution audit detected hard-coded retention ratio. "
                        f"The experiment mandates {expected_retain_ratio} retention via TOKEN_RETAIN_RATIO. "
                        f"Secretly modifying the ratio is cheating and leads to unfairly high accuracy."
                    )
                else:
                    result["feedback"] = (
                        f"[Iteration {iteration}: Tampered Ratio] Post-execution audit detected hard-coded retention ratio. "
                        f"The experiment mandates using self._mmtok_core.target_vision_tokens as the target token count. "
                        f"Secretly hard-coding a different ratio is cheating and leads to unfairly high accuracy."
                    )
                self.last_error_log = None
                # v6.1 FIX: tampered 结果必须阻止后续进入 below_mmtok / 超越 mmtok 分支更新 best_result
                result["_skip_best_update"] = True

            # FIX: 检测回退到基础代码的情况（accuracy=0 意味着算法未真正执行）
            if (getattr(self.config, 'enable_zero_metric_crash_detection', True)
                    and inner_success and acc == 0.0):
                self.logger.warning(
                    f"⚠️ 第 {iteration} 轮检测到 accuracy=0，判定为回退到基础代码或算法未生效。"
                )
                inner_success = False
                result["success"] = False
                result["feedback"] = (
                    f"[Iteration {iteration}: Zero Accuracy] accuracy=0.0000 indicates the code fell back to "
                    f"the base template or the algorithm did not execute properly. "
                    f"Please ensure the generated code contains a valid algorithm implementation "
                    f"and does not silently crash or fallback."
                )
                self.last_error_log = None
                result["_skip_best_update"] = True

            # Fake-success detection
            has_retain_ratio = getattr(self.config, 'has_retain_ratio', True)
            # 口径守卫：original == mmtok baseline 时（如 OCRBench 均为 0.32），
            # 等值 fake 判定会把真实命中 baseline 的合法结果误判为作弊，禁用该检查。
            _baseline_collapsed = (abs(self.ORIGINAL_BASELINE - self.MMTOK_BASELINE)
                                   < self.FAKE_SUCCESS_TOLERANCE)
            if _baseline_collapsed and not getattr(self, '_warned_baseline_collapse', False):
                self._warned_baseline_collapse = True
                self.logger.warning(
                    f"⚠️ [Engine] ORIGINAL_BASELINE == MMTOK_BASELINE ({self.MMTOK_BASELINE:.4f})，"
                    "等值 fake-success 检测已禁用（真实命中 baseline 的结果不应判作弊）"
                )
            is_fake = (has_retain_ratio and not _baseline_collapsed
                       and abs(acc - self.ORIGINAL_BASELINE) < self.FAKE_SUCCESS_TOLERANCE)

            # LLaVA fallback detection: even without retain_ratio, detect silent fallback
            # to retain-all via try/except or other mechanisms
            if not has_retain_ratio and not is_fake:
                llava_fake_tolerance = self.FAKE_SUCCESS_TOLERANCE * 1.5
                if abs(acc - self.ORIGINAL_BASELINE) < llava_fake_tolerance and acc > self.MMTOK_BASELINE + 0.005:
                    is_fake = True
                    self.logger.warning(
                        f"⚠️ LLaVA Fake Success detected! accuracy={acc:.4f} is suspiciously close to "
                        f"original baseline ({self.ORIGINAL_BASELINE:.4f}). Possible try/except fallback retaining all tokens."
                    )
            is_uniform_fallback = (getattr(self.config, 'enable_uniform_fallback_detection', True)
                                   and abs(acc - self.UNIFORM_STEP_BASELINE) < self.FAKE_SUCCESS_TOLERANCE)
            below_mmtok = acc < self.MMTOK_BASELINE

            if is_fake:
                # 虚假成功：算法实际未执行压缩（异常回退到保留全部 token）
                self.logger.warning(
                    f"⚠️ 第 {iteration} 轮为虚假成功！accuracy={acc:.4f} 等于原始模型 baseline "
                    f"({self.ORIGINAL_BASELINE:.4f})，说明代码异常回退到保留全部 token，算法未真正生效。"
                )
                inner_success = False
                result["success"] = False
                result["is_fake_success"] = True
                result["metrics"] = inner_metrics
                result["feedback"] = (
                    f"[Iteration {iteration}: Fake Success] accuracy={acc:.4f} equals the original uncompressed model baseline "
                    f"({self.ORIGINAL_BASELINE:.4f}). This indicates the try/except fallback branch was triggered, retaining all visual tokens. "
                    f"Please fix numerical stability (e.g., half-precision SVD issues) or algorithm implementation to ensure real token compression is executed."
                )
                # Fake success 的信息已通过 result['feedback'] 进入 history_feedback，
                # 不应污染 CoderAgent 的下一轮 error_log
                self.last_error_log = None
            elif is_uniform_fallback:
                # 新陷阱：算法崩溃后回退到均匀步长采样
                self.logger.warning(
                    f"⚠️ 第 {iteration} 轮检测到均匀步长回退指纹！accuracy={acc:.4f} 等于均匀步长 baseline "
                    f"({self.UNIFORM_STEP_BASELINE:.4f})，说明代码因异常（非法参数、半精度崩溃、未定义方法等）"
                    f"静默回退到了均匀步长采样，核心算法从未执行。"
                )
                inner_success = False
                result["success"] = False
                result["is_fake_success"] = True
                result["is_uniform_fallback"] = True
                result["metrics"] = inner_metrics
                result["feedback"] = (
                    f"[Iteration {iteration}: Uniform Step Fallback Trap] accuracy={acc:.4f} equals the uniform step sampling fingerprint "
                    f"({self.UNIFORM_STEP_BASELINE:.4f}). This indicates the core algorithm in the code was silently caught by try/except due to some error (e.g., illegal API parameter, "
                    f"half-precision LAPACK crash, calling a non-existent method), and actually fell back to uniform step sampling. "
                    f"Please check: 1) Whether unsupported PyTorch parameters such as `pivoting=True` were used; "
                    f"2) Whether torch.linalg.svd/qr/eigh was called directly on fp16/bf16 (must convert to float32 first); "
                    f"3) Whether unverified methods (e.g., self.text_encoder) were called."
                )
                # Uniform fallback 的信息已通过 result['feedback'] 进入 history_feedback
                self.last_error_log = None
            elif below_mmtok:
                # 真实压缩了，但表现不如 MMTok baseline
                self.logger.info(
                    f"✅ 第 {iteration} 轮执行成功（真正压缩），准确率: {acc:.4f}，"
                    f"但低于 MMTok baseline ({self.MMTOK_BASELINE:.4f})"
                )
                summary = eval_log[-1000:] if len(eval_log) > 1000 else eval_log
                result["feedback"] = (
                    f"[Iteration {iteration}: Success (Real Compression)] accuracy={acc:.4f}\n"
                    f"Comparison baselines: Original model={self.ORIGINAL_BASELINE:.4f}, MMTok baseline={self.MMTOK_BASELINE:.4f}, Uniform step={self.UNIFORM_STEP_BASELINE:.4f}\n"
                    f"Current result is below the MMTok baseline, indicating the compression strategy still has room for improvement.\nLog summary: {summary}"
                )
                self.last_error_log = None

                # 仍然可能更新 best_result（如果这是目前真压缩中最好的）
                best_acc = 0
                if self.state_manager.state.get("best_result"):
                    best_acc = float(self.state_manager.state["best_result"].get("metrics", {}).get("accuracy") or 0)
                if acc > best_acc and not result.get("_skip_best_update"):
                    best_code_path = os.path.join(self.config.log_dir, "best_code.py")
                    temp_best_path = best_code_path + ".tmp"
                    with open(temp_best_path, 'w', encoding='utf-8') as f:
                        f.write(best_code_in_iter)
                    os.replace(temp_best_path, best_code_path)
                    self.logger.info(f"🏆 最佳代码已保存到: {best_code_path} (准确率: {acc:.4f})")
            else:
                # 真实压缩且超越或达到 MMTok baseline
                self.logger.info(
                    f"✅ 第 {iteration} 轮执行成功（真正压缩），准确率: {acc:.4f}，"
                    f"超越 MMTok baseline ({self.MMTOK_BASELINE:.4f})"
                )
                summary = eval_log[-1000:] if len(eval_log) > 1000 else eval_log
                result["feedback"] = (
                    f"[Iteration {iteration}: Success (Real Compression)] accuracy={acc:.4f}\n"
                    f"Comparison baselines: Original model={self.ORIGINAL_BASELINE:.4f}, MMTok baseline={self.MMTOK_BASELINE:.4f}, Uniform step={self.UNIFORM_STEP_BASELINE:.4f}\n"
                    f"Current result reaches or surpasses the MMTok baseline, which is effective progress.\nLog summary: {summary}"
                )
                self.last_error_log = None

                best_acc = 0
                if self.state_manager.state.get("best_result"):
                    best_acc = float(self.state_manager.state["best_result"].get("metrics", {}).get("accuracy") or 0)
                if acc > best_acc and not result.get("_skip_best_update"):
                    best_code_path = os.path.join(self.config.log_dir, "best_code.py")
                    temp_best_path = best_code_path + ".tmp"
                    with open(temp_best_path, 'w', encoding='utf-8') as f:
                        f.write(best_code_in_iter)
                    os.replace(temp_best_path, best_code_path)
                    self.logger.info(f"🏆 最佳代码已保存到: {best_code_path} (准确率: {acc:.4f})")

                # v2: 不再使用 deepen_threshold 门槛机制
                # 探索完全由 UCT 树驱动，不需要人工设置门槛
                
            # === Depth mode eval_time constraint check ===
            if current_mode in ("depth", "escape_component"):
                parent_eval_time = strategy.get("parent_eval_time", 0.0)
                current_eval_time = float(inner_metrics.get("eval_time", 0.0))
                if parent_eval_time > 0 and current_eval_time > parent_eval_time * 1.20:
                    self.logger.warning(
                        f"⚠️ 第 {iteration} 轮深耕模式测试时间超标！ "
                        f"current={current_eval_time:.1f}s > parent={parent_eval_time:.1f}s * 1.20"
                    )
                    # 追加时间约束警告到 feedback
                    time_warning = (
                        f"\n[CRITICAL TIME CONSTRAINT] The evaluation time ({current_eval_time:.1f}s) "
                        f"exceeds the parent algorithm's time ({parent_eval_time:.1f}s) by more than 20%. "
                        f"In depth mode, the new algorithm MUST NOT increase test time by more than 20%. "
                        f"Please optimize the implementation to reduce computational overhead while preserving accuracy."
                    )
                    result["feedback"] = result.get("feedback", "") + time_warning
                    # FIX: eval_time 超标的算法不应加入高分模板，标记为失败
                    result["is_eval_time_violation"] = True
                    result["success"] = False
                    inner_success = False
        else:
            error_summary = (inner_feedback or "Unknown error")
            self.logger.error(f"❌ Iteration {iteration} failed: {error_summary[:200]}")

            max_inner_repairs = getattr(self.config, 'max_inner_repairs', 2)
            feedback_parts = [f"[Iteration {iteration}: Failed] Still failed after {max_inner_repairs} inner-loop repairs. Latest error: {error_summary}"]
            feedback_parts.append("\nPlease analyze the failure cause and correct mathematical boundary conditions or engineering implementation in the next round.")
            result["feedback"] = "".join(feedback_parts)

            self.last_error_log = error_summary if error_type == "execution" else None

        # C包：基于代码重分类方向，修正 Idea 文本标签
        selected_direction = self.explorer.history_tracker._extract_idea_direction(best_idea)
        approach_type = self.explorer.history_tracker._extract_approach_type(best_idea)
        true_direction = selected_direction
        ops = []
        if best_code_in_iter:
            true_direction = self._reclassify_direction_by_code(best_code_in_iter, selected_direction)
            ops = list(self.explorer.history_tracker._extract_operators(best_code_in_iter))
            if true_direction != selected_direction:
                self.logger.info(
                    f"🔄 [Engine] 方向重分类: {selected_direction} → {true_direction} "
                    f"(基于代码实际内容)"
                )

        result["direction"] = true_direction
        result["approach_type"] = approach_type
        result["topic"] = self.explorer.last_used_topic
        result["operators"] = ops
        # 原型仅 breadth 轮有意义（depth/cross 不轮转原型；facade 上的值是上一轮
        # breadth 的残留，误记会把 depth 结果算到原型饱和账上）
        result["archetype"] = (
            getattr(self.explorer, 'last_used_archetype', None) or ""
        ) if current_mode == "breadth" else ""
        # 行为指纹：取评测前门禁登记的 pending 签名落 result（供 history/state 持久化；
        # update_idea_history 会 pop 消费，此处只读）
        _sel_sig = getattr(self.explorer.history_tracker, 'pending_signatures', {}).get(iteration)
        if _sel_sig:
            result["selection_signature"] = _sel_sig

        # === v12.0: Research Notebook 轮末 hook（观察记录 + 假设对账 + 蒸馏）===
        # 注意：必须先于 update_idea_history / elite 追加执行，
        # 因为 record_observation 派生的 high_score claim 是这两处闸门（is_verified）的依据。
        if getattr(self, 'notebook', None) is not None:
            try:
                # 机制签名：算子集合 + 流程序列
                flow_seq = []
                if best_code_in_iter:
                    try:
                        flow_seq = list(self.explorer.history_tracker._extract_code_flow_sequence(best_code_in_iter))
                    except Exception:
                        flow_seq = []
                from autoresearch_explorer.research_notebook import ResearchNotebook
                mech_sig = ResearchNotebook.mechanism_sig(ops, flow_seq)
                self.notebook.record_observation(iteration, result, strategy or {}, mech_sig)
                # 仅真实成功（非 fake/非回退）时才对账：实现失败不是对假设的证伪
                _can_reconcile = (
                    result.get("success")
                    and not result.get("is_fake_success")
                    and not result.get("is_uniform_fallback")
                    and acc > 0
                )
                if (_can_reconcile and current_mode in ("depth", "escape_component")
                        and (strategy or {}).get("hypothesis_id")):
                    _ref_acc = (strategy or {}).get("parent_accuracy")
                    verdict = self.notebook.reconcile(iteration, acc, reference_acc=_ref_acc)
                    if verdict:
                        self.logger.info(f"🧮 [Engine v12.0] 假设对账结果: {verdict}")
                elif (current_mode in ("depth", "escape_component")
                        and (strategy or {}).get("hypothesis_id")):
                    # 实现失败的实验：假设标记为 inconclusive，避免污染校准统计
                    try:
                        self.notebook.mark_experiment_failed(
                            (strategy or {}).get("hypothesis_id"))
                    except Exception:
                        pass
                self.notebook.maybe_distill(iteration)
            except Exception as nb_err:
                self.logger.warning(f"⚠️ [Engine v12.0] Notebook 轮末更新失败（不影响主流程）: {nb_err}")

        # 【重要】更新Explorer的Idea历史，用于防止死胡同
        # v2.8 FIX: 传入 strategy（含 target_leaf），确保树更新和 plan 阶段选定的 leaf 一致
        if _smoke_mode:
            # [SmokeMode] 照常调用 update_idea_history（承载机制覆盖统计），
            # 但跳过探索树 backprop（无 acc 信号；最安全方式是临时摘除树，见该helper注释）
            self._update_idea_history_maybe_skip_tree(
                best_idea, result, iteration, code=best_code_in_iter, strategy=strategy,
            )
        else:
            self.explorer.update_idea_history(
                best_idea, result, iteration, code=best_code_in_iter,
                static_smoke_test=self._static_smoke_test,
                fallback_detector=getattr(self.coder, 'fallback_detector', None),
                api_validator=getattr(self.coder, 'api_validator', None),
                strategy=strategy,
            )

        # Phase I1: 张量组合 bandit 反馈（acc 口径 == extract_primary_metric，
        # baseline 为生效的 mmtok_baseline —— config 可覆盖，经 MMTOK_BASELINE 属性解析）
        _combo_alloc = getattr(getattr(self, 'tree_controller', None), 'combo_allocator', None)
        if _combo_alloc is not None and (strategy or {}).get("tensor_combo"):
            _combo_alloc.record_reward(
                strategy["tensor_combo"], acc, self.MMTOK_BASELINE,
                arch=(strategy or {}).get("combo_archetype") or "")

        # Phase I2: 将本轮张量组合写入 result，供 state 持久化与 results.md 展示
        # （组合探索关闭时 strategy 无该键，result 不新增字段，行为逐字不变）
        if (strategy or {}).get("tensor_combo"):
            result["tensor_combo"] = strategy["tensor_combo"]

        # 方案B：将历史代码记录到 MathTheoryMapper，用于后续结构距离新颖度计算
        if best_code_in_iter and hasattr(self.explorer, 'math_theory_mapper'):
            self.explorer.math_theory_mapper.record_historical_code(best_code_in_iter)

        # 记录本轮 topic 效果，用于 RobustTopicGenerator 学习
        if hasattr(self.explorer, 'topic_generator') and hasattr(self.explorer, 'last_used_topic') and self.explorer.last_used_topic:
            acc = float(result["metrics"].get("accuracy") or 0.0)
            self.explorer.topic_generator.record_result(
                self.explorer.last_used_topic,
                success=result["success"],
                accuracy=acc
            )
            # v2.7: 高分 topic 加入池子，用于未来复用（合成标签除外，防召回污染）
            if (result.get("success") and not result.get("is_fake_success")
                    and not result.get("is_tampered_ratio")
                    and hasattr(self.explorer, 'candidate_pool')
                    and not self._is_synthetic_topic(self.explorer.last_used_topic)):
                self.explorer.candidate_pool.add_good_topic(
                    self.explorer.last_used_topic, acc, iteration
                )

            # v6.0 FIX: 使用自适应 elite_threshold 替代硬编码 0.86
            from autoresearch_tasks import current_task
            base = getattr(self.config, 'original_baseline', None) or getattr(
                self.config, 'mmtok_baseline', current_task().baselines.mmtok_baseline)
            elite_threshold = getattr(self.config, 'deepen_threshold_base', float(base) + 0.015)
            if acc >= elite_threshold and not result.get("is_fake_success") and not result.get("is_tampered_ratio"):
                self._append_elite_algorithm(result, iteration)

        # v2: 旧 EvolutionController 已禁用，策略由 UCT 树主导

        # 【自动记录】运行完一个 iteration 就立即持久化结果
        self.state_manager.update_iteration(iteration, result)
        self._update_results_md()
            
        # === v2: 保存探索树 + controller 状态 ===
        if hasattr(self, 'exploration_tree') and self.exploration_tree:
            try:
                controller_state = self.tree_controller.get_state() if hasattr(self, 'tree_controller') else None
                self.exploration_tree.save(controller_state=controller_state)
                self.logger.debug("🌲 [Engine] 探索树已保存")
            except Exception as tree_err:
                self.logger.warning(f"⚠️ [Engine] 探索树保存失败: {tree_err}")
        return result

    def run_iteration(self, iteration: int, idea: str = None, strategy: Dict[str, Any] = None) -> Dict[str, Any]:
        """运行单轮迭代（v7.4: 每轮生成并验证单个 idea）"""
        iter_dir = os.path.join(self.config.log_dir, f"iter_{iteration:03d}")
        os.makedirs(iter_dir, exist_ok=True)

        # 更新日志处理器
        self.logger = setup_logging(iter_dir, iteration)
        self.state_manager.logger = self.logger
        self.llm_client.logger = self.logger
        self.executor.logger = self.logger
        # Fix: 同步更新各 Agent 的 logger，避免日志分散到旧文件
        self.coder.logger = self.logger
        self.explorer.logger = self.logger

        # [AsyncEval] 回收点 1：每轮开始处 drain 已完成评测并补记账（主线程）
        if self._async_eval_enabled():
            self._drain_async_evals()

        # [LookAhead] 取出槽位等待期间预生产的本轮产物（无论是否消费都清空，
        # 避免陈旧缓存；strategy 的复用发生在 run() 主循环）
        lookahead_pkg = None
        if (idea is None and isinstance(self._lookahead, dict)
                and self._lookahead.get("iteration") == iteration):
            lookahead_pkg = self._lookahead
        self._lookahead = None

        result = {
            "iteration": iteration,
            "success": False,
            "metrics": {},
            "feedback": "",
            "timestamp": datetime.now().isoformat(),
            "logic_check_passed": False,
            "logic_issues": []
        }

        try:
            # 初始化 current_mode，确保所有代码路径都有定义（包括调试模式）
            current_mode = strategy.get("mode", "breadth") if strategy else "breadth"
            best_idea = None  # 提前初始化，避免 except 块中 NameError

            # [LookAhead] 快速路径：idea+code 已在槽位等待期间预生产完毕，
            # 直接快照提交异步评测，跳过全部 LLM 生成阶段（秒级交出）
            if lookahead_pkg is not None and lookahead_pkg.get("code"):
                fast_result = self._consume_lookahead(
                    iteration, lookahead_pkg, iter_dir, result, strategy)
                if fast_result is not None:
                    return fast_result
                # 提交失败（None）：落入正常路径现产；预生产已落盘的 idea
                # 文件会被下方 Resume 检测复用，不重复生成

            # 0. 断点续跑检测：若目录下已有预生成 idea 文件，直接读取
            single_idea = None
            if idea is None:
                idea_path = os.path.join(iter_dir, "1_algorithm_idea.md")
                if os.path.exists(idea_path):
                    with open(idea_path, 'r', encoding='utf-8') as f:
                        single_idea = f.read()
                    self.logger.info("🔄 [Resume] 检测到预生成 idea 文件，跳过搜索与生成阶段")

            # === 多 GPU 分支并行评测：parallel_branches > 0 且 ≥2 张合格 GPU 时启用 ===
            if (single_idea is None and idea is None
                    and getattr(self.config, 'parallel_branches', 0) > 0):
                _parallel_gpus = self._parallel_eligible_gpus()
                _B = min(int(getattr(self.config, 'parallel_branches', 0)),
                         len(_parallel_gpus),
                         len(getattr(self, '_branch_envs', [])))
                if _B >= 2:
                    return self._execute_iteration_parallel_branches(
                        iteration, strategy, iter_dir, result, _parallel_gpus[:_B])
                self.logger.info(
                    f"ℹ️ [Parallel] 合格 GPU 数不足（{len(_parallel_gpus)} 张），使用单 idea 路径")

            if single_idea is None and idea is None:
                # 1. Explorer Agent 生成单个 idea
                single_idea = self.explorer.generate_idea(
                    iteration,
                    self.state_manager.state["history_feedback"],
                    iter_dir,
                    strategy=strategy
                )
                if not single_idea:
                    # v7.3/v7.4: Agent failed to generate ideas — classify failure and return early
                    failure_mode = self._classify_agent_failure(strategy)
                    result["agent_failure_mode"] = failure_mode
                    result["feedback"] = (
                        f"[Iteration {iteration}: Agent Failure] No ideas generated. "
                        f"Failure mode: {failure_mode}"
                    )
                    # FIX: 即使 Agent 失败也记录 topic，确保断点续跑时能恢复 TopicGenerator 状态
                    result["topic"] = getattr(self.explorer, 'last_used_topic', None)
                    # 失败 == acc 0.0 记账，清除 combo pending（口径与评测失败一致）
                    self._record_combo_reward(strategy, 0.0)
                    self.state_manager.update_iteration(iteration, result)
                    self._update_results_md()
                    return result

                if self._interrupted:
                    self.logger.info("🛑 检测到中断请求，跳过本轮剩余步骤")
                    self._record_combo_reward(strategy, 0.0)
                    return result

            elif idea is not None:
                self.logger.info(f"🐛 调试模式：跳过搜索，直接使用提供的 idea")
                single_idea = idea

            # Save single idea
            idea_path = os.path.join(iter_dir, "1_algorithm_idea.md")
            with open(idea_path, 'w', encoding='utf-8') as f:
                f.write(single_idea)

            # 2. Single idea validation (v7.4: one idea per iteration)
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"🔬 验证 Idea")
            self.logger.info(f"{'='*60}")

            try:
                inner_success, inner_metrics, best_code_in_iter, eval_log, logic_passed, logic_issues, inner_feedback, error_type = self._execute_single_idea(
                    iteration, single_idea, iter_dir, 0, strategy
                )
            except Exception as exec_e:
                self.logger.error(f"❌ Idea 执行异常: {exec_e}")
                result["feedback"] = f"[Iteration {iteration}: Execution Exception] {exec_e}"
                self._record_combo_reward(strategy, 0.0)
                self.state_manager.update_iteration(iteration, result)
                self._update_results_md()
                return result

            best_idea = single_idea

            # [AsyncEval] 评测已异步交出：以最小 pending 状态收尾，立即返回。
            # 不写 completed_iterations（last_iteration 不前进 → 进程重启后本轮
            # 自然整体重做，与 interrupted 迭代处理一致），不写 experiment_results.md；
            # 评测完成后由 _drain_async_evals 用真实 acc 补全全部记账。
            if isinstance(inner_metrics, dict) and inner_metrics.get("eval_pending"):
                result["eval_pending"] = True
                result["idea"] = best_idea
                result["topic"] = getattr(self.explorer, 'last_used_topic', None)
                self.logger.info(
                    f"⏭️ [AsyncEval] iter={iteration} 以 pending 收尾，立即进入下一轮"
                )
                return result

            result = self._finalize_iteration_result(
                iteration, result, strategy, current_mode, best_idea,
                inner_success, inner_metrics, best_code_in_iter, eval_log,
                logic_passed, logic_issues, inner_feedback, error_type)
            return result

        except Exception as e:
            error_msg = f"迭代 {iteration} 异常: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)

            # 外层 crash 且未进入 finalize（inner_success 未定义）时，
            # combo reward 从未记录——补记 acc=0.0，清除 pending（已进入
            # finalize 的由 finalize 记账，避免双计）
            if 'inner_success' not in locals():
                self._record_combo_reward(strategy, 0.0)

            # v7.4: 单 idea 模式下，若 idea 已成功执行，保留其结果
            if 'inner_success' in locals() and inner_success:
                from autoresearch_tasks import current_task
                acc = current_task().extract_primary_metric(inner_metrics) if 'inner_metrics' in locals() else 0.0
                self.logger.warning(
                    f"⚠️ 迭代 {iteration} 发生异常，但 Idea 已成功执行 "
                    f"(acc={acc:.4f})，使用该结果收尾"
                )
                result["feedback"] = (
                    f"[Iteration {iteration}: Partial Success] Idea succeeded with "
                    f"accuracy={acc:.4f}, but later step crashed with: {str(e)}"
                )
                # best_idea 和 best_code_in_iter 已在 try 块中设置
            else:
                result["feedback"] = f"[Iteration {iteration}: System Crash] {str(e)}"
                best_idea = None
                best_code_in_iter = None
            
            # 统一处理 idea 历史更新
            idea_for_history = best_idea if best_idea is not None else (idea if ('idea' in locals() and idea is not None) else None)
            if idea_for_history:
                result["idea"] = idea_for_history
                self.explorer.update_idea_history(
                    idea_for_history, result, iteration, code=best_code_in_iter,
                    static_smoke_test=self._static_smoke_test,
                    fallback_detector=getattr(self.explorer, 'fallback_detector', None),
                    api_validator=getattr(self.explorer, 'api_validator', None),
                    strategy=strategy,
                )
                if hasattr(self.explorer, 'topic_generator') and hasattr(self.explorer, 'last_used_topic') and self.explorer.last_used_topic:
                    acc = float(result["metrics"].get("accuracy") or 0.0)
                    self.explorer.topic_generator.record_result(
                        self.explorer.last_used_topic,
                        success=result["success"],
                        accuracy=acc
                    )
                    # v2.7: 高分 topic 加入池子，用于未来复用（合成标签除外，防召回污染）
                    if (result.get("success") and not result.get("is_fake_success")
                            and not result.get("is_tampered_ratio")
                            and hasattr(self.explorer, 'candidate_pool')
                            and not self._is_synthetic_topic(self.explorer.last_used_topic)):
                        self.explorer.candidate_pool.add_good_topic(
                            self.explorer.last_used_topic, acc, iteration
                        )
                # 提取方向/类型/算子
                selected_direction = self.explorer.history_tracker._extract_idea_direction(idea_for_history)
                approach_type = self.explorer.history_tracker._extract_approach_type(idea_for_history)
                true_direction = selected_direction
                ops = []
                if best_code_in_iter:
                    true_direction = self._reclassify_direction_by_code(best_code_in_iter, selected_direction)
                    ops = list(self.explorer.history_tracker._extract_operators(best_code_in_iter))
                    if true_direction != selected_direction:
                        self.logger.info(
                            f"🔄 [Engine] 方向重分类: {selected_direction} → {true_direction} "
                            f"(基于代码实际内容)"
                        )
                result["direction"] = true_direction
                result["approach_type"] = approach_type
                result["topic"] = self.explorer.last_used_topic
                result["operators"] = ops
                # Phase I2: 异常路径同样持久化本轮张量组合（关闭时不新增字段）
                if (strategy or {}).get("tensor_combo"):
                    result["tensor_combo"] = strategy["tensor_combo"]
            
            # 【自动记录】异常情况下也立即持久化结果
            self.state_manager.update_iteration(iteration, result)
            self._update_results_md()
            return result
    
    def _extract_execution_error(self, eval_log: str, max_chars: int = 3000) -> str:
        """从评测日志中提取可用于代码修复的核心错误信息"""
        if not eval_log:
            return ""
        
        lines = eval_log.split('\n')
        
        # 提取 Python Traceback
        traceback_lines = []
        in_traceback = False
        for line in lines:
            if 'Traceback (most recent call last)' in line:
                in_traceback = True
            if in_traceback:
                traceback_lines.append(line)
            # Traceback 通常以空行或错误类型行结束，简单退出
            if in_traceback and line.strip() and not line.startswith(' '):
                in_traceback = False
        
        # 提取 ERROR / CRITICAL 级别日志
        error_logs = [l for l in lines if 'ERROR' in l or 'CRITICAL' in l]
        
        # 提取 CUDA / 数值 / 运行时相关错误
        numeric_errors = [l for l in lines if any(k in l for k in [
            'CUDA error', 'RuntimeError', 'AssertionError', 
            'Shape mismatch', 'device', 'dtype', 'IndexError',
            'TypeError', 'ValueError', 'KeyError', 'AttributeError'
        ])]
        
        summary_parts = []
        if traceback_lines:
            summary_parts.extend(traceback_lines[-30:])
        if error_logs:
            summary_parts.extend(error_logs[-10:])
        if numeric_errors:
            summary_parts.extend(numeric_errors[-5:])
        
        if not summary_parts:
            # 兜底：保留最后 30 行
            summary_parts = lines[-30:]
        
        summary = "\n".join(summary_parts)
        if len(summary) > max_chars:
            summary = summary[-max_chars:]
        
        return f"[Execution Error Summary - original {len(eval_log)} chars]\n{summary}"
    
    @staticmethod
    def _is_contention_oom(error_log: str) -> bool:
        """检测环境性 OOM：显存被其他进程抢占（PyTorch 报错中会列出其他 Process 的占用）。
        此类错误 LLM 修复无效，应换卡重跑或跳过，而不是进入修复循环。"""
        if not error_log:
            return False
        lower_log = error_log.lower()
        if "out of memory" not in lower_log and "outofmemoryerror" not in lower_log:
            return False
        return bool(re.search(r"Process \d+ has [\d.]+ GiB memory in use", error_log))

    def _is_environment_error(self, error_log: str) -> bool:
        """检测错误是否为环境依赖问题（如 ImportError、版本不兼容等）"""
        if not error_log:
            return False
        env_error_patterns = [
            'cannot import name',
            'ImportError',
            'ModuleNotFoundError',
            'No module named',
            'requires a newer version',
            'version conflict',
            'incompatible',
            'pip install',
        ]
        lower_log = error_log.lower()
        return any(p.lower() in lower_log for p in env_error_patterns)
    
    def _extract_env_error_msg(self, error_log: str) -> str:
        """从环境错误日志中提取最关键的一行信息"""
        for line in error_log.split('\n'):
            if any(p in line for p in ['cannot import name', 'ImportError', 'ModuleNotFoundError', 'No module named']):
                return line.strip()
        return "Environment dependency error detected"
    
    def _signal_handler(self, signum, frame):
        """处理 Ctrl+C 信号，设置中断标志并保存状态。
        注意：每次收到信号都必须抛出 KeyboardInterrupt，不能因 _interrupted 已设置而跳过，
        否则用户连续按 Ctrl+C 将完全失效（信号处理器被静默忽略）。
        """
        was_already_interrupted = self._interrupted
        self._interrupted = True
        if not was_already_interrupted:
            self.logger.warning("\n⚠️ 收到中断信号 (Ctrl+C)，将在当前步骤完成后安全退出...")
            # 立即保存状态
            try:
                # v7.6 FIX: 记录当前正在进行的迭代号，便于重启时识别未完成迭代
                current_iter = getattr(self, '_current_iteration', 0)
                if current_iter > 0:
                    self.state_manager.state["interrupted_at_iteration"] = current_iter
                self.state_manager.save()
                self.logger.info("💾 状态已保存，可以安全中断")
            except Exception as e:
                self.logger.warning(f"保存状态时出错: {e}")
            # v7.6 FIX: 同时保存探索树状态
            try:
                if hasattr(self, 'exploration_tree') and self.exploration_tree:
                    controller_state = self.tree_controller.get_state() if hasattr(self, 'tree_controller') else None
                    self.exploration_tree.save(controller_state=controller_state)
                    self.logger.info("🌲 探索树状态已保存")
            except Exception as e:
                self.logger.warning(f"⚠️ 探索树保存失败: {e}")
        else:
            self.logger.warning("\n⚠️ 再次收到中断信号 (Ctrl+C)，强制终止当前操作...")
        # 每次都抛出 KeyboardInterrupt，确保阻塞调用返回后能立即中断。
        # 若程序正处于 C 扩展阻塞调用（如 HTTP 请求）中，信号会被推迟到 C 调用返回后处理，
        # 但此处必须持续抛出异常，以保证调用返回后的第一时间中断执行。
        raise KeyboardInterrupt("用户中断")
    
    def _decide_strategy(self, i: int) -> Dict[str, Any]:
        """第 i 轮的探索策略决策（Phase A Combo 轮转 / Phase C UCT 引导）。

        每轮必须恰好调用一次（内部有 UCT 计数、Combo 轮转等状态副作用）；
        LookAhead 预生产会为下一轮提前调用，消费轮 run() 直接复用缓存的
        strategy，不会二次调用。
        """
        # === 阶段门控（组合探索开启时）===
        # Phase A（探索期，UCT 不介入）：还有 (combo, archetype) 对未试满
        # min_trials 时，跳过 decide_strategy，由 ComboAllocator 两级亏空轮转
        # 直接给出 strategy；迭代结果仍经 update_idea_history 写入探索树
        # （无 target_leaf 时 ensure_leaf 兜底），为 Phase C 的 UCT 积累 Q 值。
        # Phase C（深耕期）：恢复 decide_strategy（UCT 引导 leaf/topic）；
        # 深耕对由 _maybe_mark_objective_deepen 标记 objective_deepen
        # （全程 breadth，无父本）。
        alloc = getattr(self, 'combo_allocator', None)
        exploration_complete = bool(alloc and alloc.depth_pairs())
        if alloc is not None and not exploration_complete:
            strategy = {"mode": "breadth"}
            combo = alloc.select_combo(i)
            if combo:
                strategy["tensor_combo"] = combo
                strategy["combo_archetype"] = alloc.current_archetype(combo)
            self.logger.info(
                f"🎯 [Engine] Phase A 轮转探索: combo={strategy.get('tensor_combo', 'N/A')}, "
                f"archetype={strategy.get('combo_archetype', 'N/A')}（UCT 未介入）"
            )
        elif hasattr(self, 'tree_controller') and self.tree_controller:
            # Phase C: 动态探索策略由 UCT 树引导控制器决定
            strategy = self.tree_controller.decide_strategy(i)
            # === FIX: cross 模式需要 deepen_templates，若为空则降级 ===
            if strategy.get("mode") == "cross":
                if not getattr(self.explorer.history_tracker, 'deepen_templates', None):
                    self.logger.warning(
                        "⚠️ [Engine] TreeController 选择 cross 模式，但 deepen_templates 为空，"
                        "降级为 breadth 模式"
                    )
                    strategy = dict(strategy)
                    strategy["mode"] = "breadth"
                    strategy["target_domain"] = "general"
                    strategy["target_theory"] = "general"
                    # 修正 budget counter（decide_strategy 内已按 cross 计数）
                    self.tree_controller.cross_count = max(0, self.tree_controller.cross_count - 1)
                    self.tree_controller.breadth_count += 1
            # Phase 3: 组合深耕标记 —— 高 success-rate (combo, archetype) 对
            # 打 objective_deepen=True（全程 breadth，无父本；
            # enable_combo_deepen 关闭时原样返回）
            strategy = self._maybe_mark_objective_deepen(strategy)
            self.logger.info(
                f"🌲 [Engine] Phase C UCT strategy: mode={strategy.get('mode')}, "
                f"leaf={strategy.get('target_leaf', 'N/A')}, "
                f"domain={strategy.get('target_domain', 'N/A')}, "
                f"objective_deepen={bool(strategy.get('objective_deepen'))}"
            )
        else:
            strategy = {"mode": "breadth"}
        return strategy

    def run(self):
        """主循环"""
        # 注册 Ctrl+C 信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)

        # 挂起诊断：每 30 分钟向 stderr 转储一次所有线程堆栈。
        # 背景：2026-08-01 引擎在 idea 生成阶段静默挂起 9+ 小时（网络连接
        # CLOSE-WAIT、主线程无日志），无法事后定位；faulthandler 开销为零，
        # 转储内容进入 console/system 日志，便于下次挂起时直接看到卡点。
        import faulthandler
        faulthandler.dump_traceback_later(1800, repeat=True)

        # 确保原始代码已备份
        self._ensure_backup()
        
        # 调试模式：只跑一轮，使用现成的 idea
        if self.debug_idea_path:
            self.logger.info("\n" + "="*60)
            self.logger.info("🐛 调试模式：只运行一轮代码生成与评测")
            self.logger.info("="*60)
            with open(self.debug_idea_path, 'r', encoding='utf-8') as f:
                idea = f.read()
            result = self.run_iteration(1, idea=idea)
            # update_iteration 与 _update_results_md 已在 run_iteration 内部自动执行
            # [AsyncEval] 调试模式同样需要等待在途评测回收后再返回
            self._async_wait_all()
            return
        
        start_iter = self.state_manager.state["last_iteration"] + 1

        # [AsyncEval] 断点续跑：中断时在途评测已随进程丢失，其迭代未写入
        # completed_iterations（pending 不记账），last_iteration 不前进，
        # 本轮自然从最早丢失轮整体重做；此处仅打 WARNING 提示。
        if self._async_eval_enabled() and os.path.isdir(self.config.log_dir):
            _has_lost = any(
                re.match(r'iter_(\d{3})$', d) and int(d.split('_')[1]) >= start_iter
                for d in os.listdir(self.config.log_dir)
            )
            if _has_lost:
                self.logger.warning(
                    "⚠️ [AsyncEval] 检测到上次运行中断时仍有在途评测（评测进程已丢失），"
                    "对应迭代将整体重做"
                )

        # v7.6 FIX: 启动时检测并清理未完成的迭代目录
        self._cleanup_incomplete_iteration_dirs(start_iter)
        
        if start_iter > 1:
            self.logger.info(f"🔄 断点续跑: 从第 {start_iter} 轮开始")
        
        for i in range(start_iter, self.config.max_iterations + 1):
            if self._interrupted:
                self.logger.info("🛑 检测到中断请求，安全退出主循环")
                break
            
            # v7.6 FIX: 记录当前迭代号，供 signal handler 使用
            self._current_iteration = i
            
            self.logger.info("\n" + "="*60)
            self.logger.info(f"🔬 第 {i}/{self.config.max_iterations} 轮自动研究迭代")
            self.logger.info("="*60)
            
            # === 策略决策（LookAhead 预生产时已为本轮决策过的直接复用，
            # UCT/Combo 计数已在预生产时消耗，不能二次决策）===
            if (isinstance(self._lookahead, dict)
                    and self._lookahead.get("iteration") == i
                    and self._lookahead.get("strategy")):
                strategy = self._lookahead["strategy"]
                self.logger.info(
                    f"🔮 [LookAhead] 复用预生产阶段已决策的 strategy: "
                    f"mode={strategy.get('mode', 'breadth')}"
                )
            else:
                strategy = self._decide_strategy(i)

            # === deepen_mode 同步：深耕已改为 breadth 内 objective_deepen 生成路径，
            # mode 只会是 breadth/cross，deepen_mode 恒为 False ===
            mode = strategy.get("mode", "breadth")
            is_cross = mode in ("escape_paradigm", "cross", "escape_domain")
            self.explorer.history_tracker.deepen_mode = False
            self.coder.deepen_mode = False

            if strategy.get("objective_deepen"):
                self.logger.info(
                    f"📊 [Engine] 目标深耕模式: objective_deepen=True (breadth), "
                    f"combo={strategy.get('tensor_combo')}, "
                    f"archetype={strategy.get('combo_archetype')}")
            elif is_cross:
                self.logger.info(f"📊 [Engine] 跨范式模式: mode={mode}")
            else:
                self.logger.info(f"📊 [Engine] 广度模式: mode={mode}, deepen_mode=False")
            # 所有模式均不注入历史高分模板（config.disable_high_score_template_injection）
            self.coder.load_high_performer_templates([])
            
            # Fix 7.7: Fusion Round 已并入跨范式模式 (_generate_cross_paradigm_idea)，
            # 不再作为独立触发逻辑。跨范式探索统一由 EvolutionController 策略驱动。
            result = self.run_iteration(i, strategy=strategy)
            # update_iteration 与 _update_results_md 已在 run_iteration 内部自动执行

            # v7.3: feedback loop — if Agent failed before experiment, record failure to TreeController
            if result.get("agent_failure_mode") and self.tree_controller:
                leaf_id = strategy.get("target_leaf", "")
                if leaf_id:
                    self.tree_controller.record_failure(
                        leaf_id=leaf_id,
                        failure_mode=result["agent_failure_mode"],
                        iteration=i,
                        metadata={"strategy": strategy},
                    )

            # （depth 模式已废弃：原"深耕成功后自动提高门槛"逻辑随之移除，
            #  深耕现为 breadth 内 objective_deepen 生成路径）

            # 轮间休息，避免 API 限流
            if i < self.config.max_iterations:
                time.sleep(2)
        
        # [AsyncEval] 回收点 3：主循环结束后等待全部在途评测完成并 drain 记账
        self._async_wait_all()

        # 最终总结
        best = self.state_manager.state.get("best_result")
        self.logger.info("\n" + "="*60)
        self.logger.info("🎉 所有迭代完成！")
        if best:
            self.logger.info(f"🏆 最佳结果: 第 {best['iteration']} 轮, 准确率={best['metrics'].get('accuracy', 0):.4f}")
        else:
            self.logger.info("📭 本次运行未产生有效最佳结果（所有迭代均为失败/虚假成功/均匀回退）")
        self.logger.info(f"📊 详细日志保存在: {os.path.abspath(self.config.log_dir)}")
        self.logger.info("="*60)


# ==========================================
# 命令行入口
# ==========================================

def create_default_base_code(config: AutoResearchConfig):
    """备份原始代码文件（从配置读取目标路径，支持 Qwen 和 LLaVA）"""
    target_file = config.target_patch_file
    
    if not os.path.exists(target_file):
        print(f"❌ 错误: 找不到目标文件 {target_file}")
        print("请确保 MMTok 库已正确安装，并且配置中的 target_patch_file 路径正确。")
        sys.exit(1)
    
    # 备份原始文件
    backup_path = backup_original_code(target_file)
    
    if backup_path:
        print(f"✅ 已备份原始代码到: {backup_path}")
        print(f"📋 目标文件已确认: {target_file}")
        print("你可以运行 autoresearch_main.py 开始自动优化。")
    else:
        print("⚠️ 备份失败，但目标文件存在。建议手动备份后继续。")


def interactive_setup(config_path: str) -> AutoResearchConfig:
    """交互式配置向导"""
    config = AutoResearchConfig()
    
    print("\n" + "="*60)
    print("🔧 AutoResearch 配置向导")
    print("="*60)
    
    # API 配置 - Qwen Max (用于所有 LLM 操作)
    print("\n📡 Qwen Max API 配置 (Explorer Agent 使用):")
    qwen_key = input(f"请输入 Qwen API Key (默认已内置): ").strip()
    if qwen_key:
        config.qwen_api_key = qwen_key
    config.qwen_model_name = input(f"请输入 Qwen 模型名称 (默认: {config.qwen_model_name}): ").strip() or config.qwen_model_name
    
    print("\n🌙 Kimi API 配置 (Coder Agent 使用):")
    kimi_key = input(f"请输入 Kimi API Key (默认已内置): ").strip()
    if kimi_key:
        config.kimi_api_key = kimi_key
    config.kimi_model_name = input(f"请输入 Kimi 模型名称 (默认: {config.kimi_model_name}): ").strip() or config.kimi_model_name
    
    # 执行配置
    print("\n⚡ 执行配置:")
    config.eval_cmd = input("请输入评测命令 (如: CUDA_VISIBLE_DEVICES=0 python eval.py): ").strip()
    config.cuda_device = input(f"请输入 CUDA 设备号 (默认: {config.cuda_device}): ").strip() or config.cuda_device
    
    # 可选配置
    print("\n⚙️  可选配置 (直接回车使用默认值):")
    max_iter = input(f"最大迭代次数 (默认: {config.max_iterations}): ").strip()
    if max_iter:
        config.max_iterations = int(max_iter)
    
    # 保存
    config.to_file(config_path)
    print(f"\n💾 配置已保存到: {config_path}")
    print("\n📋 配置摘要:")
    print(f"   - Explorer LLM: Qwen ({config.qwen_model_name})")
    print(f"   - Coder LLM: Kimi ({config.kimi_model_name})")
    print(f"   - 功能: 搜索 + Idea 生成 + 代码生成")
    print(f"   - 允许搜索来源: SciPy/scikit-learn/PyTorch API, Wikidata/DBpedia, MathWorld")
    return config


def main():
    # 【OpenCLI 自动检测】若用户目录存在 node22，自动加入 PATH
    node22_bin = os.path.expanduser("~/node22/bin")
    if os.path.isdir(node22_bin) and node22_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = node22_bin + os.pathsep + os.environ.get("PATH", "")
        print(f"🔧 自动检测到 Node.js 22，已加入 PATH: {node22_bin}")
    
    parser = argparse.ArgumentParser(
        description="AutoResearch 引擎 - 自动化深度学习研究框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 首次运行（交互式配置）
  python autoresearch_main.py
  
  # 正常执行
  python autoresearch_main.py --config config.json
  
  # 断点续跑（自动）
  python autoresearch_main.py
  
  # 强制重置并重新开始
  python autoresearch_main.py --reset
  
  # 创建默认基础代码模板
  python autoresearch_main.py --init-base
        """
    )
    parser.add_argument("--config", type=str, default="autoresearch_config.json", 
                        help="配置文件路径 (默认: autoresearch_config.json)")
    parser.add_argument("--reset", action="store_true", 
                        help="重置状态，清除历史记录并重新开始")
    parser.add_argument("--init-base", action="store_true",
                        help="创建默认的基础代码模板文件")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="自动确认，非交互式模式")
    parser.add_argument("--debug-idea", type=str, default=None,
                        help="调试模式：直接加载指定的 idea 文件，跳过搜索，仅运行一轮代码生成与评测")
    parser.add_argument("--start-from", type=int, default=None,
                        help="从指定轮次开始运行（截断 state 到该轮次之前，保留所有日志文件）")
    
    args = parser.parse_args()
    
    # 加载或创建配置
    config = None
    if os.path.exists(args.config):
        config = AutoResearchConfig.from_file(args.config)
        if not args.yes:
            print(f"📋 已加载配置文件: {args.config}")
            print(f"   Explorer 模型: {config.qwen_model_name}")
            if getattr(config, 'coder_use_kimi', False):
                print(f"   Coder 模型: {config.kimi_model_name}")
            else:
                coder_model = getattr(config, 'coder_qwen_model_name', None) or config.qwen_model_name
                print(f"   Coder 模型: {coder_model}")
            print(f"   命令: {config.eval_cmd[:50]}..." if config.eval_cmd else "   命令: 未设置")
    else:
        if args.yes:
            print("❌ 错误: 非交互模式下配置文件必须存在")
            sys.exit(1)
        config = interactive_setup(args.config)
    
    # 初始化基础代码（需要 config 已加载）
    if args.init_base:
        create_default_base_code(config)
        return
    
    # 验证必要配置
    if not config.qwen_api_key:
        print("❌ 错误: Qwen API Key 未配置")
        sys.exit(1)
    if getattr(config, 'coder_use_kimi', False) and not config.kimi_api_key:
        print("❌ 错误: Kimi API Key 未配置（或关闭 coder_use_kimi）")
        sys.exit(1)
    if not config.eval_cmd:
        print("❌ 错误: 评测命令未配置")
        sys.exit(1)
    
    # 检查基础代码文件
    if not os.path.exists(config.base_code_file):
        print(f"❌ 错误: 找不到基础代码文件: {config.base_code_file}")
        print("请确保 MMTok 库已正确安装，并且文件路径正确。")
        sys.exit(1)
    else:
        # 检查并备份原始代码
        backup_marker = os.path.join(
            os.path.dirname(config.target_patch_file), 
            ".autoresearch_backups", 
            ".original_backed_up"
        )
        if not os.path.exists(backup_marker):
            print(f"📋 首次运行，备份原始代码...")
            backup_original_code(config.target_patch_file)
            # 创建标记文件
            os.makedirs(os.path.dirname(backup_marker), exist_ok=True)
            with open(backup_marker, 'w') as f:
                f.write(datetime.now().isoformat())
            print("✅ 备份完成，开始自动研究...")
    
    # 重置状态
    if args.reset:
        import glob
        import shutil
        
        if os.path.exists(config.state_file):
            os.remove(config.state_file)
            print("🗑️  已清除历史状态")
        
        # 同时恢复原始基础代码（如果备份存在）
        backup_dir = os.path.join(os.path.dirname(config.target_patch_file), ".autoresearch_backups")
        backup_files = sorted(glob.glob(os.path.join(backup_dir, "*.original")))
        if backup_files:
            shutil.copy2(backup_files[-1], config.target_patch_file)
            print(f"🔄 已恢复原始基础代码: {config.target_patch_file}")
        
        # 彻底重置：清理历史日志目录，防止旧历史影响新探索
        if os.path.exists(config.log_dir):
            shutil.rmtree(config.log_dir)
            print(f"🗑️  已清除历史日志目录: {config.log_dir}")
        
        # v3: 同时清除 candidate_pool 的 good_topics，避免 reset 后仍复用旧方向
        cache_dir = os.path.join(os.path.dirname(config.log_dir), "autoresearch_candidate_pool")
        good_topics_path = os.path.join(cache_dir, "good_topics.json")
        if os.path.exists(good_topics_path):
            os.remove(good_topics_path)
            print(f"🗑️  已清除历史 good_topics: {good_topics_path}")
    
    # 运行引擎
    try:
        engine = AutoResearchEngine(config, debug_idea_path=args.debug_idea, start_from=args.start_from)
        
        if args.start_from is not None and args.start_from > 1:
            target_iter = args.start_from
            print(f"🔄 已截断状态到 iteration {target_iter - 1}，将从 iteration {target_iter} 开始")
            print(f"📁 保留所有日志文件，iter_{target_iter:03d}/ 下的 idea 文件将触发断点续跑")
        
        engine.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断执行")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 致命错误: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
