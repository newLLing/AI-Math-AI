"""
AutoResearch 共享基础设施模块
包含配置、日志、LLM客户端、状态管理等通用组件
"""

import ast
import os
import re
import sys
import time
import json
import signal
import logging
import subprocess
import threading
import queue
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, asdict, field
from functools import wraps

# 导入限定域名搜索模块
from domain_restricted_search import DomainRestrictedSearch, create_domain_search


# ==========================================
# 全局统一阈值常量
# ==========================================
class ResearchThresholds:
    """全局统一的实验阈值常量，防止各组件定义不一致导致策略漂移"""
    MMTOK_BASELINE = 0.8522
    ORIGINAL_BASELINE = 0.8882
    UNIFORM_STEP_BASELINE = 0.8382
    DIRECTION_LOCK_THRESHOLD = 0.8550
    PRECISION_BAN_THRESHOLD = 0.8550
    BREAKTHROUGH_PROTECT_BASELINE = 0.8522
    # 注意：DEEPEN_THRESHOLD_BASE 是全局默认值。
    # 实际运行时会从 AutoResearchConfig.deepen_threshold_base 读取配置值覆盖。
    # IdeaHistoryTracker 优先使用 config 中的值，避免硬编码导致策略失效。
    DEEPEN_THRESHOLD_BASE = 0.8600
    DEEPEN_THRESHOLD_INCREMENT = 0.002
    EVOLUTION_SUCCESS_THRESHOLD = 0.8600
    META_ANALYZER_THRESHOLD = 0.8600
    FAKE_SUCCESS_TOLERANCE = 1e-4


# ==========================================
# 配置管理与数据类
# ==========================================

@dataclass
class AutoResearchConfig:
    """集中化配置管理"""
    # Qwen Max API 配置 (用于所有 LLM 操作)
    api_key: str = ""  # 向后兼容
    base_url: str = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    model_name: str = "qwen3.6-plus"
    
    qwen_api_key: str = field(default_factory=lambda: os.environ.get("AUTORESEARCH_QWEN_API_KEY", ""))
    qwen_base_url: str = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    qwen_model_name: str = "qwen3.6-plus"
    
    # Coder Agent 使用 Qwen 时的专用模型（可独立于 Explorer）
    coder_qwen_model_name: str = "qwen3.6-plus"
    
    # Kimi API 配置 (供 Coder Agent 使用)
    kimi_api_key: str = field(default_factory=lambda: os.environ.get("AUTORESEARCH_KIMI_API_KEY", ""))
    kimi_base_url: str = "https://api.moonshot.cn/v1"
    kimi_model_name: str = "kimi-k2-6"  # kimi-k2.6 在 API 中的标准拼写为 kimi-k2-6
    
    # Agent 分工配置
    coder_use_kimi: bool = False  # Coder Agent 是否使用 Kimi（False 则使用 Qwen）
    
    # 执行配置
    eval_cmd: str = ""
    cuda_device: str = "0"
    # GPU 自动选择（cuda_device="auto" 时生效）：
    # 单次评测所需的最小剩余显存（GiB），默认 30GB：在剩余显存 > 该门槛的 GPU 中
    # 优先选择剩余显存最多的卡；无卡满足时按 gpu_poll_interval 轮询等待；
    # gpu_wait_seconds > 0 时等待超过该时长后兜底使用当前最空闲的卡；
    # gpu_wait_seconds <= 0 时无限等待，直到有卡满足显存门槛。
    # min_free_gpu_gb 为 0 表示不做门槛检查。
    min_free_gpu_gb: float = 30.0
    gpu_wait_seconds: int = 600
    gpu_poll_interval: int = 60
    # 多 GPU 分支并行评测：parallel_branches > 0 时启用——当剩余显存
    # ≥ parallel_branch_min_free_gb 的 GPU 数 ≥ 2 时，一个迭代内并行跑
    # B = min(parallel_branches, 合格 GPU 数) 个分支（独立 idea + 独立代码 +
    # 并行评测，经 PYTHONPATH 影子包隔离分支代码）。0 = 关闭（默认，
    # 行为与原有单 idea 路径逐字一致）。
    parallel_branches: int = 0
    parallel_branch_min_free_gb: float = 30.0
    timeout_seconds: int = 1200
    max_iterations: int = 10
    
    # 路径配置
    log_dir: str = "./autoresearch_iterations"
    # 实际的 Qwen2.5-VL MMTok 模型文件路径
    # 注意：base_code_file 应该指向备份的原始代码，target_patch_file 是实际修改的目标
    base_code_file: str = "/home/anonymous/code/MMTok/mmtok/qwen/.autoresearch_backups/qwen2_5_VLmodel_mmtok.py.original"
    target_patch_file: str = "/home/anonymous/code/MMTok/mmtok/qwen/qwen2_5_VLmodel_mmtok.py"
    # patch_mode: "model" 表示直接修改 model forward 方法，"selector" 表示修改 autoresearch_selector.py
    patch_mode: str = "model"
    state_file: str = "./autoresearch_state.json"
    # 隔离配置：candidate_pool 和 state 的存储目录（支持多模型并行/隔离）
    candidate_pool_dir: str = "./autoresearch_candidate_pool"
    state_dir: str = "./autoresearch_state"
    # 重试配置
    max_api_retries: int = 3
    api_retry_delay: int = 5
    eval_retry_attempts: int = 2
    
    # LLM 参数
    explorer_temperature: float = 0.7
    coder_temperature: float = 0.3  # 适度提高以获得更多创造性解决方案
    max_tokens: int = 16384
    deepen_code_truncation: int = 8000  # 深度探索时父本代码截断长度，0表示不截断（传入完整代码）
    
    # 代码生成策略
    code_gen_max_retries: int = 2  # 代码生成失败时的最大重试次数
    code_gen_temperature_fallback: float = 0.1  # 重试时的温度（更低，更保守）
    code_gen_max_repairs: int = 2  # 同一策略内基于验证错误的自修复最大轮次
    code_gen_enable_critic: bool = False  # 是否启用 LLM Critic 二次审查
    
    # 评测执行策略
    eval_stream_kill_on_error: bool = True  # 检测到关键错误时是否提前终止评测进程

    # 断点续跑历史重建（IdeaHistoryTracker._parse_metrics_from_log）的旧解析 bug 兼容开关：
    # True（默认）== 旧行为——内联正则从 exact_match 起消耗 4 列后捕获，实际抓到
    # lmms-eval 表格末列（stderr）数值而非 accuracy 主值，并无条件覆盖 parse_basic_metrics
    # 已解析对的值；False 时委托 current_task().get_metric_parser() 的表格正则取正确主值。
    history_log_metric_stderr_bug_compat: bool = True
    
    # 模型族配置（支持多模型 autoresearch）
    model_family: str = "qwen"                    # "qwen" | "llava"
    # 任务 profile（Phase 4）：autoresearch_tasks 中注册的任务名；
    # 空字符串 = 按 model_family 映射到默认 profile（向后兼容现有全部配置）
    task_profile: str = ""
    # elite archive 路径覆盖（Phase F3）：空 = 按 model_family 推导现默认路径（行为不变）
    elite_archive_path: str = ""
    main_tensor_name: str = "image_embeds"        # 主参考 tensor 名（用于静态检查和 prompt）
    device_tensor_expr: str = "image_embeds.device"  # device 表达式（用于确定性修复）
    has_retain_ratio: bool = True                 # 是否通过 retain_ratio 计算 target_vision_tokens
    original_baseline: float = 0.8882             # 原始未压缩模型 baseline（可从配置覆盖）
    mmtok_baseline: float = 0.8522                # MMTok baseline（可从配置覆盖）
    uniform_step_baseline: float = 0.8382         # 均匀步长 fallback baseline（可从配置覆盖）
    deepen_threshold_base: float = 0.86           # 深度探索门槛基准值（可从配置覆盖）
    # cross Fusion Round 门槛（可从配置覆盖）。None = 用 profile baselines
    # （qwen 默认 0.86，系 LLaVA 0.8x 时代残留；低分任务必须显式覆盖，
    # 否则 qualified 父本恒 < 2、cross 融合永不触发）
    fusion_threshold: float = 0.0
    
    # Google Custom Search API 配置 (可选，用于更精确的限定域名搜索)
    google_api_key: str = ""  # Google Custom Search API Key
    google_cx: str = ""  # Custom Search Engine ID
    
    # 批量 Idea 生成配置
    enable_batch_ideas: bool = True  # 是否启用一次召回生成3个idea的批量模式
    
    # 新增：历史蒸馏与重启配置
    seed_history_file: str = ""  # 蒸馏历史文件路径
    intra_iteration_branches: int = 1  # 单迭代内探索分支数
    enable_fingerprint: bool = True  # 是否启用指纹预跑
    
    # v7.0 新架构配置（实验性，默认关闭以保持向后兼容）
    use_new_architecture: bool = False  # 启用 ExplorationOrchestrator + SubAgent 架构
    enable_unified_state_store: bool = False  # 启用 SQLite 统一状态存储
    
    # v8.0: 强制探索模式 ("depth" | "breadth" | "cross" | "")
    force_mode: str = ""  # 空字符串表示不强制，由 TreeController 自动决策

    # v11.1: Wikipedia 完整定义召回（breadth/depth 共用，经 api.wikimedia.org）
    enable_wiki_definition_recall: bool = True
    wiki_definition_max_chars: int = 6000  # 单个词条定义最大字符数
    wiki_definition_max_pages: int = 2     # 每次召回最多几个词条

    # v11.2: LLM 理论召回（用当前 LLM 生成数学 topic 的定义/定理/性质完整介绍，与论文摘要互补）
    enable_llm_theory_recall: bool = True
    llm_theory_max_chars: int = 6000     # 单次理论简介最大字符数
    llm_theory_temperature: float = 0.3  # 低温以保证准确性
    llm_theory_model_name: str = ""      # 理论简介专用模型（空 = 用系统默认 qwen_model_name）

    # CrossRef 数学期刊源（无摘要噪声大，默认关闭）
    enable_math_journal_recall: bool = False

    # Cell-first 复合 idea：每 N 轮从未充分覆盖 cell 中选 2 个异 domain 的 cell，
    # 分别召回知识、融合后生成同时实例化两个 (domain, mechanism) 的融合算法；
    # judge 分别校验两个 cell。默认关闭，关闭时零行为变化。
    enable_composite_ideas: bool = False
    composite_idea_every_n: int = 3        # 每 N 轮迭代做一次复合；1 = 每次都复合

    # v11.0: 演进算子驱动的深度探索（Evolution-Move Deep Exploration）
    # 设计文档: docs/depth_prompt_schemes.md 方案一
    depth_strategy: str = "fusion"  # "fusion"（原机制融合）| "evolution"（演进算子驱动）
    depth_evolution_num_candidates: int = 3  # 每轮生成的演进草案数（每个草案对应一个演进算子）
    # 人类决策由人类主动发起（非阻塞）：主循环永不等待；人类停止 autoresearch 后用
    # autoresearch_evolution_gate.py 调整 pin/ban/policy（写入 <state_dir>/evolution_decisions.json），
    # 重启后在后续 depth 迭代生效。本项仅是无人类决策时的默认值，CLI 侧设置优先。
    evolution_default_policy: str = "first"  # "first" | "random"
    # v10.0 行为开关: llava/qwen 下强制 breadth-only（禁用 depth/cross 通路）。
    # 置 False 可恢复 depth/cross（evolution 深度探索依赖 depth 通路）。
    force_breadth_only: bool = True

    # ======== 消融实验开关 ========
    # "formulation"（默认，数学重构搜索）| "direct"（A1 消融臂：无重构、
    # 直接给最佳代码+分数做代码变异，FunSearch 式）| "funsearch"（F1 对照臂：
    # FunSearch 岛模型进化）| "eoh"（F2 对照臂：EoH thought+code 稳态进化）
    explorer_mode: str = "formulation"
    # A2 消融臂：关闭机制级多样性机制（算子签名去重、机制饱和闸门、行为克隆闸门、
    # 原型强制轮转、cell 钉选、方向/机制配额），回到自由生成
    disable_diversity_mechanisms: bool = False
    # 消融臂：关闭 idea 相似度去重（Jaccard 0.72 / LCS 0.75）
    disable_dedup: bool = False
    # 消融臂：关闭饱和规则与 35% 窗口配额（方向/机制簇配额、
    # success/failure 饱和、原型饱和）
    disable_saturation: bool = False
    # 消融臂：关闭 Explorer 的理论知识检索（保留 formulation 结构）
    disable_knowledge_recall: bool = False
    # 消融臂：关闭 Engine 返回给 Explorer 的 history_feedback 文本反馈
    disable_score_feedback: bool = False
    # 消融臂（严格 Score guide 臂配套）：策略选择去分数化——TreeController 不再用
    # UCT/Q 值选 leaf 与 mode，改为均匀随机（leaf 随机、mode 在 breadth/cross 间均匀）
    random_strategy_selection: bool = False
    # E2 消融臂：去掉 Explorer prompt 的强制数学论证段，目标描述只需
    # concise objective / scoring rule（与 direct-mutation 基线对齐）；
    # PRIMARY_DIRECTION 声明与 retain_ratio 禁令保留
    formulation_lite: bool = False
    # E3 消融臂：去掉人工先验（目标原型 archetype 注入/轮转、(domain, mechanism)
    # cell 钉选）；idea 去重、饱和/配额、PRIMARY_DIRECTION 记账、
    # anti_deadlock_guidance 全部保留
    disable_priors: bool = False

    # ======== F1/F2 对照臂：进化式搜索（explorer_mode = "funsearch" | "eoh"）========
    # FunSearchDB（Romera-Paredes et al. Nature 2024 小规模版）：island 模型 +
    # best-shot exemplar prompting + 周期岛重置；EoHPopulation（Liu et al.
    # ICML 2024 稳态版）：(thought, code) 个体 + E1/E2/M1/M2/M3 算子。
    funsearch_islands: int = 4              # island 数
    funsearch_island_size: int = 20         # 每岛最多保留程序数（淘汰最低分）
    funsearch_exemplars: int = 2            # prompt 中岛内 top-N exemplars
    funsearch_reset_every_evals: int = 40   # 每 N 次评测将最低分岛替换为最高分岛副本
    eoh_population_size: int = 8            # EoH 稳态种群大小（前 pop_size-1 轮为初始化）

    # ======== v12.0: Research Notebook（depth-as-learning）========
    # depth 模式重建为学习闭环：选问题→立假设→受控实验→对账→沉淀原理。
    enable_learning_depth: bool = True        # 总开关；False 时 depth 完全走旧管线
    depth_budget: float = 0.5                 # depth 预算占比（学习需要样本量）
    notebook_brief_max_tokens: int = 600      # depth prompt 注入知识摘要的硬预算
    notebook_distill_interval: int = 20       # 蒸馏间隔（轮）
    notebook_noise_floor: float = 0.01        # |Δacc| 小于此值视为 maintain/inconclusive
    notebook_contradiction_threshold: float = 0.1  # 同机制签名分差超过此值派生矛盾声称
    notebook_high_score_sigma: float = 3.0    # 超过全局均值+Nσ 的单点高分派生待复现声称
    notebook_require_verified_for_elite: bool = True  # n=1 高分离群未复现前不进父本池/elite

    # v12.1: 运行时 token 数探针（防动态超选 hack，iter_461 CAWR 事件）
    enable_token_count_probe: bool = True     # 评测前用合成输入实测 selector 保留数
    probe_retain_ratio: float = 0.05          # 探针使用的 retain_ratio（应与评测一致）
    # v15: 探针子进程隔离 + 超时（2026-08-02 事故：生成算法调 torch.lobpcg 不收敛，
    # 探针在主线程同步执行无超时，引擎卡死 9+ 小时）。探针改为子进程执行，
    # 超过该时长 kill 进程组并判定探针失败（走现有 pre-eval 失败/修复路径）。
    probe_timeout_seconds: int = 180

    # Phase I1: 张量组合自适应探索总开关（默认 False == 现行为逐字一致）。
    # 开启后：UCB1 bandit 在 7 种输入张量组合间分配迭代（ComboAllocator），
    # 每组合独立 archetype 轮转，低价值组合目标自动修改，explorer prompt
    # 注入组合约束块替代 usage_mode 块；召回/topic 机制不变。
    enable_tensor_combo_exploration: bool = False

    # Phase 3: 组合深耕开关（enable_tensor_combo_exploration 关闭时整体不生效）。
    # 开启后：探索期结束的高 success-rate (combo, archetype) 对被 Engine 标记为
    # objective_deepen 迭代（全程 breadth 模式，无父本；对该对的优化目标本身做
    # 深度推理演进），并在利用期独占 profile.combo_depth_share 的目标份额。
    enable_combo_deepen: bool = True

    # Phase A: 每个 (combo, archetype) 对的固定探索轮数（合法区间 20-30，
    # 越界截断并告警；0 = 回落 profile.combo_objective_min_trials / 自动折算）。
    combo_min_trials: int = 25

    # objective_feedback.compute_empirical_accuracy 的归一化 scale：
    # normalized = clip((acc - mmtok_baseline) / scale, 0, 1)。Engine 启动时
    # 随 config.mmtok_baseline 一并覆盖到 profile baselines，统一 UCT 价值口径。
    empirical_accuracy_scale: float = 0.08

    # Topic 复用限制: 同一个 topic 最多被选中几次 (1=不允许复用, 2=允许1次复用, ...)
    topic_max_reuse_count: int = 2  # 默认允许复用 1 次
    
    # 高分 topic 阈值: 超过此 accuracy 的 topic 会被加入 good_topics 池用于复用
    # 默认使用 deepen_threshold_base (0.33)，可单独覆盖
    good_topic_acc_threshold: float = 0.0  # 0 表示自动使用 deepen_threshold_base
    
    # v9.0: 禁用高分模板注入（Coder Agent 不再接收历史高分算法的完整代码）
    disable_high_score_template_injection: bool = False

    # G1: accuracy==0 ⇒ “算法崩溃/未生效”判定开关（默认 True == 现行为）；
    # 新任务指标语义不同（如 0 为合法取值）时可关闭
    enable_zero_metric_crash_detection: bool = True
    # G1: 均匀步长 fallback 指纹检测开关（默认 True == 现行为）
    enable_uniform_fallback_detection: bool = True
    # G2: 执行后保留比例审查（post-exec tampered audit）开关（默认 True == 现行为）
    enable_tamper_audit: bool = True

    # v13: 冒烟模式（默认 False == 现行为逐字一致，所有新逻辑门控于该开关）。
    # 开启后：代码生成后做 "GPU 显存闸 + 真实 GPU 冒烟跑通"（真实实例化 patch 后的
    # selector、合成输入 forward、校验输出 token 数与维度），通过即进下一轮，
    # 跳过正式评测（eval_cmd）。探索信号由机制覆盖统计承担，本模式不依赖 acc。
    smoke_only_mode: bool = False
    # 冒烟前显存闸：至少一张卡空闲显存 >= 该值（GiB）才跑冒烟；
    # 无卡满足时按 gpu_poll_interval 轮询等待（gpu_wait_seconds<=0 表示无限等待）。
    smoke_min_free_gpu_gb: int = 30

    # v14: 异步评测流水线（默认 False == 现行为逐字一致）。
    # 生效条件：async_eval_mode 且非 smoke_only_mode 且 parallel_branches<=0。
    # 开启后：单分支串行模式下 idea+code 生成完成、pre-eval 管线通过后，
    # 把 lmms_eval 评测交给后台 worker 线程（快照环境隔离代码），主循环不等
    # 评测跑完立即开始下一 iteration 的生成；评测完成后由主线程 drain 回收，
    # 用真实 acc 补全全部记账（finalize 路径）。
    async_eval_mode: bool = False
    # 同时在途评测上限：在途数达到该值时主循环阻塞等槽位（poll + drain）。
    max_async_evals: int = 2
    # 单个在途评测的硬超时（秒）：0 = 自动（gpu_wait_seconds(<=0 按 3600 计)
    # + timeout_seconds * (eval_retry_attempts + 1) + 900）。超过该时长仍未完成的
    # 在途评测被主线程收割：按失败记账（async_hard_timeout），其迟到结果回收时丢弃，
    # 防止 worker 卡死（如无限等 GPU）拖死主流程。
    async_eval_hard_timeout_seconds: int = 0
    # 槽位等待期间预生产（look-ahead）：True 时，_async_wait_slot 阻塞等 GPU 槽位
    # 期间主线程提前生成下一轮 iteration 的 idea+code，槽位释放后秒级提交评测，
    # 消除评测间的 LLM 串行空窗（生成仍在主线程，无额外线程/锁）。
    enable_lookahead_production: bool = True

    # LLM 调用硬预算（防 API 挂起拖死主流程；历史上 idea 生成曾静默挂起 9+ 小时）：
    # llm_request_timeout_seconds 为单次请求超时；llm_max_total_wait_seconds 为
    # 带重试调用的总预算（含各次尝试耗时与退避等待），超预算抛 LLMHardTimeoutError，
    # 由 explorer/coder 的兜底逻辑按 agent 失败处理，主循环继续。
    llm_request_timeout_seconds: int = 600
    llm_max_total_wait_seconds: int = 1800

    def __post_init__(self):
        """配置校验。"""
        # combo_min_trials 合法区间 20-30，越界截断
        if self.combo_min_trials != 0 and not (20 <= self.combo_min_trials <= 30):
            _clamped = min(max(self.combo_min_trials, 20), 30)
            print(
                f"⚠️  combo_min_trials={self.combo_min_trials} 超出合法区间 [20, 30]，"
                f"已截断为 {_clamped}"
            )
            self.combo_min_trials = _clamped
        # depth 相关配置仅在 breadth-only（force_breadth_only=True，depth/cross
        # 通路关闭）时不生效——此时深耕以 breadth 内 objective_deepen 方式完成。
        # force_breadth_only=False 时 depth 管线（含 learning 闭环）正常工作，
        # 不再打印这些（已过时的）废弃警告。
        if self.force_breadth_only and self.depth_strategy != "fusion":
            print(
                f"⚠️  depth_strategy='{self.depth_strategy}' 不生效：当前为 breadth-only "
                "（force_breadth_only=true），深耕以 breadth 内 objective_deepen 方式完成。"
            )
        if self.force_breadth_only and self.enable_learning_depth:
            print(
                "ℹ️  enable_learning_depth 不生效：当前为 breadth-only "
                "（force_breadth_only=true），learning 管线不会进入。"
            )
        if self.enable_combo_deepen and not self.enable_tensor_combo_exploration:
            print(
                "⚠️  enable_combo_deepen=true 但 enable_tensor_combo_exploration=false："
                "组合深耕不生效（无组合探索则无深耕对）。"
            )

    @classmethod
    def from_file(cls, config_path: str) -> "AutoResearchConfig":
        """从 JSON 配置文件加载"""
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 过滤掉不存在的字段
                    valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
                    return cls(**valid_fields)
            except Exception as e:
                print(f"⚠️  配置文件加载失败: {e}，使用默认配置")
        return cls()
    
    def to_file(self, config_path: str):
        """保存配置到文件"""
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)


def derive_elite_archive_path(config: Optional["AutoResearchConfig"], base_dir: str) -> str:
    """统一推导 elite archive 路径（Phase F3）。

    config.elite_archive_path 非空时优先（任务自定义路径）；
    否则按 model_family 推导现默认路径 <base_dir>/elite_algorithms_{model_family}.json。
    各调用点的 base_dir 保持其原有目录，行为不变。
    """
    override = getattr(config, 'elite_archive_path', '') if config is not None else ''
    if override:
        return override
    model_family = getattr(config, 'model_family', 'qwen') if config is not None else 'qwen'
    return os.path.join(base_dir, f"elite_algorithms_{model_family}.json")


# ==========================================
# 日志系统配置
# ==========================================

def setup_logging(log_dir: str, iteration: int) -> logging.Logger:
    """设置分级日志系统，同时输出到控制台和文件"""
    logger = logging.getLogger(f"AutoResearch_{iteration}")
    logger.setLevel(logging.DEBUG)
    
    # 清除旧处理器
    logger.handlers.clear()
    
    # 格式化器
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器 (INFO 级别)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器 (DEBUG 级别)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"system_{iteration:03d}.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


# ==========================================
# 状态持久化管理器
# ==========================================

class StateManager:
    """管理实验状态，支持断点续跑"""
    
    def __init__(self, state_file: str, logger: logging.Logger,
                 config: Optional["AutoResearchConfig"] = None):
        self.state_file = state_file
        self.logger = logger
        self.config = config
        self.state = self._load_state()
    
    def _load_state(self) -> Dict[str, Any]:
        """加载状态文件"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    self.logger.info(f"✅ 已加载历史状态，上次跑到第 {state.get('last_iteration', 0)} 轮")
                    return state
            except (json.JSONDecodeError, IOError) as e:
                self.logger.error(f"⚠️ 状态文件损坏: {e}，将重新开始")
        return self._init_state()
    
    def _init_state(self) -> Dict[str, Any]:
        return {
            "last_iteration": 0,
            "completed_iterations": [],
            "best_result": None,
            "history_feedback": "",
            "start_time": datetime.now().isoformat()
        }
    
    def save(self):
        """原子性保存状态"""
        temp_file = f"{self.state_file}.tmp"
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
            os.replace(temp_file, self.state_file)
            self.logger.debug(f"💾 状态已保存")
        except Exception as e:
            self.logger.error(f"❌ 保存状态失败: {e}")
    
    def update_iteration(self, iteration: int, result_data: Dict[str, Any]):
        """更新当前迭代进度，并保存足够元数据以支持断点续跑时的历史重建"""
        self.state["last_iteration"] = iteration
        entry = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "success": result_data.get("success", False),
            "metrics": result_data.get("metrics", {}),
            "is_fake_success": result_data.get("is_fake_success", False),
            "is_uniform_fallback": result_data.get("is_uniform_fallback", False),
            "is_tampered_ratio": result_data.get("is_tampered_ratio", False),
            "direction": result_data.get("direction", "general"),
            "approach_type": result_data.get("approach_type", "Unknown"),
            "operators": result_data.get("operators", []),
            "idea_summary": (result_data.get("idea", "")[:500] if result_data.get("idea") else ""),
            "topic": result_data.get("topic"),
            # 目标原型（breadth 轮；用于原型级饱和降权的历史归因与断点重建）
            "archetype": result_data.get("archetype", ""),
            "feedback": result_data.get("feedback", "")[:1000],
        }
        # 行为指纹（功能克隆查重；仅探针通过的轮次携带）
        if result_data.get("selection_signature"):
            entry["selection_signature"] = result_data["selection_signature"]
        # Phase I2: 张量组合（仅组合探索开启时 result 才携带该字段；
        # 关闭时 state entry 不新增键，断点续跑重建路径容忍缺失）
        if result_data.get("tensor_combo"):
            entry["tensor_combo"] = result_data["tensor_combo"]
        # [SmokeMode] 冒烟标记透传（results.md 渲染 smoke✓/smoke✗ 依赖该键）；
        # result 顶层缺失时从 metrics 兜底（smoke 分支 metrics={"smoke_passed": bool}）
        _smoke = result_data.get("smoke_passed")
        if _smoke is None:
            _smoke = (result_data.get("metrics") or {}).get("smoke_passed")
        if _smoke is not None:
            entry["smoke_passed"] = _smoke
        # 分支并行评测（仅并行路径的 result 携带；单路径不新增键）
        if result_data.get("parallel_branches_used"):
            entry["parallel_branches_used"] = result_data["parallel_branches_used"]
        if result_data.get("branch_results"):
            entry["branch_results"] = result_data["branch_results"]
        # 分支独立记录标识（results.md 的 Iter 列显示为 "迭代-分支"）
        if result_data.get("branch") is not None:
            entry["branch"] = result_data["branch"]
        self.state["completed_iterations"].append(entry)
        # 保留最佳结果
        if result_data.get("success"):
            from autoresearch_tasks import current_task
            current_acc = current_task().extract_primary_metric(result_data.get("metrics", {}))
            best_acc = current_task().extract_primary_metric(self.state["best_result"].get("metrics", {})) if self.state["best_result"] else 0
            if current_acc > best_acc:
                self.state["best_result"] = result_data
                self.logger.info(f"🏆 新的最佳结果！准确率: {current_acc}")
        
        # 消融开关：关闭 Engine 返回给 Explorer 的 history_feedback 文本反馈
        # （跳过失败窗口维护并直接置空；其余记账逻辑——score、elite、tree 等——不受影响）
        if getattr(self.config, 'disable_score_feedback', False):
            self.state["history_feedback"] = ""
            self.save()
            return

        # Minimal history injection: only keep recent failure patterns (max 3 rounds)
        # Suppress success logs to avoid positive reinforcement locking
        feedback = result_data.get("feedback", "")
        iteration = result_data.get("iteration", 0)
        is_failure = (
            not result_data.get("success", False)
            or "Fake" in feedback
            or "Tampered" in feedback
            or "Uniform Step Fallback" in feedback
            or "Zero Accuracy" in feedback
        )
        if is_failure and feedback:
            window = self.state.get("history_feedback_window", [])
            window.append(f"[Iter {iteration}] {feedback[:800]}")
            self.state["history_feedback_window"] = window[-3:]
            self.state["history_feedback"] = "\n\n[Recent Failure Patterns - Learn from these]\n" + "\n\n".join(self.state["history_feedback_window"])
        else:
            # For successful real compression: do NOT inject detailed logs.
            # Only keep a minimal marker to indicate the system is running.
            self.state["history_feedback"] = f"[Iter {iteration}: Success] No recent failures to report."
        self.save()


# ==========================================
# 重试装饰器与错误处理
# ==========================================

def _is_authentication_error(exc: Exception) -> bool:
    """判断是否为认证/鉴权类错误（HTTP 401/403）。

    此类错误（API key 缺失/无效/无权限）重试不会成功，应立即失败。
    依次按：status_code 属性 -> openai 异常类型 -> 错误消息关键字 判定。
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in (401, 403):
        return True
    try:
        import openai  # 延迟导入：openai 仅在实际调用 LLM 时才需要
        if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
            return True
    except ImportError:
        pass
    err_lower = str(exc).lower()
    return any(kw in err_lower for kw in (
        "error code: 401", "error code: 403",
        "401 unauthorized", "403 forbidden",
        "invalid api key", "invalid_api_key", "incorrect api key",
        "authentication error", "permission denied",
    ))


def retry_with_backoff(max_retries: int, base_delay: int, logger: Optional[logging.Logger] = None,
                         max_total_wait: int = 3600, server_error_delay: int = 30,
                         max_total_wait_attr: Optional[str] = None):
    """指数退避重试装饰器
    
    Args:
        max_retries: 最大重试次数
        base_delay: 普通错误的基础延迟（秒）
        logger: 日志器
        max_total_wait: 最大总等待时间（秒），默认3600
        server_error_delay: 服务端错误（503/500等）的基础延迟（秒），默认30
        max_total_wait_attr: 可选——被装饰方法所属实例 config 上的属性名
            （如 'llm_max_total_wait_seconds'），运行时从实例读取以覆盖
            max_total_wait，使总预算可配置。
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            log = logger or logging.getLogger("AutoResearch")
            # 运行时解析总预算：优先实例 config 上的指定属性
            effective_max_total_wait = max_total_wait
            if max_total_wait_attr and args:
                cfg = getattr(args[0], 'config', None)
                if cfg is not None:
                    effective_max_total_wait = int(
                        getattr(cfg, max_total_wait_attr, max_total_wait) or max_total_wait)
            t_start = time.time()
            total_waited = 0
            for attempt in range(max_retries):
                # 墙钟硬预算：历次尝试耗时 + 退避等待超过预算即终止
                # （单次进行中的调用无法打断，由客户端 request timeout 兜底）
                if attempt > 0 and time.time() - t_start > effective_max_total_wait:
                    log.error(
                        f"❌ {func.__name__} 超过总时间预算 {effective_max_total_wait}s"
                        f"（墙钟），终止重试")
                    raise LLMHardTimeoutError(
                        f"{func.__name__} 超过 LLM 总时间预算 "
                        f"{effective_max_total_wait}s（含各次尝试与退避等待）")
                try:
                    return func(*args, **kwargs)
                except LLMAuthenticationError:
                    raise  # 已包装的认证错误直接上抛，避免重复包装
                except Exception as e:
                    err_str = str(e)
                    err_lower = err_str.lower()
                    # 认证/鉴权类错误（HTTP 401/403）重试无意义，立即快速失败
                    if _is_authentication_error(e):
                        log.error(f"❌ {func.__name__} 认证失败（HTTP 401/403），不重试: {err_str[:200]}")
                        raise LLMAuthenticationError(
                            "LLM API 认证/鉴权失败（HTTP 401/403），请检查 API key 配置"
                            "（AUTORESEARCH_QWEN_API_KEY / AUTORESEARCH_KIMI_API_KEY 环境变量）。"
                            f"原始错误: {err_str[:300]}"
                        ) from e
                    # 判断是否为服务端暂时不可用错误
                    is_server_error = (
                        isinstance(e, ServerUnavailableError)
                        or any(code in err_lower for code in ["503", "500", "502", "504", "internalerror.algo"])
                        or any(kw in err_lower for kw in ["service unavailable", "internal error", "algo error", "model serving"])
                    )
                    
                    if attempt == max_retries - 1:
                        log.error(f"❌ {func.__name__} 最终失败 (已重试 {attempt+1} 次): {err_str[:200]}")
                        raise
                    
                    if is_server_error:
                        delay = server_error_delay * (2 ** attempt)
                        log.warning(f"⚠️  {func.__name__} 第 {attempt+1}/{max_retries} 次失败 [服务端错误]: {err_str[:150]}，{delay}秒后重试...")
                    else:
                        delay = base_delay * (2 ** attempt)
                        log.warning(f"⚠️  {func.__name__} 第 {attempt+1}/{max_retries} 次失败: {err_str[:150]}，{delay}秒后重试...")
                    
                    # 确保不超过总等待时间上限
                    if total_waited + delay > effective_max_total_wait:
                        remaining = max(0, effective_max_total_wait - total_waited)
                        if remaining > 0:
                            log.warning(f"⏱️  调整延迟为 {remaining} 秒以遵守总时间上限 {effective_max_total_wait}s")
                            time.sleep(remaining)
                        log.error(f"❌ {func.__name__} 达到总等待时间上限 {effective_max_total_wait}s，终止重试")
                        raise LLMHardTimeoutError(
                            f"{func.__name__} 达到 LLM 总等待时间上限 "
                            f"{effective_max_total_wait}s，终止重试") from e
                    
                    total_waited += delay
                    time.sleep(delay)
            return None
        return wrapper
    return decorator


class APIError(Exception):
    """自定义 API 错误"""
    pass


class ContextOverflowError(APIError):
    """上下文长度溢出错误"""
    pass


class ServerUnavailableError(APIError):
    """服务端暂时不可用错误（如 503）"""
    pass


class LLMAuthenticationError(APIError):
    """认证/鉴权类错误（HTTP 401/403）：API key 缺失/无效/无权限，重试无意义"""
    pass


class LLMHardTimeoutError(APIError):
    """LLM 调用超过硬时间预算（retry_with_backoff 墙钟/等待上限）：
    不再重试，交由上层按 agent 失败记账，防止 API 挂起拖死主流程。"""
    pass


# ==========================================
# 鲁棒的 LLM 客户端
# ==========================================

class RobustLLMClient:
    """带重试、限流处理、验证的 LLM 客户端"""
    
    # 定义允许搜索的域名白名单
    ALLOWED_SEARCH_DOMAINS = [
        # 科学计算库官方文档
        "docs.scipy.org",
        "scikit-learn.org",
        "scikit-learn.org/stable",
        "pytorch.org",
        "pytorch.org/docs",
        "pytorch.org/tutorials",
        "numpy.org",
        "numpy.org/doc",
        # 学术知识图谱
        "wikidata.org",
        "dbpedia.org",
        # Wolfram MathWorld
        "mathworld.wolfram.com",
        # 学术论文
        "paperswithcode.com",
        # Python 文档
        "docs.python.org",
    ]
    
    def __init__(self, config: AutoResearchConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.client = None
        self.qwen_client = None
        self.domain_searcher = None
        self._init_clients()
        self._init_domain_searcher()
    
    def _init_clients(self):
        """延迟初始化客户端"""
        try:
            from openai import OpenAI
            
            # 通用客户端 (向后兼容)
            self.client = OpenAI(
                api_key=self.config.qwen_api_key,
                base_url=self.config.qwen_base_url,
                timeout=60.0,
                max_retries=2
            )
            
            # Qwen Max 客户端 (用于所有操作)
            # 增加超时时间到 600 秒，因为 thinking 模式需要更长时间
            self.qwen_client = OpenAI(
                api_key=self.config.qwen_api_key,
                base_url=self.config.qwen_base_url,
                timeout=float(getattr(self.config, 'llm_request_timeout_seconds', 600)),
                max_retries=2
            )
            
            # Kimi 客户端 (供 Coder Agent 使用)
            self.kimi_client = OpenAI(
                api_key=self.config.kimi_api_key,
                base_url=self.config.kimi_base_url,
                timeout=float(getattr(self.config, 'llm_request_timeout_seconds', 600)),
                max_retries=2
            )
            
            self.logger.info("✅ LLM 客户端初始化成功 (Qwen + Kimi)")
        except ImportError:
            self.logger.error("❌ 请先安装 openai 库: pip install openai")
            raise
        except Exception as e:
            self.logger.error(f"❌ LLM 客户端初始化失败: {e}")
            raise
    
    def _init_domain_searcher(self):
        """初始化限定域名搜索器"""
        # 检查是否有 Google Custom Search API 配置
        google_api_key = getattr(self.config, 'google_api_key', None)
        google_cx = getattr(self.config, 'google_cx', None)

        try:
            self.domain_searcher = create_domain_search(
                allowed_domains=self.ALLOWED_SEARCH_DOMAINS,
                google_api_key=google_api_key,
                google_cx=google_cx,
                logger=self.logger
            )
            self.logger.info(f"✅ 限定域名搜索器初始化成功，允许域名: {len(self.ALLOWED_SEARCH_DOMAINS)} 个")
        except Exception as e:
            self.logger.warning(f"⚠️ 限定域名搜索器初始化失败: {e}，搜索功能将不可用")
            self.domain_searcher = None
    
    def generate(self, prompt: str, temperature: float, system_msg: str, return_raw: bool = False, enable_thinking: bool = True, min_response_length: int = 50, model_name: str = None):
        """带完整错误处理的生成方法 (使用 Qwen)
        
        Args:
            prompt: 提示词
            temperature: 温度参数
            system_msg: 系统消息
            return_raw: 是否返回原始响应数据
            enable_thinking: 是否启用 Qwen thinking 模式
            min_response_length: 响应内容最小长度要求（默认50，短回答场景可设为1）
            model_name: 指定模型名称（默认使用 config.qwen_model_name）
            
        Returns:
            如果 return_raw=False: 返回生成的内容字符串
            如果 return_raw=True: 返回 (生成的内容, 原始响应字典)
        """
        return self._generate_with_qwen(prompt, temperature, system_msg, return_raw, enable_thinking, min_response_length, model_name)
    
    @retry_with_backoff(10, 5, max_total_wait=3600, server_error_delay=30,
                        max_total_wait_attr='llm_max_total_wait_seconds')
    def _generate_with_qwen(self, prompt: str, temperature: float, system_msg: str, return_raw: bool = False, enable_thinking: bool = True, min_response_length: int = 50, model_name: str = None):
        """使用 Qwen 生成"""
        import time
        model = model_name or self.config.qwen_model_name
        # 安全上限：防止配置值超过 API 限制（deepseek 系列最大 393216，其他模型通常更小）
        max_tokens = min(self.config.max_tokens, 393216)
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if enable_thinking is not None:
            kwargs["extra_body"] = {"enable_thinking": enable_thinking}
        
        self.logger.info(
            f"⏳ [LLM]  请求开始 | model={model} | thinking={enable_thinking} | "
            f"temp={temperature} | prompt_len={len(prompt)} | max_tokens={max_tokens}"
        )
        t0 = time.time()
        try:
            response = self.qwen_client.chat.completions.create(**kwargs)
        except Exception as e:
            elapsed = time.time() - t0
            err_msg = str(e).lower()
            self.logger.warning(f"❌ [LLM] Qwen 请求失败 (耗时 {elapsed:.1f}s): {e}")
            if any(k in err_msg for k in ["context length", "too many tokens", "token limit", "maximum context", "context_length_exceeded"]):
                raise ContextOverflowError(f"上下文溢出: {e}")
            # 识别服务端暂时不可用错误（503/500/502/504）
            if any(code in err_msg for code in ["503", "500", "502", "504", "internalerror.algo"]):
                raise ServerUnavailableError(f"Qwen 服务端暂时不可用: {e}")
            if any(kw in err_msg for kw in ["service unavailable", "internal error", "algo error", "model serving"]):
                raise ServerUnavailableError(f"Qwen 服务端暂时不可用: {e}")
            raise
        
        elapsed = time.time() - t0
        content = response.choices[0].message.content
        if not content or len(content.strip()) < min_response_length:
            raise APIError(f"Qwen 返回内容异常，长度: {len(content) if content else 0}")
        
        self.logger.info(
            f"✅ [LLM] 生成成功 | 耗时={elapsed:.1f}s | 长度={len(content)} | model={model}"
        )
        
        if return_raw:
            raw_data = {
                "model": model,
                "timestamp": datetime.now().isoformat(),
                "prompt_length": len(prompt),
                "response_length": len(content) if content else 0,
                "temperature": temperature,
                "system_msg": system_msg,
                "prompt": prompt,
                "response": content
            }
            return content, raw_data
        
        return content
    
    @retry_with_backoff(10, 5, max_total_wait=3600, server_error_delay=30,
                        max_total_wait_attr='llm_max_total_wait_seconds')
    def generate_with_search(self, query: str, temperature: float = 0.7) -> Tuple[str, List[Dict], Dict]:
        """
        使用真正的限定域名搜索生成 Idea
        强制只允许搜索: SciPy/scikit-learn/PyTorch API 文档, Wikidata/DBpedia, Wolfram MathWorld 等
        
        流程:
        1. 使用 DomainRestrictedSearch 搜索限定域名的内容
        2. 将搜索结果格式化为文本
        3. 调用 LLM 基于搜索结果生成回答
        
        返回: (生成的内容, 搜索结果列表, 原始响应数据)
        """
        if not self.qwen_client:
            raise APIError("Qwen 客户端未初始化")
        
        try:
            # 步骤 1: 执行真正的限定域名搜索
            self.logger.info(f"🔍 执行限定域名搜索: {query}")
            search_results_list = []
            
            if self.domain_searcher:
                search_results = self.domain_searcher.search(
                    query=query,
                    num_results=5,
                    fetch_content=True
                )
                search_results_list = [
                    {
                        'title': r.title,
                        'url': r.url,
                        'snippet': r.snippet,
                        'content': r.content,
                        'source': r.source,
                        'is_allowed_domain': True  # 搜索器已经过滤了
                    }
                    for r in search_results
                ]
                self.logger.info(f"✅ 限定域名搜索完成，找到 {len(search_results_list)} 个结果")
            else:
                self.logger.warning("⚠️ 限定域名搜索器未初始化，无法进行搜索")
            
            # 步骤 2: 将搜索结果格式化为文本
            if self.domain_searcher and search_results_list:
                search_context = self.domain_searcher.format_results_for_llm(search_results)
            else:
                search_context = "未找到相关搜索结果。请基于已有知识回答问题。"
            
            # 步骤 3: 构建提示词，明确要求基于搜索结果回答
            prompt = f"""你是一个专业的研究助手，专注于科学计算和机器学习领域。

【搜索结果】
以下是从限定域名（SciPy、scikit-learn、PyTorch 等官方文档）搜索到的相关信息：

{search_context}

【用户查询】
{query}

【严格要求】
1. 你必须基于上述【搜索结果】中的信息来回答
2. 所有引用的内容必须来自以下允许的域名：
   - docs.scipy.org (SciPy 官方文档)
   - scikit-learn.org (scikit-learn 官方文档)
   - pytorch.org (PyTorch 官方文档)
   - numpy.org (NumPy 官方文档)
   - wikidata.org (Wikidata 知识图谱)
   - dbpedia.org (DBpedia 知识图谱)
   - mathworld.wolfram.com (Wolfram MathWorld)
   - docs.python.org (Python 官方文档)

3. 在回答中明确标注信息来源的 URL
4. 如果搜索结果中没有相关信息，请明确说明"搜索未找到相关信息"
5. 禁止编造或引用非限定域名的信息

请基于搜索结果，提供详细、准确的技术分析。"""

            # 步骤 4: 调用 Qwen 生成回答
            self.logger.info("🤖 调用 Qwen 基于搜索结果生成回答...")
            try:
                # 安全上限保护
                search_max_tokens = min(self.config.max_tokens, 393216)
                response = self.qwen_client.chat.completions.create(
                    model=self.config.qwen_model_name,
                    messages=[
                        {"role": "system", "content": "你是严谨的技术研究助手，只基于提供的搜索结果回答问题。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=search_max_tokens
                )
            except Exception as e:
                err_msg = str(e).lower()
                if any(k in err_msg for k in ["context length", "too many tokens", "token limit", "maximum context", "context_length_exceeded"]):
                    raise ContextOverflowError(f"上下文溢出: {e}")
                if any(code in err_msg for code in ["503", "500", "502", "504", "internalerror.algo"]):
                    raise ServerUnavailableError(f"Qwen 服务端暂时不可用: {e}")
                if any(kw in err_msg for kw in ["service unavailable", "internal error", "algo error", "model serving"]):
                    raise ServerUnavailableError(f"Qwen 服务端暂时不可用: {e}")
                raise
            
            content = response.choices[0].message.content if response.choices[0].message.content else ""
            
            # 步骤 5: 构建原始响应数据
            raw_response = {
                "model": self.config.qwen_model_name,
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "search_results": search_results_list,
                "search_result_count": len(search_results_list),
                "output_text": content,
                "search_domains": self.ALLOWED_SEARCH_DOMAINS
            }
            
            if not content or len(content.strip()) < 50:
                raise APIError(f"Qwen 返回内容异常，长度: {len(content) if content else 0}")
            
            self.logger.info(f"✅ Qwen 基于搜索结果生成成功，长度: {len(content)}, 搜索结果数: {len(search_results_list)}")
            return content, search_results_list, raw_response
            
        except Exception as e:
            self.logger.error(f"Qwen API 调用异常: {str(e)[:200]}")
            # 如果搜索失败，返回保底内容
            return f"搜索功能暂时不可用，使用本地知识库进行分析。\n\n原始查询: {query}", [], {}
    
    @retry_with_backoff(10, 5, max_total_wait=3600, server_error_delay=30,
                        max_total_wait_attr='llm_max_total_wait_seconds')
    def generate_with_kimi(self, prompt: str, temperature: float, system_msg: str, return_raw: bool = False):
        """使用 Kimi 生成"""
        # 安全上限：Kimi 上下文通常最大 256k，max_tokens 建议不超过 128k
        max_tokens = min(self.config.max_tokens, 131072)
        kwargs = {
            "model": self.config.kimi_model_name,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            response = self.kimi_client.chat.completions.create(**kwargs)
        except Exception as e:
            err_msg = str(e).lower()
            if any(k in err_msg for k in ["context length", "too many tokens", "token limit", "maximum context", "context_length_exceeded"]):
                raise ContextOverflowError(f"上下文溢出: {e}")
            if any(code in err_msg for code in ["503", "500", "502", "504", "internalerror.algo"]):
                raise ServerUnavailableError(f"Kimi 服务端暂时不可用: {e}")
            if any(kw in err_msg for kw in ["service unavailable", "internal error", "algo error", "model serving"]):
                raise ServerUnavailableError(f"Kimi 服务端暂时不可用: {e}")
            raise
        content = response.choices[0].message.content
        if not content or len(content.strip()) < 50:
            raise APIError(f"Kimi 返回内容异常，长度: {len(content) if content else 0}")
        self.logger.info(f"✅ Kimi 生成成功，长度: {len(content)}")
        if return_raw:
            raw_data = {
                "model": self.config.kimi_model_name,
                "timestamp": datetime.now().isoformat(),
                "prompt_length": len(prompt),
                "response_length": len(content) if content else 0,
                "temperature": temperature,
                "system_msg": system_msg,
                "prompt": prompt,
                "response": content
            }
            return content, raw_data
        return content


# ==========================================
# 代码提取与验证
# ==========================================

class CodeExtractor:
    """多策略代码提取器 - 增强版，支持截断响应处理和智能合并"""
    
    @staticmethod
    def extract_code(text: str, logger: logging.Logger, base_code: str = None, require_forward: bool = None) -> Tuple[str, str]:
        """
        从 LLM 输出中提取代码，返回 (代码, 提取方法)
        尝试多种策略：markdown 代码块、截断响应处理、Python 文件模式、启发式提取、AST分块
        require_forward 为 None 时从当前任务 patch_spec.require_forward 读取（默认 True == 现行为）
        """
        if require_forward is None:
            from autoresearch_tasks import current_task  # 延迟导入防循环
            require_forward = current_task().patch_spec.require_forward
        original_text = text
        
        # 策略 1: 标准 markdown 代码块 (python) - 完整块
        match = re.search(r'```python\s*\n(.*?)\n```', text, re.DOTALL)
        if match:
            code = match.group(1).strip()
            if CodeExtractor._validate_python_syntax(code, logger, require_forward=require_forward):
                return code, "markdown_python"
        
        # 策略 2: 通用 markdown 代码块 - 完整块
        match = re.search(r'```\s*\n(.*?)\n```', text, re.DOTALL)
        if match:
            code = match.group(1).strip()
            if CodeExtractor._validate_python_syntax(code, logger, require_forward=require_forward):
                return code, "markdown_generic"
        
        # 策略 3: 处理截断响应 - 找到 ```python 后提取到文本末尾
        if '```python' in text:
            start = text.find('```python') + len('```python')
            # 跳过可能的换行
            while start < len(text) and text[start] in '\n\r':
                start += 1
            code = text[start:].strip()
            # 尝试修复截断的代码
            fixed_code = CodeExtractor._fix_truncated_code(code, logger)
            if CodeExtractor._validate_python_syntax(fixed_code, logger, require_forward=require_forward):
                logger.info(f"✅ 从截断响应中提取并修复代码 ({len(fixed_code)} 字符)")
                return fixed_code, "truncated_python_fixed"
            # 如果修复后仍无效，尝试与基础代码合并
            if base_code:
                merged = CodeExtractor._smart_merge_with_base(base_code, code, logger)
                if merged and CodeExtractor._validate_python_syntax(merged, logger, require_forward=require_forward):
                    logger.info(f"✅ 截断代码与基础代码合并成功 ({len(merged)} 字符)")
                    return merged, "truncated_merged"
        
        # 策略 4: 假设整个回复就是代码 (如果包含 def/class/import)
        if any(keyword in text for keyword in ['def ', 'class ', 'import ', 'from ']):
            if CodeExtractor._validate_python_syntax(text, logger, require_forward=require_forward):
                return text.strip(), "full_text"
        
        # 策略 5: AST 分块提取 - 找最大的有效 Python 代码块
        ast_code = CodeExtractor._extract_largest_valid_ast_block(text, logger)
        if ast_code:
            if base_code:
                merged = CodeExtractor._smart_merge_with_base(base_code, ast_code, logger)
                if merged and CodeExtractor._validate_python_syntax(merged, logger, require_forward=require_forward):
                    return merged, "ast_merged"
            return ast_code, "ast_block"
        
        # 策略 6: 启发式提取 (找第一个 def/class 到最后)
        try:
            lines = text.split('\n')
            start_idx = None
            for i, line in enumerate(lines):
                if line.strip().startswith(('def ', 'class ', 'import ')):
                    if start_idx is None:
                        start_idx = i
            
            if start_idx is not None:
                # 找到合理的结束位置 (文件尾或下一个 markdown 块)
                end_idx = len(lines)
                for j in range(start_idx, len(lines)):
                    if lines[j].strip() == '```':
                        end_idx = j
                        break
                
                code = '\n'.join(lines[start_idx:end_idx]).strip()
                fixed_code = CodeExtractor._fix_truncated_code(code, logger)
                if CodeExtractor._validate_python_syntax(fixed_code, logger, require_forward=require_forward):
                    return fixed_code, "heuristic"
                if base_code:
                    merged = CodeExtractor._smart_merge_with_base(base_code, fixed_code, logger)
                    if merged and CodeExtractor._validate_python_syntax(merged, logger, require_forward=require_forward):
                        return merged, "heuristic_merged"
        except Exception as e:
            logger.warning(f"启发式提取失败: {e}")
        
        # 最后尝试：如果基础代码存在，直接返回基础代码
        if base_code:
            logger.warning("⚠️ 所有提取策略均失败，返回基础代码")
            return base_code, "fallback_base"
        
        logger.error("❌ 所有代码提取策略均失败")
        raise ValueError("无法从 LLM 响应中提取有效 Python 代码")
    
    @staticmethod
    def _count_unmatched_brackets(code: str) -> Tuple[int, int, int]:
        """Tokenizer-aware bracket counter that skips strings and comments."""
        stack = []
        in_string = False
        string_char = None
        escape_next = False
        in_comment = False

        for ch in code:
            if escape_next:
                escape_next = False
                continue
            if ch == '\\':
                escape_next = True
                continue
            if in_comment:
                if ch == '\n':
                    in_comment = False
                continue
            if not in_string:
                if ch == '#':
                    in_comment = True
                    continue
                if ch in ('"', "'"):
                    in_string = True
                    string_char = ch
                    continue
                if ch in '([{':
                    stack.append(ch)
                elif ch in ')]}':
                    if stack:
                        last_open = stack[-1]
                        if (last_open == '(' and ch == ')') or \
                           (last_open == '[' and ch == ']') or \
                           (last_open == '{' and ch == '}'):
                            stack.pop()
            else:
                if ch == string_char:
                    in_string = False
                    string_char = None

        open_parens = sum(1 for c in stack if c == '(')
        open_brackets = sum(1 for c in stack if c == '[')
        open_braces = sum(1 for c in stack if c == '{')
        return open_parens, open_brackets, open_braces

    @staticmethod
    def _fix_truncated_code(code: str, logger: logging.Logger) -> str:
        """尝试修复截断的 Python 代码"""
        lines = code.split('\n')
        
        # 1. 移除末尾的不完整行（如果最后一行没有正确结束）
        # 但保留包含特殊标记的注释行（如 MMTok 占位符）
        while lines:
            last_line = lines[-1].strip()
            # 保留占位符标记行和纯注释行
            if last_line.startswith('#') and ('MMTOK' in last_line or 'END' in last_line or 'START' in last_line):
                break
            # 如果最后一行是一个不完整的赋值/表达式，移除它
            if last_line and not last_line.startswith('#') and not last_line.endswith(('#', '"\'"\'', "'\"'\"'", '"', "'", ')', ']', '}', ':', '\\')):
                # 检查是否是完整语句
                if last_line.endswith(('=', '+', '-', '*', '/', '%', '&', '|', '^', '<', '>', ',')):
                    lines.pop()
                    continue
            break
        
        fixed = '\n'.join(lines)
        
        # 2. 如果代码已经语法正确，直接返回，不做任何修改
        try:
            ast.parse(fixed)
            return fixed
        except SyntaxError:
            pass
        
        # 3. 使用 tokenizer-aware 计数器统计不匹配的括号（仅当语法确实有错误时）
        open_parens, open_brackets, open_braces = CodeExtractor._count_unmatched_brackets(fixed)
        
        # 添加缺失的闭合符号
        fixed += ')' * max(0, open_parens)
        fixed += ']' * max(0, open_brackets)
        fixed += '}' * max(0, open_braces)
        
        # 4. 补全 try/except 块
        if 'try:' in fixed and 'except' not in fixed.split('try:')[-1]:
            # 找到最后一个 try: 的位置
            last_try_pos = fixed.rfind('try:')
            indent = ''
            for i in range(last_try_pos - 1, -1, -1):
                if fixed[i] == '\n':
                    indent_start = i + 1
                    while indent_start < last_try_pos and fixed[indent_start] == ' ':
                        indent += ' '
                        indent_start += 1
                    break
            fixed += f"\n{indent}except Exception as e:\n{indent}    pass"
        
        # 5. 补全 if/for/while/def/class 块
        last_line = fixed.rstrip().split('\n')[-1] if fixed else ''
        if last_line.rstrip().endswith(':') and not last_line.strip().startswith('#'):
            indent = len(last_line) - len(last_line.lstrip())
            fixed += f"\n{' ' * (indent + 4)}pass"
        
        # 6. 补全三引号字符串
        triple_double = fixed.count('"""') % 2
        triple_single = fixed.count("'''") % 2
        if triple_double:
            fixed += '\n"""'
        if triple_single:
            fixed += "\n'''"
        
        return fixed
    
    @staticmethod
    def _extract_largest_valid_ast_block(text: str, logger: logging.Logger) -> Optional[str]:
        """使用 AST 找到文本中最大的有效 Python 代码块"""
        try:
            import ast
        except ImportError:
            return None
        
        lines = text.split('\n')
        best_code = None
        best_score = 0
        
        # 尝试从每一行开始，找最长的有效代码块
        for start in range(len(lines)):
            # 只从可能的开头开始
            first_line = lines[start].strip()
            if not first_line or first_line.startswith('#'):
                continue
            if not any(first_line.startswith(kw) for kw in ['import ', 'from ', 'class ', 'def ']):
                if not any(kw in first_line for kw in ['=', 'print', 'return', 'if ', 'for ', 'while ']):
                    continue
            
            for end in range(len(lines), start, -1):
                candidate = '\n'.join(lines[start:end])
                if len(candidate) < 100:  # 太短的跳过
                    continue
                try:
                    ast.parse(candidate)
                    score = len(candidate)
                    if score > best_score:
                        best_score = score
                        best_code = candidate
                    break  # 从当前 start 找到最长的了，继续下一个 start
                except SyntaxError:
                    continue
        
        if best_code:
            logger.info(f"✅ AST 提取到有效代码块 ({len(best_code)} 字符)")
        return best_code
    
    @staticmethod
    def _smart_merge_with_base(base_code: str, generated_code: str, logger: logging.Logger) -> Optional[str]:
        """将生成的代码片段智能合并到基础代码中"""
        try:
            import ast
            
            # 如果生成的代码本身就是完整有效的，直接返回
            try:
                ast.parse(generated_code)
                return generated_code
            except SyntaxError:
                pass
            
            base_lines = base_code.split('\n')
            gen_lines = generated_code.split('\n')
            
            # 策略 A: 如果生成的代码包含完整的类定义，直接替换整个类
            class_name = None
            for line in gen_lines:
                match = re.match(r'^(\s*)class\s+(\w+)', line)
                if match:
                    class_name = match.group(2)
                    break
            
            if class_name:
                # 在基础代码中找这个类
                class_start = None
                class_end = None
                base_indent = None
                for i, line in enumerate(base_lines):
                    match = re.match(r'^(\s*)class\s+' + class_name + r'\b', line)
                    if match:
                        class_start = i
                        base_indent = len(match.group(1))
                        break
                
                if class_start is not None:
                    # 找到类的结束位置
                    class_end = len(base_lines)
                    for j in range(class_start + 1, len(base_lines)):
                        line = base_lines[j]
                        if line.strip() and not line.startswith(' ' * (base_indent + 1)) and not line.startswith('\t'):
                            if not line.strip().startswith(('#', '@', 'class ', 'def ')) or (
                                line.strip().startswith('class ') or line.strip().startswith('def ')
                            ):
                                # 检查是否是同级别的 class/def
                                current_indent = len(line) - len(line.lstrip())
                                if current_indent <= base_indent:
                                    class_end = j
                                    break
                    
                    # 替换类
                    merged = base_lines[:class_start] + gen_lines + base_lines[class_end:]
                    result = '\n'.join(merged)
                    try:
                        ast.parse(result)
                        logger.info(f"✅ 成功替换类 {class_name}")
                        return result
                    except SyntaxError:
                        pass
            
            # 策略 B: 如果生成的代码包含方法定义，尝试替换特定方法
            method_name = None
            for line in gen_lines:
                match = re.match(r'^(\s*)def\s+(\w+)', line)
                if match:
                    method_name = match.group(2)
                    break
            
            if method_name and class_name:
                # 在基础代码中找这个方法
                method_start = None
                method_end = None
                in_class = False
                class_indent = None
                
                for i, line in enumerate(base_lines):
                    if re.match(r'^(\s*)class\s+' + class_name + r'\b', line):
                        in_class = True
                        class_indent = len(line) - len(line.lstrip())
                        continue
                    if in_class:
                        current_indent = len(line) - len(line.lstrip())
                        if line.strip() and current_indent <= class_indent and not line.strip().startswith('#'):
                            in_class = False
                            continue
                        
                        match = re.match(r'^(\s*)def\s+' + method_name + r'\b', line)
                        if match:
                            method_start = i
                            method_indent = len(match.group(1))
                            # 找方法结束 - 跳过方法签名延续行
                            in_signature = True
                            method_end = len(base_lines)
                            for j in range(method_start + 1, len(base_lines)):
                                next_line = base_lines[j]
                                stripped = next_line.strip()
                                if not stripped:
                                    continue
                                next_indent = len(next_line) - len(next_line.lstrip())
                                
                                if in_signature:
                                    if stripped.endswith('):'):
                                        in_signature = False
                                        continue
                                    elif stripped.endswith((')', ',')) or '->' in stripped:
                                        continue
                                    else:
                                        in_signature = False
                                
                                if next_indent <= method_indent and not stripped.startswith(('@', '#')):
                                    method_end = j
                                    break
                            break
                
                if method_start is not None:
                    merged = base_lines[:method_start] + gen_lines + base_lines[method_end:]
                    result = '\n'.join(merged)
                    try:
                        ast.parse(result)
                        logger.info(f"✅ 成功替换方法 {method_name}")
                        return result
                    except SyntaxError:
                        pass
            
            # 策略 C: 简单插入 - 如果生成的代码有 forward 方法，替换 base_code 中的 forward 方法
            if 'def forward(' in generated_code:
                return CodeExtractor._replace_method_in_base(base_code, 'forward', generated_code, logger)
            
            logger.warning("⚠️ 智能合并未找到合适的替换策略")
            return None
            
        except Exception as e:
            logger.warning(f"智能合并失败: {e}")
            return None
    
    @staticmethod
    def _replace_method_in_base(base_code: str, method_name: str, generated_code: str, logger: logging.Logger) -> Optional[str]:
        """在基础代码中替换指定方法"""
        try:
            import ast
            base_lines = base_code.split('\n')
            gen_lines = generated_code.split('\n')
            
            # 动态推断目标类名（支持 Qwen 和 LLaVA）
            target_class_name = None
            for line in base_lines:
                m = re.match(r'^\s*class\s+(\w+)', line)
                if m:
                    target_class_name = m.group(1)
                    break
            if not target_class_name:
                logger.warning("⚠️ _replace_method_in_base: 无法在基础代码中找到 class 定义")
                return None
            
            # 找到基础代码中方法的位置
            method_start = None
            method_end = None
            method_indent = None
            in_target_class = False
            
            for i, line in enumerate(base_lines):
                if f'class {target_class_name}' in line:
                    in_target_class = True
                    continue
                if in_target_class and line.strip().startswith('class '):
                    in_target_class = False
                    continue
                
                if in_target_class:
                    match = re.match(r'^(\s*)def\s+' + method_name + r'\(', line)
                    if match:
                        method_start = i
                        method_indent = len(match.group(1))
                        # 找方法结束 - 跳过方法签名延续行
                        in_signature = True
                        method_end = len(base_lines)
                        for j in range(method_start + 1, len(base_lines)):
                            next_line = base_lines[j]
                            stripped = next_line.strip()
                            if not stripped:
                                continue
                            next_indent = len(next_line) - len(next_line.lstrip())
                            
                            if in_signature:
                                if stripped.endswith('):'):
                                    in_signature = False
                                    continue
                                elif stripped.endswith((')', ',')) or '->' in stripped:
                                    continue
                                else:
                                    in_signature = False
                            
                            if next_indent <= method_indent and not stripped.startswith(('@', '#')):
                                method_end = j
                                break
                        break
            
            if method_start is not None:
                # 从生成的代码中提取方法体
                gen_method_start = None
                gen_method_end = None
                for i, line in enumerate(gen_lines):
                    if re.match(r'^(\s*)def\s+' + method_name + r'\(', line):
                        gen_method_start = i
                        gen_indent = len(line) - len(line.lstrip())
                        gen_method_end = len(gen_lines)
                        in_signature = True
                        for j in range(gen_method_start + 1, len(gen_lines)):
                            next_line = gen_lines[j]
                            stripped = next_line.strip()
                            if not stripped:
                                continue
                            next_indent = len(next_line) - len(next_line.lstrip())
                            
                            if in_signature:
                                if stripped.endswith('):'):
                                    in_signature = False
                                    continue
                                elif stripped.endswith((')', ',')) or '->' in stripped:
                                    continue
                                else:
                                    in_signature = False
                            
                            if next_indent <= gen_indent and not stripped.startswith(('@', '#')):
                                gen_method_end = j
                                break
                        break
                
                if gen_method_start is not None:
                    method_lines = gen_lines[gen_method_start:gen_method_end]
                    merged = base_lines[:method_start] + method_lines + base_lines[method_end:]
                    result = '\n'.join(merged)
                    try:
                        ast.parse(result)
                        logger.info(f"✅ 成功替换方法 {method_name}")
                        return result
                    except SyntaxError:
                        pass
            
            return None
        except Exception as e:
            logger.warning(f"替换方法失败: {e}")
            return None
    
    @staticmethod
    def _validate_python_syntax(code: str, logger: logging.Logger, require_forward: bool = True) -> bool:
        """验证 Python 语法有效性 - 增强版
        
        Args:
            code: 要验证的 Python 代码
            logger: 日志记录器
            require_forward: 是否强制要求包含 forward 方法。
                           对于 selector 模式（如 QwenAlgorithmCore），不需要 forward 方法，应设为 False。
        """
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            logger.warning(f"语法验证失败: {e}")
            return False
        
        # 额外检查：使用 AST 分析
        try:
            import ast
            tree = ast.parse(code)
            
            # 检查是否有必要的结构
            has_class = False
            has_method = False
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    has_class = True
                elif isinstance(node, ast.FunctionDef) and node.name == 'forward':
                    has_method = True
            
            if not has_class:
                logger.warning("⚠️ 代码中缺少类定义")
                return False
            if require_forward and not has_method:
                logger.warning("⚠️ 代码中缺少 forward 方法")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"AST 分析失败: {e}")
            # AST 失败但不影响语法正确性，仍然返回 True
            return True
    
    @staticmethod
    def quick_validate(code: str) -> Tuple[bool, str]:
        """快速验证代码 - 返回 (是否有效, 错误信息)"""
        # 1. 基本语法检查
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            return False, f"语法错误: {e}"
        
        # 2. 检查关键变量
        required_vars = ['selected_indices', 'selected_image_embeds']
        for var in required_vars:
            if var not in code:
                return False, f"缺少必需变量: {var}"
        
        # 3. 检查类型安全
        if re.search(r'selected_indices\s*=\s*\[', code):
            # 检查是否有转换
            if not re.search(r'torch\.tensor\([^)]*selected_indices', code):
                return False, "selected_indices 被赋值为 list，需要转换为 torch.Tensor"
        
        # 4. 检查 try-except
        if 'try:' in code:
            if 'except' not in code:
                return False, "try 块缺少 except"
        
        return True, "验证通过"


# ==========================================
# 安全的执行环境
# ==========================================

class SafeExecutor:
    """安全的命令执行器，带超时、资源限制、输出解析"""
    
    def __init__(self, config: AutoResearchConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        # 线程安全的当前进程集合（多 GPU 分支并行评测时多个 run_evaluation 并发执行）
        self._current_processes = set()
        self._process_lock = threading.Lock()
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """设置信号处理，支持优雅退出"""
        def signal_handler(signum, frame):
            self.logger.warning(f"⚠️  收到信号 {signum}，正在终止子进程并退出...")
            with self._process_lock:
                processes = list(self._current_processes)
            for proc in processes:
                try:
                    if sys.platform != 'win32':
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    else:
                        proc.kill()
                except Exception:
                    pass
            raise KeyboardInterrupt
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    _gpu_claim_lock = threading.Lock()
    _gpu_claims: Dict[str, str] = {}  # gpu_idx(str) -> owner tag（防并发评测叠卡）

    @classmethod
    def _claimed_gpus(cls) -> set:
        with cls._gpu_claim_lock:
            return set(cls._gpu_claims.keys())

    @classmethod
    def _release_gpu_claims(cls, owner: str):
        """释放某次评测认领的全部 GPU（attempt 结束即调用）。"""
        with cls._gpu_claim_lock:
            for idx in [i for i, o in cls._gpu_claims.items() if o == owner]:
                cls._gpu_claims.pop(idx, None)

    @classmethod
    def eligible_gpu_count(cls, min_free_mib: float) -> int:
        """合格卡数（空闲达标 + 未被引擎评测认领），自适应并发槽位用。"""
        claimed = cls._claimed_gpus()
        return sum(1 for idx, free in cls._query_free_gpus()
                   if free >= min_free_mib and str(idx) not in claimed)

    @staticmethod
    def _query_free_gpus() -> List[Tuple[int, float]]:
        """查询所有 GPU 的剩余显存，返回 [(index, free_mib), ...] 按剩余显存降序。

        优先用 nvidia-smi；当 nvidia-smi 不可用（如 NVML 驱动/库版本不匹配导致
        "Failed to initialize NVML"）时，回退到 PyTorch 的 torch.cuda.mem_get_info
        （走 CUDA driver API，不依赖用户态 NVML 库）。全部失败返回空列表。
        """
        gpus: List[Tuple[int, float]] = []
        # 方案一：nvidia-smi
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                text=True, timeout=15, stderr=subprocess.DEVNULL)
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    continue
                try:
                    idx, used, total = int(parts[0]), float(parts[1]), float(parts[2])
                except ValueError:
                    continue
                gpus.append((idx, total - used))
        except Exception:
            gpus = []
        if gpus:
            gpus.sort(key=lambda x: x[1], reverse=True)
            return gpus
        # 方案二：PyTorch CUDA driver API（NVML 失效时仍可用）
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    free_b, _total_b = torch.cuda.mem_get_info(i)
                    gpus.append((i, free_b / 1024.0 / 1024.0))
        except Exception:
            gpus = []
        gpus.sort(key=lambda x: x[1], reverse=True)
        return gpus

    @classmethod
    def _select_freest_gpu(cls, logger=None) -> Optional[str]:
        """选择剩余显存最多的 GPU。失败时返回 None（回退到 config 固定值）。"""
        gpus = cls._query_free_gpus()
        if not gpus:
            if logger:
                logger.warning("⚠️ [GPU AutoSelect] GPU 探测失败（nvidia-smi 与 PyTorch 均不可用），回退到固定 cuda_device")
            return None
        best_idx, best_free = gpus[0]
        if logger:
            logger.info(f"🎮 [GPU AutoSelect] 选择 GPU {best_idx}（剩余显存 {best_free/1024:.1f} GiB）")
        return str(best_idx)

    def _wait_for_gpu(self, owner: str = "") -> Optional[str]:
        """
        cuda_device=auto 时的选卡逻辑：
        先判断各卡剩余显存是否满足 min_free_gpu_gb（单次评测所需），
        满足才选用；都不满足时按 gpu_poll_interval 轮询等待，
        超过 gpu_wait_seconds 后兜底使用当前最空闲的卡。

        原子认领（防叠卡 OOM）：查询→排除已被引擎评测认领的卡→选最空闲→
        登记认领，全程在同一把锁内完成，并发 worker 不会选中同一张卡。
        认领随 attempt 结束由 _release_gpu_claims 释放。
        """
        min_free_mib = float(getattr(self.config, 'min_free_gpu_gb', 0.0) or 0.0) * 1024
        wait_budget = int(getattr(self.config, 'gpu_wait_seconds', 600))
        poll = max(5, int(getattr(self.config, 'gpu_poll_interval', 60)))
        waited = 0
        while True:
            with SafeExecutor._gpu_claim_lock:
                gpus = self._query_free_gpus()
                if not gpus:
                    self.logger.warning("⚠️ [GPU AutoSelect] GPU 探测失败（nvidia-smi 与 PyTorch 均不可用），回退到 GPU 0")
                    return "0"
                claimed = set(SafeExecutor._gpu_claims.keys())
                best_idx, best_free = next(
                    ((i, f) for i, f in gpus if str(i) not in claimed), (None, 0.0))
                if best_idx is not None and best_free >= min_free_mib:
                    if owner:
                        SafeExecutor._gpu_claims[str(best_idx)] = owner
                    self.logger.info(
                        f"🎮 [GPU AutoSelect] 选择 GPU {best_idx}"
                        f"（剩余 {best_free/1024:.1f} GiB ≥ 需求 {min_free_mib/1024:.0f} GiB"
                        f"{f'，已排除 {len(claimed)} 张被认领卡' if claimed else ''}）"
                    )
                    return str(best_idx)
                # gpu_wait_seconds <= 0 表示无限等待，直到有卡满足门槛
                if wait_budget > 0 and waited >= wait_budget:
                    # 兜底：用未被认领的最空闲卡；全被认领则退回全局最空闲卡
                    if best_idx is None:
                        best_idx, best_free = gpus[0]
                    if owner:
                        SafeExecutor._gpu_claims[str(best_idx)] = owner
                    self.logger.warning(
                        f"⚠️ [GPU AutoSelect] 等待 {waited}s 后仍无未被认领的 GPU 满足 "
                        f"{min_free_mib/1024:.0f} GiB（最空闲 GPU {best_idx} 仅余 "
                        f"{best_free/1024:.1f} GiB），兜底使用 GPU {best_idx}"
                    )
                    return str(best_idx)
            # 无合格卡：锁外等待（认领快照下次重查，不持锁睡眠）
            wait_hint = f"已等待 {waited}s" if wait_budget > 0 else f"已等待 {waited}s（无限等待直到有卡可用）"
            self.logger.info(
                f"⏳ [GPU AutoSelect] 所有未被认领的 GPU 剩余显存均不足 {min_free_mib/1024:.0f} GiB"
                f"{f'（{len(claimed)} 张被认领）' if claimed else ''}，{poll}s 后重试（{wait_hint}）..."
            )
            time.sleep(poll)
            waited += poll

    def run_evaluation(self, iteration: int, cuda_dev: Optional[str] = None,
                       eval_cmd_override: Optional[str] = None,
                       extra_env: Optional[Dict[str, str]] = None) -> Tuple[str, bool, Dict[str, Any]]:
        """
        执行评测命令，返回 (完整日志, 是否成功, 指标字典)
        实时流式输出日志到控制台和日志文件，支持关键错误提前终止

        可选覆盖参数（多 GPU 分支并行评测使用；不传时行为与原单参数调用逐字一致）：
            cuda_dev: 显式指定 GPU，跳过 _wait_for_gpu 自动选择；
            eval_cmd_override: 替代 config.eval_cmd（CUDA_VISIBLE_DEVICES 覆写仍生效）；
            extra_env: 额外合并进子进程环境变量（如 PYTHONPATH 注入）。
        """
        import threading

        # GPU 自动选择改为【按次尝试】：每次 attempt 独立选卡+原子认领
        # （防并发评测叠卡 OOM；OOM/失败重试换卡而非撞同一张卡），
        # attempt 结束立即释放。显式 cuda_dev（分支并行路径）不参与认领。
        if cuda_dev is None:
            cuda_dev = self.config.cuda_device
        auto_gpu = (cuda_dev == "auto")
        claim_owner = f"eval:{iteration}:{id(self)}"

        base_cmd = eval_cmd_override if eval_cmd_override is not None else self.config.eval_cmd
        if not base_cmd:
            raise ValueError("评测命令未配置")

        self.logger.info(f"🚀 开始执行评测 (Iteration {iteration})...")
        self.logger.info(f"📋 命令: {base_cmd[:120]}...")
        
        CRITICAL_ERROR_PATTERNS = [
            # FIX: 移除 'Traceback (most recent call last)'，避免 deprecation warning 等非致命 traceback 触发 kill。
            # 致命错误的具体类型（RuntimeError/CUDA error 等）会在 traceback 之后的行中被捕获。
            'RuntimeError:',
            'CUDA error',
            'OutOfMemoryError',
            'AssertionError:',
            'IndexError:',
            'TypeError:',
            'ValueError:',
            'KeyError:',
            'AttributeError:',
            'ImportError:',
            'ModuleNotFoundError:',
            'cannot import name',
        ]
        
        for attempt in range(self.config.eval_retry_attempts):
            # 每次尝试：选卡+认领（auto 时）→ 构建环境 → 重写命令前缀
            if auto_gpu:
                cuda_dev = self._wait_for_gpu(owner=claim_owner)
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = cuda_dev
            if extra_env:
                env.update(extra_env)
            cmd = base_cmd
            if re.search(r'CUDA_VISIBLE_DEVICES=\S+', cmd):
                cmd = re.sub(r'CUDA_VISIBLE_DEVICES=\S+', f'CUDA_VISIBLE_DEVICES={cuda_dev}', cmd)
            stdout_lines = []
            stderr_lines = []
            kill_event = threading.Event()
            lines_after_kill = [0]
            
            process = None
            try:
                popen_kwargs = {}
                if sys.platform != 'win32':
                    popen_kwargs['preexec_fn'] = os.setsid
                
                eval_start_time = time.time()
                process = subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                    bufsize=1,
                    universal_newlines=True,
                    **popen_kwargs
                )
                with self._process_lock:
                    self._current_processes.add(process)
                
                def read_stream(stream, lines, prefix):
                    """读取流并实时输出，支持关键错误检测"""
                    # 已知的非致命错误模式（如可选模块导入失败），匹配到则不应终止评测
                    NON_FATAL_PATTERNS = [
                        'Failed to import llava',
                        'LLaVA not installed',
                        'Patch skipped',
                    ]
                    try:
                        for line in iter(stream.readline, ''):
                            if not line:
                                break
                            line = line.rstrip('\n\r')
                            lines.append(line)
                            # 实时输出到日志
                            self.logger.info(f"{prefix} {line}")
                            
                            # 流式错误检测
                            if getattr(self.config, 'eval_stream_kill_on_error', True) and not kill_event.is_set():
                                if any(pattern in line for pattern in CRITICAL_ERROR_PATTERNS):
                                    # 排除已知的非致命错误
                                    if any(nf in line for nf in NON_FATAL_PATTERNS):
                                        self.logger.warning(f"⚠️ 检测到已知非致命错误，忽略: {line[:120]}")
                                    else:
                                        self.logger.error(f"🚨 检测到关键错误: {line[:120]}")
                                        kill_event.set()
                            
                            if kill_event.is_set():
                                lines_after_kill[0] += 1
                                if lines_after_kill[0] >= 30:
                                    self.logger.warning("⚠️  已收集足够错误上下文，终止评测进程")
                                    try:
                                        if sys.platform != 'win32':
                                            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                                        else:
                                            process.kill()
                                    except Exception:
                                        pass
                                    break
                        stream.close()
                    except Exception as e:
                        self.logger.debug(f"读取流异常: {e}")
                
                # 启动线程读取 stdout 和 stderr
                stdout_thread = threading.Thread(target=read_stream, args=(process.stdout, stdout_lines, "[OUT]"))
                stderr_thread = threading.Thread(target=read_stream, args=(process.stderr, stderr_lines, "[ERR]"))
                stdout_thread.daemon = True
                stderr_thread.daemon = True
                stdout_thread.start()
                stderr_thread.start()
                
                # 等待进程完成，带超时
                try:
                    process.wait(timeout=self.config.timeout_seconds)
                    eval_elapsed = time.time() - eval_start_time
                except subprocess.TimeoutExpired:
                    eval_elapsed = time.time() - eval_start_time
                    try:
                        if sys.platform != 'win32':
                            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        else:
                            process.kill()
                    except Exception:
                        pass
                    stdout_thread.join(timeout=5)
                    stderr_thread.join(timeout=5)
                    error_msg = f"❌ 执行超时 (超过 {self.config.timeout_seconds} 秒)"
                    self.logger.error(error_msg)
                    # FIX: 超时不重试，直接返回失败，由外层循环进入下一轮 iteration
                    return error_msg, False, {"error": "timeout", "eval_time": eval_elapsed}
                
                # 等待线程完成读取
                stdout_thread.join(timeout=10)
                stderr_thread.join(timeout=10)
                
                stdout = '\n'.join(stdout_lines)
                stderr = '\n'.join(stderr_lines)
                full_log = f"=== STDOUT ===\n{stdout}\n=== STDERR ===\n{stderr}"
                
                # 解析指标
                metrics = self._parse_metrics(stdout + stderr)
                metrics['eval_time'] = eval_elapsed
                # G1: 主指标键经 TaskProfile.primary_metric 解析（默认 "accuracy"，行为不变）
                from autoresearch_tasks import current_task
                _pm = current_task().primary_metric
                parsed_acc = metrics.get(_pm, metrics.get('accuracy', -1))
                
                # FIX: 如果 accuracy 被成功解析且有效，即使进程返回码非零
                #（通常是脚本末尾的非致命错误，如资源清理、日志写入等），
                # 也视为评测成功，避免 valid accuracy 被 returncode 阻断 deepen_templates 入库。
                if parsed_acc >= 0:
                    if process.returncode == 0:
                        is_success = True
                    elif not kill_event.is_set():
                        self.logger.warning(
                            f"⚠️  评测进程返回码非零({process.returncode})但accuracy已解析({parsed_acc:.4f})，"
                            f"视为成功（可能是脚本末尾非致命错误）"
                        )
                        is_success = True
                    else:
                        is_success = False
                else:
                    is_success = process.returncode == 0
                
                self.logger.info(f"📊 返回码: {process.returncode}, 解析指标: {metrics}")
                
                if kill_event.is_set():
                    self.logger.warning(f"⚠️  评测因关键错误被提前终止 (尝试 {attempt+1})")
                    return full_log, False, metrics
                
                if is_success:
                    self.logger.info(f"✅ 评测成功 (尝试 {attempt+1}), 耗时={eval_elapsed:.1f}s")
                    return full_log, True, metrics
                else:
                    self.logger.warning(f"⚠️  评测未通过 (尝试 {attempt+1}/{self.config.eval_retry_attempts}), 耗时={eval_elapsed:.1f}s")
                    if attempt == self.config.eval_retry_attempts - 1:
                        return full_log, False, metrics
                    time.sleep(5)
                    
            except Exception as e:
                error_msg = f"❌ 执行异常: {str(e)}"
                self.logger.error(error_msg)
                if attempt == self.config.eval_retry_attempts - 1:
                    return error_msg, False, {"error": str(e)}
            finally:
                with self._process_lock:
                    self._current_processes.discard(process)
                # attempt 结束（成功/失败/异常）立即释放 GPU 认领
                if auto_gpu:
                    self._release_gpu_claims(claim_owner)
        
        return "未知错误", False, {}
    
    def _parse_metrics(self, text: str) -> Dict[str, float]:
        """从输出日志中解析关键指标

        Phase 4：默认实现已逐字收编至 autoresearch_tasks.metric_parsers.parse_lmms_eval_metrics；
        当前任务 profile 可通过 TaskProfile.metric_parser 字段定制解析器。
        """
        from autoresearch_tasks import current_task
        from autoresearch_tasks.metric_parsers import parse_lmms_eval_metrics
        parser = current_task().get_metric_parser()
        if parser is parse_lmms_eval_metrics:
            # 默认实现支持透传 logger（保留表格解析时的日志输出）
            return parser(text, logger=self.logger)
        return parser(text)


class AsyncEvalTask:
    """[AsyncEval] 一个异步评测任务的全部上下文（回收记账所需）。

    主线程在交出时填充；worker 线程只读；回收时主线程用这些字段重建
    该评测所属轮的记账上下文（strategy/topic/mode/代码快照等）。
    """

    def __init__(self, iteration: int, iter_dir: str, pythonpath: str,
                 code: Optional[str], idea: Optional[str],
                 strategy: Optional[Dict[str, Any]], current_mode: str,
                 topic: Optional[str]):
        self.iteration = iteration
        self.iter_dir = iter_dir
        # 评测子进程的 PYTHONPATH（已含快照环境前缀）
        self.pythonpath = pythonpath
        self.code = code
        self.idea = idea
        self.strategy = strategy
        self.current_mode = current_mode
        # 交出时 explorer.last_used_topic 的快照：回收记账（finalize 内
        # result["topic"] / topic_generator / candidate_pool 均读该属性）时必须换回
        self.topic = topic
        self.submit_time = time.time()


class AsyncEvalManager:
    """[AsyncEval] 异步评测管理器：worker 线程只跑评测，记账一律由主线程 drain 完成。

    - submit(): 登记 in-flight 并起一个 daemon worker 线程；
    - worker: 调 SafeExecutor.run_evaluation（内部含 GPU 显存闸 _wait_for_gpu、
      流式输出、timeout_seconds、重试与指标解析，与同步路径同一实现），
      结果放入 completed 队列；worker 不做任何记账（引擎状态单线程设计）；
    - drain(): 主线程非阻塞取回全部已完成结果（并从 in-flight 移除）。
    """

    def __init__(self, config: AutoResearchConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        # 专用 SafeExecutor（在主线程构造，其 SIGINT/SIGTERM handler 能 kill
        # 全部在途评测子进程——所有 worker 的进程都登记在它的 _current_processes）
        self.executor = SafeExecutor(config, logger)
        self._lock = threading.Lock()
        self._inflight: Dict[int, AsyncEvalTask] = {}
        self._completed: "queue.Queue" = queue.Queue()

    def inflight_count(self) -> int:
        with self._lock:
            return len(self._inflight)

    def inflight_iterations(self) -> List[int]:
        with self._lock:
            return sorted(self._inflight.keys())

    def submit(self, task: AsyncEvalTask):
        with self._lock:
            self._inflight[task.iteration] = task
        t = threading.Thread(
            target=self._worker, args=(task,),
            daemon=True, name=f"AsyncEval-{task.iteration}")
        t.start()

    def _worker(self, task: AsyncEvalTask):
        self.logger.info(
            f"⏳ [AsyncEval] iter={task.iteration} 等 GPU（≥{getattr(self.config, 'min_free_gpu_gb', 30):.0f} GiB）...")
        try:
            eval_log, is_success, metrics = self.executor.run_evaluation(
                task.iteration,
                extra_env={"PYTHONPATH": task.pythonpath})
        except Exception as e:
            eval_log = f"[AsyncEval] worker exception: {type(e).__name__}: {e}"
            is_success, metrics = False, {"error": str(e)}
        acc = metrics.get("accuracy", -1) if isinstance(metrics, dict) else -1
        self.logger.info(
            f"📊 [AsyncEval] iter={task.iteration} 评测完成 "
            f"acc={acc if acc < 0 else f'{acc:.4f}'} success={is_success} "
            f"（耗时 {time.time() - task.submit_time:.0f}s，含等 GPU）"
        )
        self._completed.put((task, eval_log, is_success, metrics))

    def drain(self) -> List[Tuple[AsyncEvalTask, str, bool, Dict[str, Any]]]:
        """非阻塞取回全部已完成评测（同时从 in-flight 移除）。

        已被 reap_timed_out 收割（放弃记账）的迭代，其迟到完成结果在此直接
        丢弃——对应 iteration 已按 async_hard_timeout 记过失败，不能再二次记账。
        """
        out = []
        while True:
            try:
                item = self._completed.get_nowait()
            except queue.Empty:
                break
            with self._lock:
                removed = self._inflight.pop(item[0].iteration, None)
            if removed is None:
                self.logger.warning(
                    f"🗑️ [AsyncEval] iter={item[0].iteration} 的迟到结果被丢弃"
                    f"（此前已按硬超时收割记账）")
                continue
            out.append(item)
        return out

    def reap_timed_out(self, hard_timeout_seconds: float) -> List[AsyncEvalTask]:
        """[AsyncEval] 收割在途时长超过硬超时的任务（主线程调用）。

        只放弃记账（从 in-flight 移除并返回给调用方按失败记账），不杀线程——
        worker 自身有界：评测子进程受 timeout_seconds 约束，等 GPU 受
        gpu_wait_seconds（应配置 >0）约束，最终都会自行退出；其迟到结果由
        drain() 丢弃。用于防止个别卡死的 worker 把主流程无限拖住。
        """
        now = time.time()
        with self._lock:
            timed_out = [t for t in self._inflight.values()
                         if now - t.submit_time > hard_timeout_seconds]
            for t in timed_out:
                self._inflight.pop(t.iteration, None)
        return timed_out


# ==========================================
# 工具函数
# ==========================================

def backup_original_code(target_file: str, logger: logging.Logger = None) -> str:
    """备份原始代码文件"""
    if not os.path.exists(target_file):
        if logger:
            logger.warning(f"⚠️  目标文件不存在，无需备份: {target_file}")
        return None
    
    # 创建备份目录
    backup_dir = os.path.join(os.path.dirname(target_file), ".autoresearch_backups")
    os.makedirs(backup_dir, exist_ok=True)
    
    filename = os.path.basename(target_file)
    
    try:
        with open(target_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 1. 创建固定名称的原始备份（用于作为后续迭代的 base_code）
        original_backup_path = os.path.join(backup_dir, f"{filename}.original")
        if not os.path.exists(original_backup_path):
            with open(original_backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
            if logger:
                logger.info(f"💾 已创建原始代码备份: {original_backup_path}")
            else:
                print(f"💾 已创建原始代码备份: {original_backup_path}")
        
        # 2. 创建带时间戳的备份（用于历史追溯）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamp_backup_path = os.path.join(backup_dir, f"{filename}.{timestamp}.backup")
        with open(timestamp_backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        if logger:
            logger.info(f"💾 已备份原始代码: {timestamp_backup_path}")
        else:
            print(f"💾 已备份原始代码: {timestamp_backup_path}")
        
        return timestamp_backup_path
    except Exception as e:
        if logger:
            logger.error(f"❌ 备份失败: {e}")
        else:
            print(f"❌ 备份失败: {e}")
        return None


class FingerprintExecutor:
    """
    代码指纹执行器 - 在完整评测前快速筛除明显的陷阱模式
    第一版只做静态代码检测 + 轻量级 import 验证
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def quick_run(self, target_patch_file: str) -> Tuple[bool, str]:
        """
        快速验证代码是否包含 uniform fallback / retain-all fallback 等危险模式。
        返回 (pass, msg)。
        注意：非 uniform/retain-all 的 import/runtime 错误视为通过（可能是 mock 环境不足）。
        """
        try:
            with open(target_patch_file, 'r', encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            return True, f"无法读取目标文件: {e}（视为通过）"

        # 1-3. 静态检测（G2：模式表从当前任务 patch_spec.fingerprint_patterns 读取，
        # 逐字 == 现正则与判定消息；空模式表 = 新任务跳过静态检测）
        from autoresearch_tasks import current_task  # 延迟导入防循环
        for pattern in current_task().patch_spec.fingerprint_patterns:
            flags = re.DOTALL if pattern.get("dotall") else 0
            if re.search(pattern["regex"], code, flags):
                return False, pattern["message"]

        # 4. 轻量级 import 验证（可选，非致命）
        # 尝试 import 并检查类是否存在
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("fingerprint_module", target_patch_file)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            # 只要能 import 成功，就不拦截
        except Exception as e:
            # import 失败可能是 mock 环境不足，不直接拦截
            self.logger.debug(f"Fingerprint import test non-fatal error: {e}")
            pass

        return True, "Fingerprint OK"


# 运行时维度不匹配的报错特征（PyTorch matmul/逐元素/cat 的常见文案），
# 供 TokenCountProbe 把跨空间维度混用从"非致命异常"升级为拦截项。
_DIM_MISMATCH_PATTERNS = (
    "size mismatch",
    "shapes cannot be multiplied",
    "Sizes of tensors must match",
    "must match the size of tensor",
    "The size of tensor a",
    "must have identical last dimension",
    "same number of dimensions",
)


def _is_dim_mismatch_error(exc: Exception) -> bool:
    """判断异常是否为张量维度/形状不匹配（跨空间混用的典型崩溃）。"""
    msg = str(exc)
    return any(p in msg for p in _DIM_MISMATCH_PATTERNS)


# [Probe] 子进程 runner 脚本（python -c 执行，sys.executable 与引擎同解释器）。
# 在独立进程中重建 TokenCountProbe 并跑 _verify_impl，结果以 JSON 行打印到
# stdout（PROBE_RESULT_JSON: 前缀）；父进程按前缀解析。
# argv: n_tokens H W retain_ratio trials target_patch_file device strict(0/1)
_PROBE_RUNNER_SNIPPET = r'''
import sys, json, logging
sys.path.insert(0, {common_dir!r})
from autoresearch_common import TokenCountProbe
_a = sys.argv[1:]
_p = TokenCountProbe(logging.getLogger("probe"),
                     n_tokens=int(_a[0]), grid=(int(_a[1]), int(_a[2])),
                     retain_ratio=float(_a[3]), trials=int(_a[4]))
_ok, _msg = _p._verify_impl(_a[5], device=(_a[6] or None), strict=(_a[7] == "1"))
print("PROBE_RESULT_JSON:" + json.dumps({{"ok": bool(_ok), "detail": _msg,
                                          "sig": getattr(_p, "_last_sig", None)}}))
'''


class TokenCountProbe:
    """运行时 token 数探针：用合成输入实际执行 selector，验证保留数不超过 target。

    背景：iter_461 (CAWR) 的 Ward 合并使用惰性堆，堆提前掏空时以 >k_i 个簇退出，
    每簇选一个代表 → 静默保留 5-6 倍于 target 的 token（动态超选）。
    静态 tampered-ratio 正则只能抓硬编码比例，抓不到这种运行时超选。

    v15：子进程隔离 + 超时（2026-08-02 事故：生成算法调 torch.lobpcg 迭代法
    eigensolver 不收敛，探针在主线程同步执行且无超时，引擎卡死 9+ 小时）。
    verify/verify_on_gpu 现在在独立子进程中运行 _verify_impl（见
    _PROBE_RUNNER_SNIPPET），超过 self.timeout 秒 kill 进程组并判定探针失败
    （走现有 pre-eval 失败/修复路径），绝不允许无限阻塞主线程。
    """

    def __init__(self, logger: logging.Logger, n_tokens: int = 144, grid=(12, 12),
                 retain_ratio: float = 0.05, trials: int = 4, timeout: int = 180):
        self.logger = logger
        self.n_tokens = n_tokens
        self.grid = grid
        self.k = max(1, int(round(n_tokens * retain_ratio)))
        self.trials = trials
        # 探针子进程超时（秒）；引擎以 config.probe_timeout_seconds 注入
        self.timeout = timeout
        # 行为指纹（功能克隆检测）：子进程选择签名经 payload 回传后写入；
        # _last_sig 在子进程 _verify_impl 内使用
        self.last_signature = None
        self._last_sig = None

    def verify(self, target_patch_file: str, device: Optional[str] = None,
               strict: bool = False) -> Tuple[bool, str]:
        """返回 (pass, msg)。子进程隔离运行 _verify_impl + 超时 kill。

        判定规则：子进程超时 / 异常退出 / 输出无法解析 → 一律探针失败
        （探针是 pre-eval 闸门，失败进修复循环，绝不允许无限阻塞）；
        子进程正常返回时透传 _verify_impl 的原有语义（超选/维度不匹配/strict
        拦截 → fail；无法探测 → pass）。
        GPU 可见性：device="cuda:N" 时经 CUDA_VISIBLE_DEVICES=N 传给子进程
        （子进程内该卡即 cuda:0）；device="cpu" 时屏蔽 GPU。
        """
        timeout = int(getattr(self, 'timeout', 180) or 180)
        self.last_signature = None  # 每次调用重置，防止上一轮指纹残留
        common_dir = os.path.dirname(os.path.abspath(__file__))
        snippet = _PROBE_RUNNER_SNIPPET.format(common_dir=common_dir)
        dev_str = device or ""
        child_device = dev_str
        env = os.environ.copy()
        if dev_str.startswith("cuda:"):
            env["CUDA_VISIBLE_DEVICES"] = dev_str.split(":", 1)[1]
            child_device = "cuda:0"
        elif dev_str == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        # retain_ratio 按 k/n_tokens 回传，子进程重建出相同的 k
        args = [sys.executable, "-c", snippet,
                str(self.n_tokens), str(self.grid[0]), str(self.grid[1]),
                repr(self.k / self.n_tokens), str(self.trials),
                target_patch_file, child_device, "1" if strict else "0"]
        popen_kwargs = {}
        if sys.platform != 'win32':
            popen_kwargs['preexec_fn'] = os.setsid
        try:
            proc = subprocess.Popen(
                args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, env=env, **popen_kwargs)
        except Exception as e:
            return False, f"[Probe] 子进程启动失败: {type(e).__name__}: {e}"
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                if sys.platform != 'win32':
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                else:
                    proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=10)
            except Exception:
                pass
            self.logger.warning(
                f"⏱️ [Probe] 超时 {timeout}s，判定失败（算法可能存在死循环/极慢迭代）"
            )
            return False, (
                f"PROBE_TIMEOUT: 探针子进程超过 {timeout}s 未完成，已强制终止。"
                f"算法很可能存在死循环或不收敛的迭代求解（如 torch.lobpcg 等迭代法 "
                f"eigensolver）。必须改用闭式解或固定迭代次数、带收敛保证的实现。"
            )
        for line in (stdout or "").splitlines():
            if line.startswith("PROBE_RESULT_JSON:"):
                try:
                    payload = json.loads(line[len("PROBE_RESULT_JSON:"):])
                    # 行为指纹（功能克隆检测）：仅探针通过时有效
                    if payload.get("ok") and payload.get("sig"):
                        self.last_signature = payload["sig"]
                    return bool(payload.get("ok")), str(payload.get("detail", ""))
                except Exception as e:
                    return False, f"[Probe] 子进程结果解析失败: {e}"
        err_tail = (stderr or "").strip().splitlines()[-5:]
        return False, (
            f"[Probe] 子进程异常退出（returncode={proc.returncode}，无结果输出），"
            f"判定失败: " + " / ".join(err_tail)[:500]
        )

    def _verify_impl(self, target_patch_file: str, device: Optional[str] = None,
                     strict: bool = False) -> Tuple[bool, str]:
        """进程内探针实现（原 verify 主体，逐字保留判定语义）。

        只能在子进程中运行（由 verify 经 _PROBE_RUNNER_SNIPPET 调用）——
        生成代码可能含死循环/不收敛迭代，直接在引擎主线程调用会卡死整个引擎。

        可选参数（默认 == 现行为逐字一致，CPU 合成输入、异常非致命）：
            device: 指定后在对应设备（如 "cuda:0"）上创建合成输入并运行冒烟，
                    selector 实例支持 .to() 时一并迁移；
            strict: [SmokeMode] 冒烟语义 = "真实跑通"——模块加载失败、forward 抛异常
                    均判 fail（冒烟模式没有正式评测兜底，冒烟是唯一闸门）。
        """
        try:
            import importlib.util
            import hashlib
            import torch
        except ImportError:
            return True, "torch 不可用，跳过 token 数探针"

        try:
            spec = importlib.util.spec_from_file_location("token_count_probe", target_patch_file)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as e:
            # import 失败交给后续环节报告，探针不拦截
            if strict:
                # [SmokeMode] 冒烟是唯一闸门：模块加载失败必须拦截进修复循环
                return False, f"SMOKE_FAILED: 模块加载失败: {type(e).__name__}: {e}"
            return True, f"模块加载失败（探针跳过）: {type(e).__name__}"

        K, N = self.k, self.n_tokens
        H, W = self.grid
        # 维度按真实空间关系设置（取小值省内存）：image_embeds 与 text_token_embedding
        # 同属 LLM 空间（同维），image_features 属 CLIP 空间（不同维），与真实模型
        # （如 Qwen2.5-VL 的 3584 vs 1280）关系一致——跨空间维度混用会在探针阶段
        # 提前暴露，而不是等到正式评测才崩。
        d_embeds = 64
        d_clip = 32
        over_selections = []
        probed = False
        # 行为指纹片段（功能克隆检测）：每个 trial/入口的选中索引集合哈希。
        # 代码文本不同但选择行为逐位相同 = 同一机制换皮（0.493/0.308 克隆事件）
        sig_parts = []

        # G2：探测入口从当前任务 patch_spec.probe_entries 读取（逐字 == 现两条路径）；
        # 空列表 = 新任务无 selector 入口，probed 保持 False 静默通过。
        from autoresearch_tasks import current_task  # 延迟导入防循环
        probe_entries = current_task().patch_spec.probe_entries

        for trial in range(self.trials + 1):
            # 附加一次 image_grid_thw=None 试验：生产环境存在不传 grid 的调用路径
            # （qwen2_5_VLmodel_mmtok.py:305），算法必须容忍 grid 缺失
            # （iter_1613 事故：代码假设 grid 恒在，评测中期才崩）。
            none_grid_trial = (trial == self.trials)
            torch.manual_seed(trial)
            vt = torch.randn(N, d_embeds, device=device)
            vc = torch.randn(N, d_clip, device=device)
            if trial % 2 == 1:
                # 聚簇/低秩输入：最易暴露惰性堆掏空、聚类残余等超选 bug
                base = torch.randn(4, d_clip, device=device)
                vc = base[torch.randint(0, 4, (N,), device=device)] + 0.01 * torch.randn(N, d_clip, device=device)
            grid = torch.tensor([1, H, W], device=device)

            # 合成输入占位符解析（probe_entries 中的 "__xxx__" 值）。
            # text_emb 惰性生成：保持现网顺序（ctor 之后、方法调用之前），避免改变 RNG 序列。
            _synth_cache = {}

            def _resolve(v):
                if not isinstance(v, str) or not v.startswith("__"):
                    return v
                if v == "__K__":
                    return K
                if v == "__vt__":
                    return vt
                if v == "__vc__":
                    return vc
                if v == "__grid__":
                    return None if none_grid_trial else grid
                if v == "__text_emb__":
                    if "text_emb" not in _synth_cache:
                        # 与 image_embeds 同维（同属 LLM 空间）
                        _synth_cache["text_emb"] = torch.randn(8, d_embeds, device=device)
                    return _synth_cache["text_emb"]
                return v

            for entry in probe_entries:
                # 类缺失 = 非该入口模式的文件，跳过；与方法调用内的 AttributeError
                # （如 tensor.append 笔误）区分开——后者是真实代码缺陷，必须返修，
                # 因此 getattr 与实例化/调用分开捕获
                try:
                    cls = getattr(mod, entry["class_name"])
                except AttributeError:
                    continue  # 非该入口模式的文件
                try:
                    inst = cls(**{k: _resolve(v) for k, v in entry.get("ctor_kwargs", {}).items()})
                    if device is not None and hasattr(inst, "to"):
                        try:
                            inst = inst.to(device)
                        except Exception:
                            pass  # 无参数 selector / .to 不支持时保持 CPU，由 forward 异常兜底
                    result = getattr(inst, entry["method"])(
                        *[_resolve(a) for a in entry.get("args", [])],
                        **{k: _resolve(v) for k, v in entry.get("kwargs", {}).items()})
                    if entry.get("tuple_result"):
                        idx, _ = result
                    else:
                        idx = result
                    probed = True
                    try:
                        _idx_sorted = sorted(
                            int(i) for i in (idx.tolist() if hasattr(idx, "tolist") else list(idx)))
                        sig_parts.append(
                            f"{entry.get('label', '?')}|t{trial}|"
                            f"{hashlib.md5(str(_idx_sorted).encode()).hexdigest()[:12]}")
                    except Exception:
                        pass  # 指纹采集失败不阻塞探针
                    if len(idx) > K:
                        over_selections.append((entry["label"], trial, len(idx)))
                except Exception as e:
                    # 跨空间维度混用（CLIP 空间 image_features × LLM 空间 text/embeds
                    # 直接运算）是真实代码缺陷，在探针阶段拦截返修，避免浪费一次完整评测
                    if _is_dim_mismatch_error(e):
                        return False, (
                            f"DIM_MISMATCH_DETECTED: 算法在合成输入上触发维度不匹配 "
                            f"({entry.get('label', '?')}/trial{trial}): {type(e).__name__}: {e}。"
                            f"image_features 是 CLIP 空间，维度与 image_embeds/text_token_embedding "
                            f"（LLM 空间）不同，禁止直接 matmul/点积/逐元素运算/拼接；"
                            f"必须先用无参数桥接（固定种子随机投影到公共维 / 标量统计条件化 / 秩权重）对齐。"
                        )
                    if strict:
                        # [SmokeMode] 冒烟语义 = "真实跑通"：forward 任何异常都判失败
                        return False, (
                            f"SMOKE_FAILED: selector 在合成输入 forward 时抛异常 "
                            f"({entry.get('label', '?')}/trial{trial}): {type(e).__name__}: {e}"
                        )
                    # 运行时代码缺陷（UnboundLocalError / tensor.append / len() 关键字参数 /
                    # .to() 参数错误 / 下标越界等）：原逻辑仅 debug 记录并放行，这类 bug
                    # 必然白烧一次完整评测（iter_1625/1626/1643/1645/1661/1682 同类事故）。
                    # 改为拦截返修。
                    return False, (
                        f"RUNTIME_BUG_DETECTED: 算法在合成输入上抛异常 "
                        f"({entry.get('label', '?')}/trial{trial}"
                        f"{', grid=None' if none_grid_trial else ''}): {type(e).__name__}: {e}。"
                        f"正式评测会以同样方式崩溃。请修复边界条件后重试：早返回分支变量未赋值、"
                        f"image_grid_thw 为 None（生产存在不传 grid 的调用路径）、"
                        f"张量与 Python 标量/list 混用、空张量。"
                    )

        if not probed:
            return True, "未找到 selector 入口（探针跳过）"
        if over_selections:
            detail = "; ".join(f"{p}/trial{t} 保留 {n}>{K}" for p, t, n in over_selections[:3])
            return False, (
                f"OVER_SELECTION_DETECTED: 算法实际保留的 token 数超过 target={K}（{detail}）。"
                f"保留超过 target_vision_tokens 的 token 属于作弊，会虚增 accuracy。"
            )
        # 行为指纹落 self._last_sig，由 snippet 装入 payload 回传父进程
        self._last_sig = (
            hashlib.md5("||".join(sig_parts).encode()).hexdigest()[:16] if sig_parts else None)
        return True, f"Token count OK (K={K}, {self.trials} trials)"

    def verify_on_gpu(self, target_patch_file: str, device: str) -> Tuple[bool, str]:
        """[SmokeMode] 在指定 GPU 上跑冒烟探针（显存闸由调用方负责）。

        冒烟内容保持 _verify_impl 的现有语义：真实实例化 patch 后的 selector、
        用合成输入跑一次 forward、校验输出 token 数与维度；区别在于子进程
        隔离 + 超时（同 verify）：选定卡经 CUDA_VISIBLE_DEVICES 传给子进程
        （子进程内即 cuda:0），strict=True（任何异常判失败），显存随子进程
        退出自然释放（不再需要在父进程 empty_cache）。
        """
        return self.verify(target_patch_file, device=device, strict=True)
