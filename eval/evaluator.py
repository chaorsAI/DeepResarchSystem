# evaluator.py    Agent 评估 编排器


from backend.agent.configuration import Configuration
from backend.agent.graph import graph
from eval.judger import *

from __future__ import annotations
import json
import os
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Any
from dotenv import load_dotenv

from loguru import logger

from langchain_core.messages import AIMessage, HumanMessage

# ---------- 数据类 ----------
@dataclass
class TopicCfg:
    """
    每个主题的评估运行配置。
    """
    topic: str
    initial_search_query_count: int = 2
    max_research_loops: int = 2
    user_feedback: str | None = None
    expected_intent: str | None = None

@dataclass
class E2EResult:
    """
    端到端评估结果
    """
    topic: str
    report: str = ""
    sources: str = ""  # JSON 序列化的 sources_gathered
    score: E2EScore | None = None
    error: str | None = None

@dataclass
class ComponentResult:
    topic: str
    plan_score: PlanScore | None = None
    plan_query_alignment_score: PlanQueryAlignmentScore | None = None
    query_score: QueryScore | None = None
    summarization_scores: list[SummarizationScore] = field(default_factory=list)
    critique_score: CritiqueScore | None = None
    citation_score: CitationScore | None = None
    plan_reflection_score: PlanReflectionScore | None = None
    error: str | None = None

@dataclass
class EvalReport:
    """
    评估报告
    """
    timestamp: str
    e2e_results: list[E2EResult] = field(default_factory=list)
    component_results: list[ComponentResult] = field(default_factory=list)

# ---------- 评估编排器 ----------
class Evaluator:
    """
    评估编排器
    """
    def __init__(self, judge_model_id: str | None = None):
        self.judger = Judger(model_id=judge_model_id)

    # --- 端到端 ---
    def run_e2e(self, topics: list[TopicCfg]) -> list[E2EResult]:
        """
        对每个主题运行完整的 agent 并对最终报告进行评分。
        """
        results: list[E2EResult] = []
        for i, cfg in enumerate(topics):
            logger.info(f"端到端 [{i + 1}/{len(topics)}] 主题={cfg.topic[:80]}...")
            try:
                # 调用 研究 Agent
                result = self._invoke_search_agent(cfg)
                if result.error:
                    results.append(result)
                    continue

                # 调用真正的 LLM 评估器
                result.score = self.judger.evaluate_report(
                    research_topic=cfg.topic,
                    search_sources=result.sources,
                    report=result.report,
                )
                results.append(result)
                logger.info(
                    f"  总评分={result.score.overall_score if result.score else '无'}"
                )
            except Exception as exc:
                logger.error(f"端到端评估失败 '{cfg.topic[:60]}': {exc}")
                results.append(E2EResult(topic=cfg.topic, error=str(exc)))
        return results

    # --- 端到端 ---
    def run_components(self, topics: list[TopicCfg]) -> list[ComponentResult]:
        """
        对每个主题独立评估各个 agent 节点。
        """
        results: list[ComponentResult] = []
        for i, cfg in enumerate(topics):
            logger.info(f"组件级 [{i + 1}/{len(topics)}] 主题={cfg.topic[:80]}...")
            try:
                results.append(self._invoke_search_agent(cfg))
            except Exception as exc:
                logger.error(f"组件级评估失败 '{cfg.topic[:60]}': {exc}")
                results.append(ComponentResult(topic=cfg.topic, error=str(exc)))
        return results

    def _eval_components(self, cfg: TopicCfg) -> ComponentResult:
        result = ComponentResult(topic=cfg.topic)

        # 通过共享辅助方法运行完整流水线（如配置了反馈则处理反馈）
        agent_result = self._invoke_agent_with_feedback(cfg)

        # 原始计划
        plan_a = agent_result["plan_a"]
        # 原始计划不通过，重新规划的计划
        plan_b = agent_result["plan_b"]
        # 实际执行行为
        actual_behavior = agent_result["actual_behavior"]
        phase2 = agent_result["phase2_state"]

        # 实际用于研究的计划
        effective_plan = plan_b if plan_b else plan_a

        # 评估计划（用于研究的那个）
        if effective_plan:
            result.plan_score = self.judge.evaluate_plan(
                research_topic=cfg.topic, plan=effective_plan
            )

    def _invoke_search_agent(self, cfg: TopicCfg) -> E2EResult:
        """
        运行 研究 agent。

        当设置了 *cfg.user_feedback* 时，计划确认阶段通过 LLM 意图识别路径
        进行测试；否则使用现有的自动确认行为。
        """


# ---------- 报告格式化 ----------
def format_eval_report(report: EvalReport) -> str:
    """
    渲染一份人类可读的评估摘要。
    """
    lines = ["=" * 72, "  DeepResearch Agent 评估报告", "=" * 72, ""]

    # 端到端摘要
    if report.e2e_results:
        lines.append("--- 端到端报告得分 ---")
        lines.append("")
        scores = []
        for r in report.e2e_results:
            if r.score:
                scores.append(r.score)
                lines.append(f"  主题: {r.topic[:80]}")
                lines.append(f"    总评分: {r.score.overall_score:.1f}/5")
                lines.append(f"    事实准确性: {r.score.factual_accuracy.score}/5")
                lines.append(f"    信息覆盖度: {r.score.information_coverage.score}/5")
                lines.append(f"    逻辑结构:   {r.score.logical_structure.score}/5")
                lines.append(f"    时效性:     {r.score.timeliness.score}/5")
                lines.append(f"    引用质量:   {r.score.citation_quality.score}/5")
                lines.append(
                    f"    幻觉:       {'有' if r.score.hallucination_check.get('has_hallucinations') else '无'}"
                )
                lines.append("")
            elif r.error:
                lines.append(f"  主题: {r.topic[:80]}  错误: {r.error[:120]}")
                lines.append("")

        if scores:
            avg = sum(s.overall_score for s in scores) / len(scores)
            lines.append(f"  ** 平均总评分: {avg:.1f}/5 (n={len(scores)}) **")
            lines.append("")

    return "\n".join(lines)

# 保存报告
def save_eval_report(report: EvalReport, path: str = "eval_report.json") -> None:
    """
    将完整评估数据保存为 JSON 文件以供进一步分析。
    """

    def _serialize(obj):
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, default=_serialize, ensure_ascii=False, indent=2)
    logger.info(f"完整评估报告已保存至 {path}")