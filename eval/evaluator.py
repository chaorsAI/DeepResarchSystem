# evaluator.py    Agent 评估 编排器


from backend.agent.configuration import Configuration
from backend.agent.graph import graph
from eval.judger import *
from hook_context import HookContext
from backend.agent import agent

from __future__ import annotations
import json
import os
import sys
import textwrap
from dataclasses import dataclass, field
from dotenv import load_dotenv
from typing import Any, Dict, List

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


class WebSearchCapture:
    """
    业务实体：负责捕获搜索结果。
    通过 monkey-patch 注入 WebSearchAgent 的线程局部捕获容器。
    它只是一个普通的 Python 类，不依赖任何 Hook 框架。
    """

    def __init__(self):
        self.raw_results: List[Dict[str, Any]] = []

    def post_hook_handler(self, result: Any, args: tuple, kwargs: dict) -> Any:
        """
        符合 HookContext 约定的回调函数。
        注意：这个方法之所以叫 handler，是因为它是被动调用的。
        HookContext 约定了参数签名 (result, args, kwargs)。
        """
        # 业务假设：WebSearchAgent.step(self, prompt)
        if len(args) > 1:
            prompt = args[1]
            self.raw_results.append({
                "query": prompt,
                "pages": result
            })
        return result  # 必须返回 result，这是 Hook 契约的一部分

    def get_last(self) -> Dict[str, Any] | None:
        return self.raw_results[-1] if self.raw_results else None

def _get_raw_result(prompt: str):
    """
    无侵入 hook 获取 WebSearchAgent 查询到的原始内容
    评估服务：组装 Hook 框架和业务逻辑
    :param prompt:
    :return:
    """
    raw_data = None
    # 1. 实例化业务对象
    capture = WebSearchCapture()

    # 2. 实例化框架对象，并将业务对象的“方法”作为回调函数注入
    #    这里完成了依赖注入（DI）
    with HookContext(
            target=(agent.WebSearchAgent, "step"),
            post_hook=capture.post_hook_handler  # 注入回调，而非继承或组合
    ):
        # 3. 执行业务流程
        summary = agent.WebSearchAgent.step(prompt)

    # 4. 从业务对象中提取数据
    raw_data = capture.get_last()

    logger.info(f"Summary: {summary}")
    logger.info(f"Raw Data: {raw_data}")

    return raw_data

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
                results.append(self._eval_components(cfg))
            except Exception as exc:
                logger.error(f"组件级评估失败 '{cfg.topic[:60]}': {exc}")
                results.append(ComponentResult(topic=cfg.topic, error=str(exc)))
        return results

    def _eval_components(self, cfg: TopicCfg) -> ComponentResult:
        result = ComponentResult(topic=cfg.topic)

        # 通过共享辅助方法运行完整流水线（如配置了反馈则处理反馈）
        agent_result = self._invoke_search_agent(cfg)

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

            # 评估计划反思（仅在提供了用户反馈时）
            if cfg.user_feedback and cfg.expected_intent:
                result.plan_reflection_score = self.judger.evaluate_plan_reflection(
                    original_plan=plan_a,
                    user_feedback=cfg.user_feedback,
                    new_plan=plan_b,
                    actual_behavior=actual_behavior,
                    expected_intent=cfg.expected_intent,
                )

            # 评估搜索查询
            search_queries = phase2.get("search_query", [])
            if search_queries:
                query_list = list(search_queries) if isinstance(search_queries, list) else []
                result.query_score = self.judger.evaluate_queries(
                    research_topic=cfg.topic,
                    queries=query_list,
                    rationale="（内部推理未捕获；参见计划上下文）",
                )

                # 评估计划 → 查询对齐（针对用于研究的计划）
                if effective_plan and query_list:
                    result.plan_query_alignment_score = (
                        self.judger.evaluate_plan_query_alignment(
                            plan=effective_plan,
                            queries=query_list
                        )
                    )

            # 评估摘要保真度（针对每个捕获的原始搜索结果）
            web_search_results = phase2.get("web_search_result", [])
            for idx, (raw, summary) in enumerate(zip(raw_results, web_search_results)):
                if not raw or not summary:
                    continue
                score = self.judger.evaluate_summarization(
                    search_query=raw.get("query", ""),
                    raw_search_results=json.dumps(raw.get("pages", []), ensure_ascii=False, indent=2),
                    summary=str(summary),
                )
                if score:
                    result.summarization_scores.append(score)

            # 评估反思
            is_sufficient = phase2.get("is_sufficient")
            if is_sufficient is not None:
                result.critique_score = self.judger.evaluate_critique(
                    research_topic=cfg.topic,
                    summaries="\n---\n".join(
                        str(s) for s in phase2.get("web_search_result", [])
                    ),
                    is_sufficient=bool(is_sufficient),
                    knowledge_gap=phase2.get("knowledge_gap", ""),
                    follow_up_queries=phase2.get("follow_up_queries", []),
                )

            # 评估最终报告中的引用
            report = agent_result["report"]
            if report:
                sources = agent_result["sources"]
                if sources and sources != "[]":
                    result.citation_score = self.judger.evaluate_citations(
                        sources=sources, report=report
                    )

            return result


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