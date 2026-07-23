# judger.py    Agent 评估员:LLM-as-Judge



from __future__ import annotations

from backend.agent.agent import Agent, JsonAgent
from backend.agent.jsonUtils import JsonUtils
from eval.judge_prompts import *

import json
import os
import traceback
from typing import Any
from pydantic import BaseModel, Field
import sys as _sys

from loguru import logger

from backend.agent.configuration import get_judge_model_id


# ---------- Judger 结构化输出 ----------
# 打分类
class JudgeScore(BaseModel):
    score: int = Field(ge=1, le=5)
    reason: str

# E2E评分  对应 E2E_JUDGE_INSTRUCTIONS 标准
class E2EScore(BaseModel):
    factual_accuracy: JudgeScore        # 事实准确性
    information_coverage: JudgeScore    # 信息覆盖度
    logical_structure: JudgeScore       # 逻辑性
    timeliness: JudgeScore              # 时效性
    citation_quality: JudgeScore        # 引用质量
    overall_score: float                # 总分
    overall_assessment: str             # 整体评价
    hallucination_check: dict           # 幻觉检查

class PlanScore(BaseModel):
    requirement_coverage: JudgeScore
    question_clarity: JudgeScore
    structure_quality: JudgeScore
    overall_score: float
    missing_dimensions: list[str] = Field(default_factory=list)
    assessment: str


class QueryScore(BaseModel):
    coverage: JudgeScore
    independence: JudgeScore
    search_friendliness: JudgeScore
    overall_score: float
    missing_angles: list[str] = Field(default_factory=list)
    assessment: str


class SummarizationScore(BaseModel):
    factual_fidelity: JudgeScore
    key_info_extraction: JudgeScore
    source_attribution: JudgeScore
    overall_score: float
    hallucinations: list[str] = Field(default_factory=list)
    assessment: str


class CritiqueScore(BaseModel):
    sufficiency_judgment: JudgeScore
    gap_identification: JudgeScore
    follow_up_query_quality: JudgeScore
    overall_score: float
    is_sufficiency_correct: bool
    assessment: str


class CitationPerRef(BaseModel):
    """单条引用审计记录。"""
    url: str
    label: str = ""
    paragraph_summary: str = ""
    source_title: str = ""
    status: str = ""  # valid | weak | content_mismatch | url_not_found
    reason: str = ""


class CitationSummaryStats(BaseModel):
    valid_rate: float = 0.0
    most_common_issue: str = ""
    worst_offender_url: str = ""


class CitationScore(BaseModel):
    total_citations: int
    valid_citations: int
    weak_citations: int = 0
    invalid_citations: int
    per_citation: list[CitationPerRef] = Field(default_factory=list)
    citation_accuracy_score: int = Field(ge=1, le=5)
    summary_stats: CitationSummaryStats | None = None
    assessment: str


class PlanQueryAlignmentScore(BaseModel):
    coverage_consistency: JudgeScore
    plan_fidelity: JudgeScore
    structural_decomposition: JudgeScore
    overall_score: float
    covered_dimensions: list[str] = Field(default_factory=list)
    missed_dimensions: list[str] = Field(default_factory=list)
    cross_reference_table: list[dict] = Field(default_factory=list)
    assessment: str


class PlanReflectionScore(BaseModel):
    intent_recognition: JudgeScore
    feedback_incorporation: JudgeScore
    overall_score: float
    actual_behavior: str = ""
    assessment: str


def _safe_format(template: str, **kwargs) -> str:
    """
    解决极端边界场景：被替换的“值”里，恰好包含了“占位符”的字符串。

    使用两阶段标记方法，使得某个值中包含另一个键的占位符字符串
    （例如报告中包含字面文本 ``{research_topic}``）不会被意外替换。
    1. 将占位符替换为固定前缀+随机id 生成的 key
    2. 记录 新占位符 和对应 value 的 关系
    3. 再把 新占位符 替换成 value

    eg：
    research_topic = "2024年大模型监管政策"
    report = "国内监管趋严，需参考{research_topic}的国际对比部分，建议跟进欧盟AI法案。"
    # 注意：report里的{research_topic}是普通字符串，不是要替换的占位符！

    ❌直接替换：
    # 研究主题
    【2024年大模型监管政策】
    # 待评估的报告
    国内监管趋严，需参考【2024年大模型监管政策】的国际对比部分，建议跟进欧盟AI法案。

    ✅安全替换：
    # 研究主题
    【2024年大模型监管政策】

    # 待评估的报告
    国内监管趋严，需参考【{research_topic}】的国际对比部分，建议跟进欧盟AI法案。
    """
    import uuid as _uuid
    markers: dict[str, str] = {}
    for key, value in kwargs.items():
        marker = f"__FMT_{_uuid.uuid4().hex}__"
        template = template.replace("{" + key + "}", marker)
        markers[marker] = str(value)
    for marker, value in markers.items():
        template = template.replace(marker, value)
    return template


# ---------- Judger 类 ----------
class Judger:
    """
    传入评分提示词调用 LLM 评价
    """
    def __init__(self, model_id: str | None = None):
        self.model = model_id or os.getenv("EVAL_MODEL", os.getenv("JUDGE_MODEL", ""))
        if not self.model:
            # 回退到可用模型列表中的最后一个模型
            self.model = get_judge_model_id()
        logger.info(f"Judge 已初始化，模型={self.model}")

    def _call(self, prompt: str) -> dict[str, Any]:
        """
        调用 LLM 评估。
        """
        agent = Agent(model_id=self.model)
        last_raw = ""
        for attempt in range(3):
            try:
                raw = agent(prompt)
                last_raw = raw
                json_str = JsonUtils.extract_pattern(raw, pattern="json")
                result = json.loads(json_str)
                _sys.stderr.write(f"[Judge] 第 {attempt + 1} 次尝试成功，"
                                  f"解析出 {len(result)} 个顶层键\n")
                return result
            except Exception:
                _sys.stderr.write(
                    f"[Judge] 第 {attempt + 1} 次尝试失败\n"
                    f"  raw[:500]: {last_raw[:500]}\n"
                    f"  错误: {traceback.format_exc()}\n"
                )
                continue
        _sys.stderr.write("[Judge] 全部 3 次尝试均失败，返回 {}\n")
        return {}

    # -- 端到端 --
    def evaluate_report(
        self, *, research_topic: str, search_sources: str, report: str
    ) -> E2EScore:
        prompt = _safe_format(E2E_JUDGE_INSTRUCTIONS,
            research_topic=research_topic,
            search_sources=search_sources,
            report=report,
        )
        result = self._call(prompt)
        return E2EScore(**result) if result else None

    # -- 组件级评估 --
    def evaluate_plan(self, *, research_topic: str, plan: str) -> PlanScore:
        """
        评估 计划 部分
        :param research_topic:
        :param plan:
        :return:
        """
        prompt = _safe_format(
            PLAN_JUDGE_INSTRUCTIONS,
            research_topic=research_topic,
            # 上下文截断，防止 LLM 出现严重的位置偏见。这里更好的方式是先进行摘要
            plan=plan[:8000]
        )
        result = self._call(prompt)
        return PlanScore(**result) if result else None

    def evaluate_queries(self, *, queries: list[str], reason: str) -> QueryScore:
        """
        评估 查询 部分
        :param queries: 
        :param reason: 
        :return: 
        """
        return None

    def evaluate_summarization(
            self, *, search_query: str, raw_search_results: str, summary: str
    ) -> SummarizationScore:
        """
        摘要 部分评估
        :param search_query:
        :param raw_search_results:
        :param summary:
        :return:
        """
        return None

    def evaluate_critique(
            self,
            *,
            research_topic: str,
            summaries: str,
            is_sufficient: bool,
            knowledge_gap: str,
            follow_up_queries: list[str],
    ) -> CritiqueScore:
        """
        反思审计 部分评估
        :param research_topic:
        :param summaries:
        :param is_sufficient:
        :param knowledge_gap:
        :param follow_up_queries:
        :return:
        """
        return None

    def evaluate_citations(self, *, sources: str, report: str) -> CitationScore:
        """
        引用 部分评估
        :param sources:
        :param report:
        :return:
        """
        return None

    def evaluate_plan_query_alignment(
            self, *, plan: str, queries: list[str]
    ) -> PlanQueryAlignmentScore:
        """
        计划/查询 对齐性评估
        :param plan:
        :param queries:
        :return:
        """
        return None

    def evaluate_plan_reflection(
            self, *,
            original_plan: str,
            user_feedback: str,
            new_plan: str,
            actual_behavior: str,
            expected_intent: str
    ) -> PlanReflectionScore:
        """
        计划反思/需求澄清 部分评估
        :param original_plan:
        :param user_feedback:
        :param new_plan:
        :param actual_behavior:
        :param expected_intent:
        :return:
        """
        return None

