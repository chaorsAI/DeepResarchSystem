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
class EvalReport:
    """
    评估报告
    """
    timestamp: str
    e2e_results: list[E2EResult] = field(default_factory=list)


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

    def _invoke_search_agent(self, cfg: TopicCfg) -> E2EResult:
        """
        运行 研究 agent。

        当设置了 *cfg.user_feedback* 时，计划确认阶段通过 LLM 意图识别路径
        进行测试；否则使用现有的自动确认行为。
        """