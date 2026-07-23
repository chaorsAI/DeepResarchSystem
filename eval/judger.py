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