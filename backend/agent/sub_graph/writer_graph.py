#writer_graph.py    WriterAgent 子图
"""
概括了报告撰写流程：

1. 提纲 — 根据研究计划和材料设计章节结构    Outline

2. 草稿 — 为每个部分撰写内容   Draft

3. 评论员评审 — 对草稿进行评分并返回结构化反馈 Review

4. 引用与润色 — 将短链接替换为有效链接，修正格式，去除重复来源   Polish

这三步流程取代了原图中的单节点 final_answer，通过迭代改进生成更高质量的报告。
"""
from typer.cli import state

from backend.agent.agent import Agent, JsonAgent
from backend.agent.configuration import Configuration
from backend.agent.constant import SUB_WRITER_AGENT
from backend.agent.jsonUtils import JsonUtils
from backend.agent.prompts import (
    draft_instructions,
    outline_instructions,
    polish_instructions, review_instructions,
)
from backend.agent.utils import (
    get_current_date
)
from backend.agent.state import OverallState
from backend.agent.utils import get_research_topic
from backend.agent.tools_and_schemas import (
    OutlineModel,
    DraftModel,
    PolishModel, ReviewModel
)

from IPython.display import Image, display

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from loguru import logger


_OUTLINENODE = "outline_node"
_DRAFTNODE = "draft_node"
_REVIEWNODE = "review_node"
_POLISHNODE = "polish_node"

DEFAULT_MAX_REVISIONS = 3

def _outline(state: OverallState, config: RunnableConfig) -> dict:
    """
    根据研究主题和计划，生成大纲
    :param state:
    :param config:
    :return:
    """

    configuration = Configuration.runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configuration.answer_model
    logger.info(f"[WriterAgent] outline using model={reasoning_model}")

    agent = Agent(model_id=reasoning_model)
    agent.step_prompt(outline_instructions)
    raw = agent.step(
        research_topic=get_research_topic(state["messages"]),
        research_proposal=state.get("plan", ""),
        summaries="\n---\n\n".join(state["web_search_result"])
    )
    outline =- JsonUtils.extract_pattern(raw, pattern="markdown")
    logger.info(f"[WriterAgent] outline 已生成 ({len(outline)} 字)")

    return {
        "report_outline": outline,
        "revision_count": 0,
        "max_revisions": DEFAULT_MAX_REVISIONS,
    }

def _draft(state: OverallState, config: RunnableConfig) -> DraftModel:
    """
    根据大纲撰写正文草稿
    """

    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model
    logger.info(f"[WriterAgent] drafting using model={reasoning_model}")

    feedback = state.get("critic_feedback", "")
    outline = state.get("report_outline", "")
    is_revision = bool(feedback)
    revision_count = state.get("revision_count", 0) + (1 if is_revision else 0)

    if is_revision:
        logger.info(f"[WriterAgent] 修改稿 (revision {revision_count})")
        revision_context = (
            f"\n# 修订说明 (第 {revision_count} 次修订)\n"
            f"请根据以下审稿建议修改草稿：\n\n"
            f"{feedback}\n\n"
            f"请逐条处理上述问题，优先修复 critical 和 major 级别的问题。"
            f"请保留上版草稿中审稿人没有异议的内容。\n"
        )
        return_update = {"revision_count": revision_count, "critic_feedback": ""}
    else:
        logger.info(f"[WriterAgent] 从零开始撰写草稿")
        revision_context = ""
        return_update = {}

    agent = Agent(model_id=reasoning_model)
    agent.set_step_prompt(draft_instructions)
    raw = agent.step(
        current_date=get_current_date(),
        research_topic=get_research_topic(state["messages"]),
        research_proposal=state.get("plan", ""),
        outline=outline,
        summaries="\n---\n\n".join(state["web_search_result"]),
        revision_context=revision_context,
    )
    draft = JsonUtils.extract_pattern(raw, pattern="markdown")
    logger.info(f"[WriterAgent] draft 已生成 ({len(draft)} 字)")

    return {**return_update, "report_draft": draft}

def _review(state: OverallState, config: RunnableConfig) -> ReviewModel:
    """
    评论员审阅草稿并提供结构化反馈。
    使用 JsonAgent 和 CritiqueResult 模式生成结构化输出。
    """

    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model
    logger.info(f"[WriterAgent] critic reviewing draft using model={reasoning_model}")

    draft = state.get("report_draft", "")
    agent = JsonAgent(model_id=reasoning_model, keys=ReviewModel)
    agent.set_step_prompt(review_instructions)
    result: ReviewModel = agent.step(
        research_topic=get_research_topic(state["messages"]),
        research_proposal=state.get("plan", ""),
        summaries="\n---\n\n".join(state["web_search_result"]),
        draft=draft,
    )

    # 针对修改稿的点评反馈
    if result.issues:
        issues_text = "\n".join(
            f"- [{iss.severity.upper()}] {iss.location}: {iss.problem}\n"
            f"  建议: {iss.suggestion}"
            for iss in result.issues
        )
    else:
        issues_text = "无明显问题。"

    feedback = (
        f"## 审稿评分: {result.overall_rating}/10\n"
        f"## 综合评价: {result.summary}\n\n"
        f"## 具体问题:\n{issues_text}"
    )

    logger.info(
        f"[WriterAgent] 审稿评分={result.overall_rating}/10, "
        f"issues={len(result.issues)} "
        f"(critical={sum(1 for i in result.issues if i.severity == 'critical')}, "
        f"主要的={sum(1 for i in result.issues if i.severity == 'major')}, "
        f"次要的={sum(1 for i in result.issues if i.severity == 'minor')}), "
        f"准备润色={result.ready_for_polish}"
    )

    return {
        "critic_feedback": feedback,
        "critic_score": result.overall_rating,
        "ready_for_polish": result.ready_for_polish,
    }

def _route_after_review(state: OverallState, config: RunnableConfig) -> str:
    """
    决定：继续修改或进入终审润色。
    进入润色的条件：
      - Critic 明确标记 ready_for_polish，或
      - revision_count >= max_revisions（安全兜底）
    否则回到 draft 继续修改。
    """

    # 获取当前状态
    revision = state.get("revision_count", 0)
    max_rev = state.get("max_revisions", DEFAULT_MAX_REVISIONS)
    ready = state.get("ready_for_polish", False)

    if ready:
        logger.info(f"[WriterAgent] Critic ready_for_polish → polish")
        return _POLISHNODE

    if revision >= max_rev:
        logger.info(f"[WriterAgent] 已达到最大修改次数 ({revision}/{max_rev}) → polish")
        return _POLISHNODE

    logger.info(f"[WriterAgent] 需要重新撰写草稿 (rev={revision}/{max_rev}) → draft")
    return _DRAFTNODE

def _polish(state: OverallState, config: RunnableConfig) -> PolishModel:
    """
    将短链接替换为真实链接，删除重复来源，润色语言。
    """
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model
    logger.info(f"[WriterAgent] polishing using model={reasoning_model}")

    draft = state.get("report_draft", "")

    # Step A — LLM polish pass
    agent = Agent(model_id=reasoning_model)
    agent.set_step_prompt(polish_instructions)
    raw = agent.step(
        research_topic=get_research_topic(state["messages"]),
        draft=draft,
        summaries="\n---\n\n".join(state["web_search_result"]),
    )
    polished = JsonUtils.extract_pattern(raw, pattern="markdown")

    unique_sources = []
    for source in state.get("sources_gathered", []):
        if source["short_url"] in polished:
            polished = polished.replace(source["short_url"], source["value"])
            unique_sources.append(source)

    logger.info(f"[WriterAgent] 已润色 ({len(polished)} 字), {len(unique_sources)} 个引用来源")
    return {
        "messages": [AIMessage(content=polished)],
        "sources_gathered": unique_sources,
    }


_builder = StateGraph(OverallState, context_schema=Configuration)

# 写大纲
_builder.add_node(_OUTLINENODE, _outline)
# 写草稿
_builder.add_node(_DRAFTNODE, _draft)
# 草稿审核
_builder.add_node(_REVIEWNODE, _review)
# 润色出终稿
_builder.add_node(_POLISHNODE, _polish)

_builder.add_edge(START, _OUTLINENODE)
_builder.add_edge(_OUTLINENODE, _DRAFTNODE)
_builder.add_edge(_DRAFTNODE, _REVIEWNODE)
_builder.add_conditional_edges(_REVIEWNODE, _route_after_review, [_DRAFTNODE, _POLISHNODE])
_builder.add_edge(_POLISHNODE, END)

writer_agent_graph = _builder.compile(name=SUB_WRITER_AGENT)

display(Image(writer_agent_graph.get_graph().draw_mermaid_png(output_file_path="../graph_images/WriterAgent子图-带评审.png")))