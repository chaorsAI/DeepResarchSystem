#writer_graph.py    WriterAgent 子图
"""
概括了报告撰写流程：

1. 提纲 — 根据研究计划和材料设计章节结构    Outline

2. 草稿 — 为每个部分撰写内容   Draft

3. 引用与润色 — 将短链接替换为有效链接，修正格式，去除重复来源   Polish

这三步流程取代了原图中的单节点 final_answer，通过迭代改进生成更高质量的报告。
"""
from typer.cli import state

from backend.agent.agent import Agent
from backend.agent.configuration import Configuration
from backend.agent.constant import SUB_WRITER_AGENT
from backend.agent.jsonUtils import JsonUtils
from backend.agent.prompts import (
    draft_instructions,
    outline_instructions,
    polish_instructions,
)
from backend.agent.utils import (
    get_current_date
)
from backend.agent.state import OverallState
from backend.agent.utils import get_research_topic
from backend.agent.tools_and_schemas import (
    OutlineModel,
    DraftModel,
    PolishModel
)

from IPython.display import Image, display

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from loguru import logger


_OUTLINENODE = "outline_node"
_DRAFTNODE = "draft_node"
_POLISHNODE = "polish_node"


def _outline(state: OverallState, config: RunnableConfig) -> OutlineModel:
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

    return {"report_outline": outline}

def _draft(state: OverallState, config: RunnableConfig) -> DraftModel:
    """
    根据大纲撰写正文草稿
    """

    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model
    logger.info(f"[WriterAgent] drafting using model={reasoning_model}")

    outline_text = state.get("report_outline", "")

    agent = Agent(model_id=reasoning_model)
    agent.set_step_prompt(draft_instructions)
    raw = agent.step(
        current_date=get_current_date(),
        research_topic=get_research_topic(state["messages"]),
        research_proposal=state.get("plan", ""),
        outline=outline_text,
        summaries="\n---\n\n".join(state["web_search_result"]),
    )
    draft = JsonUtils.extract_pattern(raw, pattern="markdown")
    logger.info(f"[WriterAgent] draft 已生成 ({len(draft)} 字)")

    return {"report_draft": draft}

def _polish(state: OverallState, config: RunnableConfig) -> PolishModel:
    """
    将短链接替换为真实链接，删除重复来源，润色语言。
    """
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model
    logger.info(f"[WriterAgent] polishing using model={reasoning_model}")

    draft_text = state.get("report_draft", "")

    # Step A — LLM polish pass
    agent = Agent(model_id=reasoning_model)
    agent.set_step_prompt(polish_instructions)
    raw = agent.step(
        research_topic=get_research_topic(state["messages"]),
        draft=draft_text,
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

_builder.add_node(_OUTLINENODE, _outline)
_builder.add_node(_DRAFTNODE, _draft)
_builder.add_node(_POLISHNODE, _polish)

_builder.add_edge(START, _OUTLINENODE)
_builder.add_edge(_OUTLINENODE, _DRAFTNODE)
_builder.add_edge(_DRAFTNODE, _POLISHNODE)
_builder.add_edge(_POLISHNODE, END)

writer_agent_graph = _builder.compile(name=SUB_WRITER_AGENT)

display(Image(writer_agent_graph.get_graph().draw_mermaid_png(output_file_path="../graph_images/WriterAgent子图.png")))