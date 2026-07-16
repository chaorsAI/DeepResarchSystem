#reasearch_graph.py    ResearchAgent 子图
"""
封装了核心研究循环：

1. generate_queries — 将主题分解为搜索查询

2. web_search（并行扇出）— 搜索并汇总每个查询

3. critique — 评估信息是否充分；如有必要，则循环返回步骤 (1)

当评估认为信息充分或达到 max_research_loops 时，循环终止。
"""

from __future__ import annotations
import json
from dotenv import load_dotenv

from loguru import logger
from IPython.display import Image, display

from backend.agent.agent import Agent, JsonAgent, WebSearchAgent
from backend.agent.configuration import Configuration
from backend.agent.jsonUtils import JsonUtils
from backend.agent.prompts import (
    query_writer_instructions,
    reflection_instructions,
    web_searcher_instructions,
)
from backend.agent.utils import (
    get_current_date
)
from backend.agent.state import OverallState, QueryGenerationState, WebSearchState, ReflectionState
from backend.agent.tools_and_schemas import Reflection, SearchQueryList
from backend.agent.utils import get_research_topic, resolve_urls
from backend.agent.constant import *

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


def _generate_search(state : OverallState, config : RunnableConfig) -> QueryGenerationState :
    """
    基于用户 的自然语言请求，拆解出搜索关键字
    使用LLM为用户的问题创建优化的网络搜索查询，用于网络研究。
    :param state:
    :param config:
    :return:

    核心流程：
    1. 配置传入
    2. 大模型拆解：提示词构造、大模型通信、格式化输出
    3. 结果传递
    """

    logger.info(f"LangGraph节点开始运行.....，配置：[{config}]")
    configuration = Configuration.runnable_config(config)
    if state.get["initial_search_query_count"] is None:
        state["initial_search_query_count"] = configuration.number_of_initial_queries

    agent = JsonAgent(model_id=configuration.query_generator_model, keys=SearchQueryList)
    agent.set_step_prompt(query_writer_instructions)
    result = agent.step(
        current_date=get_current_date(),
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
        research_proposal=state.get("plan", "")     # 这里加了人类确定的研究计划
    )

    logger.info(f"待搜索内容生成结果: {result}")
    return {"search_query": result.query}

def _fan_out_to_web_search(state: QueryGenerationState) -> list[Send]:
    """Fan-out: 每个查询调用一次 web_search。"""
    return [
        Send(WEB_SEARCH_NODE, {"search_query": q, "id": int(idx)})
        for idx, q in enumerate(state["search_query"])
    ]

def _web_search(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """
    网络搜索节点
    Args:
        state: 包含搜索查询和研究循环计数的当前图状态
        config: 可运行配置，包括搜索API设置

    Returns:
        包含状态更新的字典，包括sources_gathered、research_loop_count和web_search_results
    """
    logger.info(f"开始网络搜索......")
    # 配置
    configurable = Configuration.runnable_config(config)
    web_searcher = WebSearchAgent()

    # 执行搜索
    response = web_searcher.step(prompt=state["search_query"],
                                 count=10)

    # 检查搜索结果是否为空或None
    if not response or response is None:
        logger.error(f"网络搜索返回为空: {state['search_query']}")
        return {
            "sources_gathered": [],
            "search_query": [state["search_query"]],
            "web_search_result": [f"未找到关于 '{state['search_query']}' 的搜索结果"],
        }

    # 长URL到短URL的映射：节省Token上下文长度
    # https://search.com/id/0 - 3 代替
    # 再最终报告中再替换回原始URL，保障引用准确性
    long2short_url_mappings = resolve_urls(response, state["id"])
    sources_gathered = [
        {"short_url": long2short_url_mappings[item["url"]], "value": item["url"], "label": item["title"]} for item in
        response]
    web_search_result = [
        {"snippet": item["snippet"], "title": item["title"], "url": long2short_url_mappings[item["url"]]} for item in
        response]
    web_search_result = json.dumps(web_search_result, ensure_ascii=False, indent=4)

    agent = Agent(model_id=configurable.query_generator_model)
    agent.set_step_prompt(web_searcher_instructions)
    modified_text = agent.step(
        query=state["search_query"],
        current_date=get_current_date(),
        web_search_result=web_search_result
    )
    modified_text = JsonUtils.extract_pattern(modified_text, pattern="text")

    logger.info(f"搜索标题: {state['search_query']}")
    logger.debug(f"网络搜索结果: {modified_text}")
    return {
        "sources_gathered": sources_gathered,
        "search_query": [state["search_query"]],
        "web_search_result": [modified_text],
    }

def _critique(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """
    识别知识差距并生成潜在后续查询的节点

    分析当前摘要以识别需要进一步研究的领域，并生成潜在的后续查询。
    使用结构化 输出来提取JSON格式的后续查询。

    :param state: 当前图状态
    :param config: 运行配置
    :return: 包含状态更新的字典，包括search_query键，包含生成的后续查询
    """

    logger.info(f"反思分析识别知识差距并生成潜在后续查询的节点工作......")
    configurable = Configuration.from_runnable_config(config)
    # 增加研究循环计数并获取推理模型
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model", configurable.reflection_model)
    logger.info(f"critique反思节点模型：{reasoning_model}")

    # 格式化输出
    agent = JsonAgent(model_id=reasoning_model, keys=Reflection)
    agent.set_step_prompt(reflection_instructions)
    result = agent.step(
        ccurrent_date=get_current_date(),
        number_queries=state["initial_search_query_count"],
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_search_result"]),
        research_proposal=state.get("plan", "")
    )

    logger.info(f"反思分析：{result}")
    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
        "max_research_loops": state.get("max_research_loops", configurable.max_research_loops),
    }

def _route_after_critique(state: OverallState, config: RunnableConfig):
    """
    critique 完成后的执行路有
    :param state:
    :param config:
    :return:
    """

    configuration = Configuration.runnable_config(config)
    max_loops = state.get("max_research_loops") or configuration.max_research_loops
    if state["is_sufficient"] or state["research_loop_count"] >= max_loops:
        logger.info(f"[ResearchAgent] 退出循环，已执行 {state['research_loop_count']} 次")
        return END
    else:
        logger.info(f"[ResearchAgent] 继续循环 ({state['research_loop_count']}/{max_loops})")
        return [
            Send(WEB_SEARCH_NODE,
                 {"search_query":query, "id":state["number_of_ran_queries"] + int(idx)})
            for idx, query in enumerate(state["follow_up_queries"])
        ]

_builder = StateGraph(OverallState, context_schema=Configuration)

_builder.add_node(GENERATE_SEARCH_NODE, _generate_search)
_builder.add_node(WEB_SEARCH_NODE, _web_search)
_builder.add_node(CRITIQUE_NODE, _critique)

_builder.add_edge(START, GENERATE_SEARCH_NODE)
_builder.add_conditional_edges(GENERATE_SEARCH_NODE, _fan_out_to_web_search, [WEB_SEARCH_NODE])
_builder.add_edge(WEB_SEARCH_NODE, CRITIQUE_NODE)
_builder.add_conditional_edges(CRITIQUE_NODE, _route_after_critique, [WEB_SEARCH_NODE, END])

research_agent_graph = _builder.compile(name=SUB_RESEARCH_AGENT)

display(Image(research_agent_graph.get_graph().draw_mermaid_png(output_file_path="../graph_images/ResearchAgent子图.png")))







