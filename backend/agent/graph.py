# graph.py    Langgraph 图核心定义
from backend.agent.agent import JsonAgent
from constant import *
from state import *
from configuration import *
from tools_and_schemas import SearchQueryList, Reflection
from prompts import (
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions
)
from utils import (
    get_current_date,
    get_research_topic,
    resolve_urls
)
from agent import (
    Agent,
    JsonAgent,
    WebSearchAgent
)
from jsonUtils import *

from loguru import logger
import json

from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableConfig
from langgraph.types import Send


######---Agent 图定义 ---######
# ---边事件定义
def generate_plan() :
    return None

def generate_search(state : OverallState, config : RunnableConfig) -> QueryGenerationState :
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
        number_queries=state["initial_search_query_count"]
    )

    logger.info(f"待搜索内容生成结果: {result}")
    return {"search_query": result.query}

def send_to_web_search(state: QueryGenerationState):
    """
    为每个搜索查询生成n个网络研究节点，实现并行搜索。

    Args:
        state: 包含搜索查询的查询生成状态

    Returns:
        发送到web_search节点的消息列表
    """
    logger.info(f"准备发送请求给网络搜索节点......")
    return [
        Send(WEB_SEARCH_NODE, {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]


def web_search(state: WebSearchState, config: RunnableConfig) -> OverallState:
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
    configurable = Configuration.from_runnable_config(config)
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

    # 长URL到短URL的映射
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
    modified_text = agent.step(query=state["search_query"], current_date=get_current_date(),
                               web_search_result=web_search_result)
    modified_text = JsonUtils.extract_pattern(modified_text, pattern="text")

    logger.info(f"搜索标题: {state['search_query']}")
    logger.debug(f"网络搜索结果: {modified_text}")
    return {
        "sources_gathered": sources_gathered,
        "search_query": [state["search_query"]],
        "web_search_result": [modified_text],
    }

def critique() :
    return None
def final_answer() :
    return None

# ---条件边事件
def send_to_web_search() :
    return None
def route_evaluate() :
    return None

# ---图构建
builder = StateGraph(OverallState, config_schema=None)

# 图节点
builder.add_node(GENERATE_SEARCH_NODE, generate_search)
builder.add_node(WEB_SEARCH_NODE, web_search)
builder.add_node(CRITIQUE_NODE, critique)
builder.add_node(FINAL_ANSWER_NODE, final_answer)

# 边定义
builder.add_edge(START, GENERATE_PLAN_NODE)
builder.add_conditional_edges(GENERATE_SEARCH_NODE, send_to_web_search,path_map=[WEB_SEARCH_NODE])
builder.add_edge(WEB_SEARCH_NODE, CRITIQUE_NODE)
builder.add_conditional_edges(CRITIQUE_NODE, route_evaluate, path_map=[WEB_SEARCH_NODE, FINAL_ANSWER_NODE])
# 最终确定答案
builder.add_edge(FINAL_ANSWER_NODE, END)

# 图编译
builder.compile(name=DEEP_RESEARCH_AGENT)