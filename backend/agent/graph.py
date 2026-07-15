# graph.py    Langgraph 图核心定义
from typer.cli import state

from agent import JsonAgent
from agent.constant import *
from agent.state import *
from agent.configuration import *
from agent.tools_and_schemas import (
    SearchQueryList,
    Reflection,
    PlanReflection
)
from agent.prompts import (
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
    plan_instructions,
    plan_reflection_instructions
)
from agent.utils import (
    get_current_date,
    get_research_topic,
    resolve_urls,
    get_last_user_response
)
from agent.agent import (
    Agent,
    JsonAgent,
    WebSearchAgent
)
from agent.jsonUtils import *


from loguru import logger
import json
from IPython.display import Image, display


from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableConfig
from langgraph.types import Send
from langchain_core.messages import AIMessage
from langgraph.types import interrupt
from langgraph.graph import StateGraph, START, END

######---Agent 图定义 ---######
# ---节点事件定义
def generate_plan(state: OverallState, config: RunnableConfig) -> OverallState:
    """
    生成研究计划的节点

    基于用户的问题和需求，使用LLM生成一个详细的研究计划。

    Args:
        state: 包含用户问题的当前图状态
        config: 可运行配置，包括LLM提供商设置

    Returns:
        包含计划状态更新的字典，包括plan和plan_status键
    """

    logger.info("正在生成计划...")
    if state.get("plan_status", "unconfirmed") != "unconfirmed":
        return {}

    configurable = Configuration.from_runnable_config(config)
    agent = Agent(model_id=configurable.query_generator_model)
    agent.set_step_prompt(plan_instructions)
    response = agent.step(
        current_date=get_current_date(),
        research_topic=get_research_topic(state["messages"], [msg.content for msg in state["plan_messages"]]),
        research_proposal=state.get("plan", "")
    )
    response = JsonUtils.extract_pattern(response, pattern="markdown")
    logger.info(state)
    logger.info(response)

    return {"messages": [AIMessage(content=response)],
            "plan": response,
            "plan_status": "unconfirmed",
            "plan_messages": [AIMessage(content=response)]}

def evaluate_plan(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """
    评估研究计划的节点

    分析当前计划并根据用户反馈决定下一步行动：
    - 如果用户确认计划，继续生成查询
    - 如果用户需要修改计划，重新生成
    - 如果用户未确认，等待确认

    Args:
        state: 包含计划信息的当前图状态
        config: 可运行配置

    Returns:
        字符串字面量，指示下一个要访问的节点
    """
    configurable = Configuration.from_runnable_config(config)
    plan = state.get("plan", None)
    if state.get("plan_status", "unconfirmed") == "unconfirmed":
        logger.info("等待用户确认...")
        return AWAITING_PLAN_CONFIRMATION

    if not plan:
        logger.info("没有计划需要评估。")
        return SEARCH_REPLAN
    else:
        context = get_last_user_response(state["messages"])
        # 用户意图明确
        if "开始研究" in context or "需求确认" in context:
            return GENERATE_SEARCH_NODE

        # 无程序化关键词，需要进行用户意图识别
        agent = JsonAgent(model_id=configurable.query_generator_model, keys=PlanReflection)
        agent.set_step_prompt(plan_reflection_instructions)
        result = agent.step(
            research_proposal=state.get("plan", ""),
            context=context,
        )
        if result.satisfy:
            return GENERATE_SEARCH_NODE
        return SEARCH_REPLAN

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
        number_queries=state["initial_search_query_count"],
        research_proposal=state.get("plan", "")     # 这里加了人类确定的研究计划
    )

    logger.info(f"待搜索内容生成结果: {result}")
    return {"search_query": result.query}

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

def critique(state: OverallState, config: RunnableConfig) -> ReflectionState:
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

def final_answer(state: OverallState, config: RunnableConfig):
    """
    创建结构良好的研究报告，包含适当的引用。
    :param state: 当前状态
    :param config: 运行配置
    :return:包含状态更新的字典，包括running_summary键，包含格式化的最终摘要和源
    """

    logger.info("最终答案准备生成........")
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model
    logger.info(f"final_answer最终答案节点模型：{reasoning_model}")

    # 格式化提示
    agent = Agent(model_id=reasoning_model)
    agent.set_step_prompt(answer_instructions)
    content = agent.step(
        current_date=get_current_date(),
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_search_result"]),
        research_proposal=state.get("plan", "")
    )

    # 用原始URL替换短URL，并将所有使用的URL添加到sources_gathered
    unique_sources = []
    for source in state["sources_gathered"]:
        if source["short_url"] in content:
            content = content.replace(
                source["short_url"], source["value"]
            )
            unique_sources.append(source)

    logger.info(f"最终确定答案：{content}")
    return {
        "messages": [AIMessage(content=content)],
        "sources_gathered": unique_sources,
    }

# ---条件边事件
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

def route_evaluate(state: OverallState, config: RunnableConfig) -> OverallState:
    """
    确定 critique 后下一步的路由函数

    通过决定是否继续收集信息状态"is_sufficient"或基于配置的最大research循环次数来最终确定摘要，
    从而控制research循环。
    :param state: 当前图状态
    :param config: 运行配置
    :return: 全量状态，指示下一个要访问的节点（"web_research"或"finalize_summary"）
    """

    logger.info("准备评估当前研究......")
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )

    logger.info(state)
    logger.info(f"最大研究循环数: {max_research_loops}")
    logger.info(f"当前已研究次数: {state['research_loop_count']}")
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return FINAL_ANSWER_NODE
    else:
        return [
            Send(
                WEB_SEARCH_NODE,
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


# ---图构建
builder = StateGraph(OverallState, context_schema=None)

# 图节点
builder.add_node(GENERATE_PLAN_NODE, generate_plan)
builder.add_node(GENERATE_SEARCH_NODE, generate_search)
builder.add_node(WEB_SEARCH_NODE, web_search)
builder.add_node(CRITIQUE_NODE, critique)
builder.add_node(FINAL_ANSWER_NODE, final_answer)
# 人类干预：Human-in-the-loop
# 方法1：手动模拟节点：业务层忙等，节点不退出，执行线程被持续占用。处理不好容易出死锁
# 等待确认，不做任何额外处理；用户确认后转发 state
builder.add_node(AWAITING_PLAN_CONFIRMATION, lambda state, config: state)
# 重新规划，忽略传入的 state，返回新的状态片段。
# LangGraph会自动将这个返回值合并（merge）到全局状态中。标记当前计划为“未确认”，通常用于用户拒绝原计划后，触发重新规划的逻辑。
builder.add_node(SEARCH_REPLAN, lambda state, config: {"plan_status": "unconfirmed"})

# 方法2：LangGraph 官方 interrupt 机制。能缩小范围明确用户意图，减少LLM意图识别的成本
# def awaiting_plan_confirmation(state: OverallState):
#     # interrupt的参数会直接透传给前端，返回值就是用户的输入
#     feedback = interrupt({
#         "type": "plan_confirm",
#         "current_plan":state.get("plan"),
#         "prompt": "请确认当前执行计划，同意请输入confirm，拒绝请输入reject"
#     })
#
#     # 将用户反馈写入状态，供下游节点使用
#     return {"user_feedback": feedback}
#
# # 基于用户真实反馈决策
# def search_replan(state: OverallState):
#     if state["user_feedback"] == "reject":
#         return {"plan_status": "unconfirmed", "plan": None}  # 清空旧计划触发重规划
#     return {"plan_status": "confirmed"}
#
# builder.add_node(AWAITING_PLAN_CONFIRMATION, awaiting_plan_confirmation)
# builder.add_node(SEARCH_REPLAN, search_replan)

# 边定义
builder.add_edge(START, GENERATE_PLAN_NODE)
# 条件边：人类干预是否认可研究计划
builder.add_conditional_edges(GENERATE_PLAN_NODE, evaluate_plan, [GENERATE_SEARCH_NODE, SEARCH_REPLAN, AWAITING_PLAN_CONFIRMATION])
# 重新生成计划
builder.add_edge(SEARCH_REPLAN, GENERATE_PLAN_NODE)
# 子话题并行搜索
builder.add_conditional_edges(GENERATE_SEARCH_NODE, send_to_web_search,path_map=[WEB_SEARCH_NODE])
builder.add_edge(WEB_SEARCH_NODE, CRITIQUE_NODE)
builder.add_conditional_edges(CRITIQUE_NODE, route_evaluate, path_map=[WEB_SEARCH_NODE, FINAL_ANSWER_NODE])
# 最终确定答案
builder.add_edge(FINAL_ANSWER_NODE, END)

# 图编译
graph = builder.compile(name=DEEP_RESEARCH_AGENT)

display(Image(graph.get_graph().draw_mermaid_png(output_file_path='./graph_images/HITL版.png')))