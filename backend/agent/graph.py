# graph.py    Langgraph 图核心定义-主智能体定义

"""DeepResearch多智能体
由三个子智能体组成：
1. 计划阶段（主图，包含人机交互）
2. 研究智能体（子图）— 查询 → 搜索 → 评估循环
3. 写作智能体（子图）— 提纲 → 草稿 → 引用和润色
原有的单体图重构，每个阶段都成为一个自包含、可独立测试的子图。
"""

from typer.cli import state

from backend.agent.agent import JsonAgent
from backend.agent.constant import *
from backend.agent.state import *
from backend.agent.configuration import *
from backend.agent.tools_and_schemas import (
    SearchQueryList,
    Reflection,
    PlanReflection
)
from backend.agent.prompts import (
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
    plan_instructions,
    plan_reflection_instructions
)
from backend.agent.utils import (
    get_current_date,
    get_research_topic,
    resolve_urls,
    get_last_user_response
)
from backend.agent.agent import (
    Agent,
    JsonAgent,
    WebSearchAgent
)
from backend.agent.jsonUtils import *
from backend.agent.sub_graph import (
    research_agent_graph,
    writer_agent_graph
)

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
builder = StateGraph(OverallState, context_schema=Configuration)

# 图节点
builder.add_node(GENERATE_PLAN_NODE, generate_plan)
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

# 子图节点
builder.add_node(RESEARCH_AGENT_NODE, research_agent_graph)
builder.add_node(WRITER_AGENT_NODE, writer_agent_graph)

# 边定义
builder.add_edge(START, GENERATE_PLAN_NODE)
# 条件边：人类干预是否认可研究计划
builder.add_conditional_edges(GENERATE_PLAN_NODE, evaluate_plan, [RESEARCH_AGENT_NODE, SEARCH_REPLAN, AWAITING_PLAN_CONFIRMATION])
# 重新生成计划
builder.add_edge(SEARCH_REPLAN, GENERATE_PLAN_NODE)
# 子话题并行搜索
builder.add_edge(RESEARCH_AGENT_NODE, WRITER_AGENT_NODE)
builder.add_edge(WRITER_AGENT_NODE, END)

# 图编译
graph = builder.compile(name=DEEP_RESEARCH_AGENT)

display(Image(graph.get_graph().draw_mermaid_png(output_file_path='./graph_images/MAS_plan.png')))