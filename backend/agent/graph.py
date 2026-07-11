# graph.py    Langgraph 图核心定义


from constant import *
from state import *

from langgraph.graph import StateGraph


######---Agent 图定义 ---######
# ---边事件定义
def generate_plan() :
    return None

def generate_search(state : OverallState, config : CRITIQUE_NODE) -> QueryGenerationState :
    """
    基于用户的自然语言请求，拆解出搜索关键字
    使用LLM为用户的问题创建优化的网络搜索查询，用于网络研究。

    :param state:
    :param config:
    :return:
    """

def web_search() :
    return None
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
builder.add_node(GENERATE_PLAN_NODE, generate_plan)
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