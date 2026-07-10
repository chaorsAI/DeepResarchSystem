# graph.py    Langgraph 图核心定义


from constant import *
from state import *

from langgraph.graph import StateGraph


# Agent 图定义
builder = StateGraph(OverallState, config_schema=None)

# 图节点
builder.add_node(GENERATE_PLAN_NODE, generate_plan)
builder.add_node(GENERATE_SEARCH_NODE, generate_search)
builder.add_node(WEB_SEARCH_NODE, web_search)
builder.add_node(CRITIQUE_NODE, critique)
builder.add_node(FINAL_ANSWER_NODE, final_answer)

# 边定义
builder.add_edge(START, GENERATE_PLAN_NODE)
# ...
# 最终确定答案
builder.add_edge(FINAL_ANSWER_NODE, END)

# 图编译
builder.compile(name=DEEP_RESEARCH_AGENT)