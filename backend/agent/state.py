# state.py    LangGraph 节点定义

from dataclasses import dataclass, field
from typing import TypedDict, Annotated
import operator

from langgraph.graph import add_messages


# 总状态
class OverallState(TypedDict):
    # 消息队列
    messages: Annotated[list, add_messages]
    # 查询消息列表
    search_query: Annotated[list, operator.add]
    # 网搜结果列表
    web_search_result: Annotated[list, operator.add]
    # 网搜结果来源
    sources_gathered: Annotated[list, operator.add]
    # 初始搜索查询数量
    initial_search_query_count: int
    # 最大研究循环次数
    max_research_loops: int
    # 研究循环次数
    research_loop_count: int
    # 推理模型
    reasoning_model: str
    # 人类参与后的相关字段
    # 计划内容
    plan: str
    # 计划状态：1待确认、2已确认、3重新生成基础
    plan_status: str
    # 计划相关消息
    plan_messages: Annotated[list, add_messages]

# 反思节点状态
class ReflectionState(TypedDict):
    # 研究是否足够
    is_sufficient : bool
    # 知识差距描述
    knowledge_gap : str
    # 后续查询
    follow_up_queries : Annotated[list, operator.add]
    # 研究循环次数
    research_loop_count : int
    # 已经执行的研究次数
    number_of_ran_requeries : int
    # 最大研究循环次数
    max_research_loops : int

# 查询类
class Query(TypedDict) :
    # 搜索关键词
    query : str
    # 选择关键词的原因：为了让LLM更能信服，也和反思机制有关
    reason : str

class QueryGenerationState(TypedDict):
    search_query: list[Query]

# 网络搜索状态
class WebSearchState(TypedDict):
    search_query: str
    id: str


@dataclass(kw_only = True)
# Final report
class SearchStateOutput :
    running_summary : str = field(default=None)
