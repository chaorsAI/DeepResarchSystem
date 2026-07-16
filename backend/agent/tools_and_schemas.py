from typing import List
from pydantic import BaseModel, Field


class SearchQueryList(BaseModel):
    query: List[str] = Field(
        description="用于web搜索的搜索查询列表."
    )
    rationale: str = Field(
        description="简要解释为什么这些问题与研究主题相关."
    )


class Reflection(BaseModel):
    is_sufficient: bool = Field(
        description="所提供的概要是否足以回答用户的问题."
    )
    knowledge_gap: str = Field(
        description="描述哪些信息缺失或需要澄清."
    )
    follow_up_queries: List[str] = Field(
        description="解决知识差距的后续查询列表."
    )

class PlanReflection(BaseModel):
    satisfy: bool = Field(
        description="用户对生成的研究计划是否满意."
    )

class OutlineModel(BaseModel):
    report_outline: str = Field(
        description="子图大纲节点根据资料生成的大纲内容."
    )

class DraftModel(BaseModel):
    report_draft: str = Field(
        description="子图草稿节点根据大纲生成的草稿内容."
    )

class PolishModel(BaseModel):
    messages: List[str] = Field(
        description="子图审稿节点根据草稿润色确定最终报告."
    )
    sources_gathered: List[str] = Field(
        description="相关引用来源信息."
    )
