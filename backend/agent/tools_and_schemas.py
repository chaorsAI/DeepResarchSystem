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

# ── Debate-loop Critic schemas ─────────────────────────────────────────

class Issue(BaseModel):
    """
    评审员对改稿的修改建议
    """
    severity: str = Field(
        description="严重程度: 'critical' (事实错误/逻辑断裂), 'major' (重要遗漏/论证不足), 'minor' (措辞/格式)"
    )
    location: str = Field(
        description="问题在报告中的位置描述，如'第2.1节'或'结论部分'"
    )
    problem: str = Field(
        description="具体问题描述"
    )
    suggestion: str = Field(
        description="具体的修改建议"
    )

class ReviewModel(BaseModel):
    """
    对草稿的评审结果
    """
    overall_rating: float = Field(
        ge=0, le=10,
        description="综合评分 0-10。≥8=可发布, 6-7=需小修, 4-5=需大修, <4=需重写"
    )
    issues: list[Issue] = Field(
        default_factory=list,
        description="发现的具体问题列表"
    )
    ready_for_polish: bool = Field(
        description="是否可以进入终审润色阶段"
    )
    summary: str = Field(
        description="一句话总结整体评价"
    )

# class ReviewModel(BaseModel):
#     critic_feedback: str = Field(
#         description="子图评审节点草稿内容的反馈.包含审稿评分、综合评价、具体问题"
#     )
#     critic_score:>
#
# "critic_feedback": feedback,
#         "critic_score": result.overall_rating,
#         "ready_for_polish": result.ready_for_polish,