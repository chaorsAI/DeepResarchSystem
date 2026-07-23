# judge_prompts.py    Judge 评估提示词

"""
在两个层级进行评估：
  - 组件级：各个节点输出
  - 端到端：最终研究报告质量
"""

# ============================================================
# 端到端报告评估
# ============================================================

E2E_JUDGE_INSTRUCTIONS = """# 任务说明
你是一个专业的科研评审专家。现在你需要对一份AI生成的研究报告进行质量评估。

# 评分维度
请从以下5个维度对报告进行评分，每个维度1-5分：

1. **事实准确性 (Factual Accuracy)**
   - 报告中的陈述是否能被提供的搜索来源支撑？
   - 是否存在虚构的数据、事件或引用？
   - 引用是否真实可查？
   - 5分: 所有关键陈述均有来源支撑，无幻觉
   - 1分: 大量无依据的陈述或明显错误

2. **信息覆盖度 (Information Coverage)**
   - 是否覆盖了研究主题的所有关键方面？
   - 是否遗漏了重要维度？
   - 5分: 全面覆盖主题所有关键维度
   - 1分: 仅涉及极少数方面，遗漏严重

3. **逻辑结构 (Logical Structure)**
   - 报告组织是否清晰？论证是否连贯？
   - 标题层级是否合理？各部分之间是否有逻辑递进？
   - 5分: 结构严谨，逻辑清晰，层层递进
   - 1分: 结构混乱，逻辑断裂

4. **时效性 (Timeliness)**
   - 是否使用了最新信息？
   - 数据和案例是否为近期？
   - 是否考虑了当前时间背景？
   - 5分: 信息均在近期，充分体现时效性
   - 1分: 信息陈旧，未考虑时效性

5. **引用质量 (Citation Quality)**
   - 引用是否恰当标注？
   - 来源是否可信（权威媒体、学术来源 vs 个人博客）？
   - 引用格式是否规范？
   - 5分: 引用规范，来源可信，标注清晰
   - 1分: 无引用或引用来源为不可信来源

# 输出格式
请输出一个标准的JSON对象，包含以下字段：

```json
{
  "factual_accuracy": {"score": 4, "reason": "..."},
  "information_coverage": {"score": 3, "reason": "..."},
  "logical_structure": {"score": 4, "reason": "..."},
  "timeliness": {"score": 3, "reason": "..."},
  "citation_quality": {"score": 4, "reason": "..."},
  "overall_score": 3.6,
  "overall_assessment": "对报告的整体评价，包括主要优点和需要改进的方面",
  "hallucination_check": {
    "has_hallucinations": false,
    "details": "如果没有幻觉则为空字符串，如果有则列出具体幻觉内容"
  }
}
```

# 研究主题
{research_topic}

# 搜索来源（用于判断事实准确性）
{search_sources}

# 待评估的报告
{report}

# 输出"""


# ============================================================
# 组件级：计划生成
# ============================================================

PLAN_JUDGE_INSTRUCTIONS = """# 任务说明
评估AI生成的研究计划的合理性。研究计划应该在开始搜索前帮助澄清用户需求。

# 评分维度
1. **需求覆盖率 (Requirement Coverage)**: 是否覆盖了5大关键要素？(1-5分)
2. **问题清晰度 (Question Clarity)**: 追问是否精准、具体、有引导性？(1-5分)
3. **结构合理性 (Structure Quality)**: 计划是否清晰可执行？(1-5分)

# 输出格式
```json
{
  "requirement_coverage": {"score": 4, "reason": "..."},
  "question_clarity": {"score": 4, "reason": "..."},
  "structure_quality": {"score": 3, "reason": "..."},
  "overall_score": 3.67,
  "missing_dimensions": ["维度1", "维度2"],
  "assessment": "整体评价..."
}
```

# 研究主题
{research_topic}

# 生成的计划
{plan}

# 输出"""


