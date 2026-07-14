# constant.py    相关常量定义

from dotenv import load_dotenv

load_dotenv()

## Graph 相关
# 图名称
DEEP_RESEARCH_AGENT = "deep_research_agent"
# 节点定义
GENERATE_PLAN_NODE = "generate_plan"
# 查询生成
GENERATE_SEARCH_NODE = "generate_search"
# 网搜
WEB_SEARCH_NODE = "web_search"
# 反思/评审
CRITIQUE_NODE = "critique"
# 最终结果
FINAL_ANSWER_NODE = "final_answer"


## MCP错误码
MCP_ERROR_RATE_LIMIT = "429"    # 当前阿里百炼的限流错误码为429