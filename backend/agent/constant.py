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
#
AWAITING_PLAN_CONFIRMATION = "awaiting_plan_confirmation"
SEARCH_REPLAN = "search_replan"

RESEARCH_AGENT_NODE = "reasearch_agent_node"
WRITER_AGENT_NODE = "writer_agent_node"
# 子图名称
SUB_RESEARCH_AGENT = "sub_research_agent"
SUB_WRITER_AGENT = "sub_writer_agent"

## MCP错误码
MCP_ERROR_RATE_LIMIT = "429"    # 当前阿里百炼的限流错误码为429

# 模型ID常量
MODEL_ID_FLASH = "qwen3.6-flash"
MODEL_ID_PLUS = "qwen3.7-plus-2026-05-26"
MODEL_ID_MAX = "qwen3.7-max-2026-05-20"
MODEL_ID_JUDEG = "deepseek-v4-flash"