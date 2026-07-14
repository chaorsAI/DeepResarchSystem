# configuration.py    Agent 相关配置项，涉及大模型配置的

# 顶部必须明确定义导出接口
__all__ = ["Configuration"]


import os
import jsonUtils
from pydantic import BaseModel, Field
from typing import Any, Optional, List

from langchain_core.runnables import RunnableConfig


class ModelConfig(BaseModel) :
    """LLM模型配置"""
    id : str = Field(..., description="模型ID")
    name : str = Field(..., description="模型名称")
    icon :str = Field(default="Zap", description="图标类型(Zap/Cpu)")
    icon_color : str = Field(default="yellow-400", description="图标颜色")

def load_available_models_from_env() :
    """从环境变量加载可用模型"""
    model_json = os.getenv(AVAILABLE_MODELS)

    if not model_json :
        # 兜底模型
        return [
            ModelConfig(id="qwen3.6-flash", name="Qwen-Flash", icon="Zap", icon_color="yellow-400"),
            ModelConfig(id="qwen3.6-plus", name="Qwen-Plus", icon="Zap", icon_color="green-400"),
            ModelConfig(id="qwen3.7-max", name="Qwen-Max", icon="Cpu", icon_color="blue-400")
        ]

    try:
        models_data = json.loads(model_json)
        return [ModelConfig(**model) for model in models_data]
    except Exception as e :
        print(f"警告: 解析AVAILABLE_MODELS失败，使用默认模型列表。错误: {e}")
        return [
            ModelConfig(id="qwen3.6-flash", name="Qwen-Flash", icon="Zap", icon_color="yellow-400"),
            ModelConfig(id="qwen3.6-plus", name="Qwen-Plus", icon="Zap", icon_color="green-400"),
            ModelConfig(id="qwen3.7-max", name="Qwen-Max", icon="Cpu", icon_color="blue-400")
        ]

def get_default_model_id() :
    """获取默认模型ID（模型列表的最后一项）"""
    models = load_available_models_from_env()
    if models :
        return models[-1].id
    return "qwen3.7-max"  # 兜底默认值

class Configuration(BaseModel) :
    """Agent 配置"""
    # 可用模型
    available_models : List[ModelConfig] = Field(
        default_factory=load_available_models_from_env,
        # default=load_available_models_from_env()
        description="可用的 LLM 模型列表"
    )

    query_generator_model : str = Field(
        default_factory=get_default_model_id,
        description="Agent查询生成的LLM的名称"
    )

    reflection_model: str = Field(
        default_factory=get_default_model_id,
        description="Agent 反思的LLM名称"
    )

    answer_model: str = Field(
        default_factory=get_default_model_id,
        description="Agent 生成答案的LLM名称."
    )

    number_of_initial_queries: int = Field(
        default=2,
        description="要生成的初始搜索查询数量."
    )

    max_research_loops: int = Field(
        default=2,
        description="要执行的最大research循环次数.",
    )

    @classmethod
    # RunnableConfig到Pydantic配置模型的适配层
    def runnable_config(cls, config : Optional[RunnableConfig] = None) -> "Configuration" :
        """从RunnableConfig创建配置实例"""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        raw_values : dict[str, Any] = {}
        for name in cls.model_fields.keys() :
            # 跳过 available_models，它应该从环境变量直接加载
            if name == "available_models":
                continue
            env_value = os.environ.get(name.upper())
            config_value = configurable.get(name)
            raw_values[name] = env_value if env_value is not None else config_value

        values = {k: v for k, v in raw_values.items() if v is not None}

        return cls(**values)


