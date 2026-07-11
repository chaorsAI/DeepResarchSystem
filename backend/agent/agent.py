# agent.py    Agent 核心类
from openai.lib.azure import API_KEY_SENTINEL

from configuration import get_default_model_id
from llm import OpenAICompatibleLLM
from constant import *

import os
import copy
import traceback
import time
import threading
import json
from dotenv import load_dotenv

from loguru import logger

from dashscope import Application

load_dotenv()

class Agent :
    step_prompt = """{prompt}"""
    def __init__(self, model_id=get_default_model_id()) :
        self.llm = OpenAICompatibleLLM(model_id=model_id)

    def __call(self, prompt) :
        response = self.llm.generate_response(prompt)
        return response

    def set_step_prompt(self, prompt):
        self.step_prompt = prompt

    def step(self, **kwargs):
        step_prompt = self.prompt_format(self.step_prompt, **kwargs)
        response = ""
        for _ in range(3) :
            try:
                response = self(step_prompt)
                response = self.post_process(response)
                break
            except Exception as e :
                logger.error(f"大模型调用错误：{e}\n{traceback.format_exc()}")
                continue

            return response

    def post_process(self, response):
        return response

    def prompt_format(self, prompt, **kwargs) :
        """高效进行 Prompt 模板渲染的基础方式"""
        # 深拷贝原prompt，避免污染外部该变量
        prompt_ = copy.deepcopy(prompt)
        # kwargs：关键参数包，eg：input="Hello", style="Formal"；遍历参数包读值
        for k in  kwargs.keys() :
            # 构造占位符，并将占位符(key)对应的值修改为参数包传过来的值，完成prompt模板的渲染
            rep = "{" + k + "}"
            prompt_ = prompt_.replace(rep, str(kwargs[k]))

        return prompt_

class JsonAgent(Agent):
    """将LLM返回数据转换为JSON根式"""
    def __init__(self, model_id=get_default_model_id(), keys=None):
        super().__init__(model_id)
        self.keys = keys

    def post_process(self, response):
        """self.keys参数转换为Pydantic模型类"""
        result = json.loads(Post.extract_pattern(response, pattern="json"))
        if not self.keys:
            return result
        # **result：字典解包，将result “炸开”成 key=value 的形式
        # self.keys：Pydantic 模型类工厂函数，相当于使用result进行构造；Pydantic自动做字段验证和类型转换
        return self.keys(**result)

class MCPAgent :
    """和 MCP 通讯"""
    def step(self, **kwargs):
        try:
            step_prompt = self.step_prompt.format(**kwargs)
        except Exception as e:
            step_prompt = self.step_prompt

        for _ in range(3):
            try:
                response = Application.call(
                    api_key = os.getenv("LLM_API_KEY"),
                    app_id = os.getenv("MCP_APP_ID"),
                    prompt = step_prompt,
                    biz_params = kwargs
                )
                response = self.post_process(response)
                if response is None:
                    raise Exception("MCP返回结果不正确")
                return response
            except Exception as e:
                logger.error(f"MCP调用错误：{e}\n{traceback.format_exc()}")
                continue
        return None

    def post_process(self, response):
        if response.status_code == 200:
            response = json.loads(response.output.text)
            return response
        else:
            logger.error(f"MCP调用失败：{response}")
            return None

class WebSearchAgent(MCPAgent):
    """WebSearch MCP"""
    def step(self, prompt, **kwargs):
        try:
            step_prompt = self.step_prompt.format(prompt=prompt)
        except Exception as e:
            step_prompt = self.step_prompt

        api_key = os.getenv("LLM_API_KEY")
        app_id = os.getenv("MCP_APP_ID")

        return None

    def post_process(self, response):
        if response is None:
            raise Exception("Web搜索结果不正确")
        if response.status_code == 200:
            try:
                return None


