# agent.py    Agent 核心类
import json
from jsonpath_ng import parse as jsonpath_parse
from jsonpath_ng.exceptions import JSONPathError
from jsonpath_ng.ext import parser as jsonpath_parser

from openai.lib.azure import API_KEY_SENTINEL

from backend.agent.configuration import get_default_model_id
from backend.agent.llm import OpenAICompatibleLLM
from backend.agent.constant import *
from backend.agent.jsonUtils import *

import os
import copy
import traceback
import time
import threading
from dotenv import load_dotenv

from loguru import logger

from dashscope import Application

load_dotenv()

class RateLimiter :
    """
    基于令牌桶算法的速率限制器
    用于控制API请求频率，避免触发429错误
    """
    def __init__(self, max_qps : float = 15.0):
        """初始化速率限制器"""
        self.max_qps = max_qps
        # 前后请求最小间隔
        self.min_interval = 1.0 / max_qps
        self.last_request_time = 0
        self.lock = threading.Lock()
        logger.info(f"速率限制器初始化：QPS= {max_qps}, 最小请求间隔 = {self.min_interval:.3f}秒")

    def acquire(self):
        """
        获取请求许可，如果频率超限则等待
        :return: float: 实际等待的时间（秒）
        """
        with self.lock :
            current_time = time.time()
            time_since_last = current_time - self.last_request_time

            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                logger.debug(f"速率限制，需要等待{wait_time:.3f}秒")
                self.last_request_time = time.time()

                return wait_time
            else:
                self.last_request_time = current_time
                return 0





    # get_web_search_rate_limiter()


class Agent :
    step_prompt = """{prompt}"""
    def __init__(self, model_id="qwen2.5-72b-instruct") :
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
        result = json.loads(jsonUtils.extract_pattern(response, pattern="json"))
        if not self.keys:
            return result
        # **result：字典解包，将result “炸开”成 key=value 的形式
        # self.keys：Pydantic 模型类工厂函数，相当于使用result进行构造；Pydantic自动做字段验证和类型转换
        return self.keys(**result)

class MCPAgent(Agent) :
    """和 MCP 通讯"""
    def step(self, **kwargs):
        try:
            step_prompt = self.step_prompt.prompt_format(**kwargs)
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

_web_search_rate_limiter = None
def get_web_search_rate_limiter(max_qps: float = None) -> RateLimiter:
    """
    全局速率限制器实例（单例模式）
    """
    global _web_search_rate_limiter

    if _web_search_rate_limiter is None:
        if max_qps is None:
            # 从环境变量读取，默认为12 QPS（留有余量）
            max_qps = float(os.getenv("WEB_SEARCH_MAX_QPS", "12"))
        _web_search_rate_limiter = RateLimiter(max_qps=max_qps)

    return _web_search_rate_limiter

class WebSearchAgent(MCPAgent):
    """WebSearch MCP"""
    def step(self, prompt, **kwargs):
        try:
            step_prompt = self.step_prompt.format(prompt=prompt)
        except Exception as e:
            step_prompt = self.step_prompt

        api_key = os.getenv("LLM_API_KEY")
        app_id = os.getenv("MCP_APP_ID")

        # 获取速率限制器; 防止触发限流
        rate_limiter = get_web_search_rate_limiter()
        for attempt in range(3):
            try:
                # 在发送请求前进行速率限制检查
                wait_time = 3 # rate_limiter.acquire()
                if wait_time > 0 :
                    logger.debug(f"速率限制等待：{wait_time:.3f}秒")

                response = Application.call(
                    api_key=api_key,
                    app_id=app_id,
                    prompt=step_prompt,
                    biz_params=kwargs
                )
                response = self.extract_pages_from_mcp_response(response, None)
                return response

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Web搜错错误（尝试{attempt + 1} / 3）：{e}\n{traceback.format_exc()}")

                # 我们本身在请求时已经有了 rate_limiter 为什么还需要判断 MCP_ERROR_RATE_LIMIT 错误
                # 自定义 rate_limiter 只是能减少触发限流的可能；但是平台(阿里等)内部的限流机制我们并不清楚，所以采取双保险机制
                if "429" in error_msg :
                    # 递增等待时间：5秒、10秒、15秒
                    wait_time = 5 * (attempt + 1)
                    logger.warning(f"检测到 429 错误，等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                # 非429错误，短暂等待后重试
                elif attempt < 2:
                    time.sleep(2)
                continue
        return None

    def extract_pages_from_mcp_response(self, response, config=None):
        """从MCP WebSearch响应中提取pages数据（重构版）
        Args:
        response: MCP响应对象
        config: 解析路径配置，默认使用预定义的JSONPath列表
        """
        if response is None:
            raise Exception("Web搜索结果不正确")
        if not response.status_code == 200:
            raise Exception(f"Web搜索异常: {response.status_code}")
        # 定义兜底的解析scheme 优先级从高到低
        # 多路探测防御性解析，解决MCP等工具调用返回结构的”不确定性“问题
        DEFAULT_PARSE_CONFIG = {
            "paths": [
                "$.pages",                  # 直接在第一层
                "$.data.pages",             # data路径
                "$.result.content[0].text",  # result.content路径
                "$.choices[0].message.content",
                # 未来新增路径只需加在这里，无需改逻辑
            ]
        }

        parse_config = config if config else DEFAULT_PARSE_CONFIG

        try:
            # 假设 response.output.text 是原始的JSON字符串
            first_level = json.loads(response.output.text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}, 原始响应: {response.output.text[:500]}")
            raise Exception(f"Web搜索结果JSON解析失败: {str(e)}")

        pages = None
        matched_path = None

        # 循环遍历JSONPath进行匹配
        for path_str in parse_config["paths"]:
            try:
                # 编译表达式（生产环境建议缓存此对象）
                jsonpath_expr = jsonpath_parser.parse(path_str)
                matches = [match.value for match in jsonpath_expr.find(first_level)]
                if matches:
                    # 取第一个匹配项
                    candidate = matches[0]

                    # 特殊处理：如果路径指向的是字符串（如 content[0].text），则再次解析
                    if isinstance(candidate, str):
                        try:
                            candidate = json.loads(candidate)
                            # 二次解析后，如果里面还有pages，需要再次查找
                            # 简单处理：如果二次解析后是dict且包含pages，则替换
                            if isinstance(candidate, dict) and "pages" in candidate:
                                pages = candidate["pages"]
                                matched_path = path_str + " -> inner.pages"
                                break
                            # 如果二次解析后直接是list，也可能是我们要的数据
                            elif isinstance(candidate, list):
                                pages = candidate
                                matched_path = path_str + " -> inner.list"
                                break
                        except json.JSONDecodeError:
                            # 如果不是JSON字符串，忽略该路径
                            continue

                    # 如果直接找到了列表
                    elif isinstance(candidate, list):
                        pages = candidate
                        matched_path = path_str
                        break

            except JSONPathError as e:
                logger.warning(f"JSONPath语法错误或匹配失败: {path_str}, Error: {e}")
                continue
            except (KeyError, IndexError, TypeError) as e:
                # 路径存在但索引越界等情况，继续尝试下一条路径
                continue

        # 后置校验与错误处理
        if pages is None:
            logger.error(
                f"无法从响应中提取pages数据。尝试的路径: {parse_config['paths']}。"
                f"响应结构: {json.dumps(first_level, ensure_ascii=False)[:500]}"
            )
            raise Exception("无法从Web搜索结果中提取页面数据")

        if not isinstance(pages, list):
            logger.error(f"pages不是列表类型: {type(pages)}, 匹配路径: {matched_path}")
            raise Exception("Web搜索结果格式错误")

        logger.info(f"成功从路径 '{matched_path}' 提取到 {len(pages)} 条pages数据")

        # 数据清洗 目标-“垃圾进，精品出”，防止-“Garbage in，Garbage out”
        processed_pages = []
        for page in pages:
            if isinstance(page, dict):
                processed_pages.append({
                    "snippet": page.get("snippet", ""),
                    "title": page.get("title", ""),
                    "url": page.get("url", "")
                })

        return processed_pages


