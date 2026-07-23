# hook_context.py    通用 monkey-patch 框架

from typing import Callable, Any, Optional, List, Tuple
import threading
import functools

class HookContext:
    """
    通用 monkey-patch 上下文管理器
    """
    def __init__(self,
                 target: Tuple[Any, str],
                 post_hook: Optional[Callable]):
        """
        Args:
            target: (宿主对象, 方法名)
            post_hook: 调用后执行，签名: (result, args, kwargs) -> result
        """
        self.target = target
        self.post_hook = post_hook

        # 线程安全的原始方法存储
        self._local = threading.local()

    def _create_wrapper(self, original_func):
        """
        创建包装函数（Wrapper）。
        Hook 的灵魂：它包裹了原始函数，并在前后插入钩子。
        """
        @functools.wraps(original_func)
        def wrapper(*args, **kwargs):
            # 执行原方法
            try:
                # 1. 调用原始函数（Original Call）
                # 这一步就像接力棒，交还给原来的逻辑
                result = original_func(*args, **kwargs)
            except Exception as e:
                if self.error_hook:
                    self.error_hook(e, args, kwargs)
                raise

            # 2. 后置处理（Hook Logic）
            # 拿到结果后，执行我们定义的钩子（比如记录日志、采集数据）
            if self.post_hook:
                result = self.post_hook(result, args, kwargs)

            return result

        return wrapper

    def __enter__(self):
        """
        进入 `with` 代码块时自动调用。
        """
        obj, name = self.target
        # [关键点 1: 备份原件]
        # 读取当前 obj 下的 name 属性（即原始方法），存入线程本地存储
        self._local.original = getattr(obj, name)
        # [关键点 2: 注入替身] <--- 这就是你问的“替换的方法”
        # 创建一个包裹了原方法的新函数，并将其赋值给 obj.name
        wrapper = self._create_wrapper(self._local.original)
        setattr(obj, name, wrapper)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        退出 `with` 代码块时自动调用。
        """
        obj, name = self.target
        # [关键点 3: 还原替身] <--- 这就是 unpatch！
        # 将之前备份的原始方法重新赋值回去
        setattr(obj, name, self._local.original)

        # 清理线程本地存储，防止内存泄漏
        del self._local.original

        # 返回 False 表示不吞掉异常，让异常继续向外抛出
        return False