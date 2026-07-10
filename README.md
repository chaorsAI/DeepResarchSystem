# DeepResarchSystem
DeepResearchSystem(Handwritten-Code) 场景下基于 LangGraph 的 Agentic RAG 系 统，实现了⼀个从"研究主题"到"结构化报告"的全⾃动生成验收效果。
全程手敲代码，主要是为了验证和巩固前段时间对AI的学习成果，包括但不限于Prompt、RAG、LangChain、LangGraph等核心技术。


## 项目目录结构
```bash
deep research/
├── backend/                          # 后端服务目录
│   ├── src/
│   │   ├── agent/                    # 核心代理模块
│   │   │   ├── __init__.py
│   │   │   ├── app.py               # FastAPI 应用入口
│   │   │   ├── base_agent.py        # 基础代理类定义
│   │   │   ├── configuration.py     # 系统配置管理
│   │   │   ├── graph.py             # LangGraph 工作流定义
│   │   │   ├── state.py             # 状态数据结构定义
│   │   │   ├── tools_and_schemas.py # 工具和模式定义
│   │   │   ├── prompts.py           # 提示词模板
│   │   │   ├── post.py              # 后处理工具
│   │   │   ├── utils.py             # 工具函数
│   │   │   └── llm/                 # LLM 集成模块
│   │   │       ├── __init__.py
│   │   │       └── llm.py           # 大语言模型接口
│   │   └── main.py                  # 主程序入口
│   ├── langgraph.json               # LangGraph 配置文件
│   ├── pyproject.toml               # Python 项目配置
├── frontend/                        # 前端应用目录
│   ├── src/
│   │   ├── components/              # React 组件
│   │   │   ├── ActivityTimeline.tsx # 活动时间线组件
│   │   │   ├── ChatMessagesView.tsx # 聊天消息视图
│   │   │   ├── InputForm.tsx        # 输入表单组件
│   │   │   ├── WelcomeScreen.tsx    # 欢迎界面组件
│   │   │   └── ui/                  # UI 组件库
│   │   ├── App.tsx                  # 主应用组件
│   │   ├── main.tsx                 # 应用入口
│   │   └── global.css               # 全局样式
│   ├── package.json                 # Node.js 依赖配置
│   └── vite.config.ts               # Vite 构建配置
├── README.md                        # 项目说明文档，即本文档
└── run.sh                           # 启动脚本
```
