# MiniCPM-o 4.5 PyTorch 简易演示系统 — 项目文档

## 项目简介

MiniCPM-o 4.5 PyTorch 简易演示系统是由 MiniCPM-o 4.5 模型训练团队官方提供的演示系统。它使用 **PyTorch + CUDA** 推理后端，结合简易的前后端设计，以透明、简洁、无性能损失的方式全面演示 MiniCPM-o 4.5 的音视频全模态全双工能力。

## 交互模式

系统支持三种交互模式，共享同一个模型实例，支持毫秒级热切换（< 0.1ms）：

| 模式 | 特点 | 输入输出模态 | 范式 |
|------|------|-------------|------|
| **Turn-based Chat** | 低延迟流式交互，按钮或 VAD 触发回复，基础能力强 | 音频 + 文本 + 视频输入，音频 + 文本输出 | 轮次对话 |
| **Omnimodal Full-Duplex** | 全模态全双工，视觉+语音输入、语音输出同时进行 | 视觉 + 语音输入，文本 + 语音输出 | 全双工 |
| **Audio Full-Duplex** | 语音全双工，语音输入和输出同时进行 | 语音输入，文本 + 语音输出 | 全双工 |

## 其他特性

- 可自定义系统提示词
- 可自定义参考音频（TTS 声音克隆）
- 代码简洁易读，便于二次开发
- 可作为 API 后端供第三方应用调用
- 支持会话录制与回放

## 文档导航

| 文档 | 说明 |
|------|------|
| [系统架构](architecture.md) | 整体架构拓扑、请求流转、模块依赖关系 |
| [Gateway 模块](gateway.md) | 请求路由、WebSocket 代理、队列调度、LRU 缓存 |
| [Worker 模块](worker.md) | 模型推理服务、状态机、会话录制与清理 |
| [Core 模块](core.md) | Schema 定义、处理器架构、能力声明系统 |
| [模型模块](model.md) | MiniCPMO45 多模态模型架构详解 |
| [torch.compile](compile.md) | 编译加速实现、目标子模块、预热机制 |
| **前端模块** | |
| [前端概述](frontend/index.md) | 模块结构、技术栈、导航索引 |
| [页面与路由](frontend/pages.md) | 各页面详解、Turn-based Chat 状态管理 |
| [音频处理](frontend/audio.md) | AudioWorklet 采集、播放器、LUFS 测量 |
| [双工会话](frontend/duplex-session.md) | DuplexSession 类、状态机、录制系统 |
| [UI 组件](frontend/components.md) | 共享组件、内容编辑器、指标面板 |
| [API 参考](api.md) | HTTP REST API、WebSocket 协议消息格式 |
| [配置与部署](deployment.md) | 系统要求、配置说明、启动部署、测试 |

## 项目代码结构总览

```
minicpm-o-4_5-pytorch-simple-demo/
├── config.json / config.example.json  # 服务配置
├── config.py                          # 配置加载（Pydantic）
├── requirements.txt                   # Python 依赖
├── install.sh                         # 一键安装脚本
├── start_all.sh                       # 一键启动脚本
│
├── gateway.py                         # Gateway 服务入口
├── gateway_modules/                   # Gateway 业务模块
│   ├── worker_pool.py                 #   Worker 池调度器
│   ├── app_registry.py                #   应用注册表
│   ├── models.py                      #   数据模型
│   └── ref_audio_registry.py          #   参考音频注册表
│
├── worker.py                          # Worker 推理服务入口
├── session_recorder.py                # 会话录制器
├── session_cleanup.py                 # 会话清理器
│
├── core/                              # 核心封装层
│   ├── schemas/                       #   Pydantic Schema
│   │   ├── common.py                  #     公共类型（Role, Message, Config）
│   │   ├── chat.py                    #     Chat 请求/响应
│   │   ├── streaming.py               #     Streaming 请求/响应
│   │   └── duplex.py                  #     Duplex 请求/响应
│   ├── processors/                    #   推理处理器
│   │   ├── base.py                    #     基类与 Mixin
│   │   └── unified.py                 #     统一处理器（三 View）
│   ├── capabilities.py                #   能力声明
│   └── factory.py                     #   工厂模式
│
├── MiniCPMO45/                        # 模型核心推理代码
│   ├── configuration_minicpmo.py      #   模型配置
│   ├── modeling_minicpmo.py           #   主模型实现
│   ├── modeling_minicpmo_unified.py   #   统一模型（热切换）
│   ├── modeling_navit_siglip.py       #   SigLIP 视觉编码器
│   ├── processing_minicpmo.py         #   多模态处理器
│   ├── tokenization_minicpmo_fast.py  #   快速分词器
│   └── utils.py                       #   工具函数
│
├── static/                            # 前端页面
│   ├── index.html                     #   首页
│   ├── turnbased.html                 #   轮次对话页
│   ├── admin.html                     #   管理面板
│   ├── session-viewer.html            #   会话回放
│   ├── omni/                          #   Omni 全双工页面
│   ├── audio-duplex/                  #   音频全双工页面
│   ├── duplex/                        #   双工共享库
│   │   ├── lib/                       #     核心库（音频/视频/会话）
│   │   └── ui/                        #     UI 组件
│   ├── shared/                        #   跨页面共享组件
│   └── lib/                           #   通用工具库
│
├── resources/                         # 资源文件（参考音频等）
├── tests/                             # 测试套件
├── docs/                              # 项目文档
└── tmp/                               # 运行时日志和 PID 文件
```

## 快速开始

1. 确保有 NVIDIA GPU（显存 > 28GB）和 Linux 系统
2. 安装 FFmpeg
3. 运行 `bash install.sh` 安装依赖
4. 复制 `cp config.example.json config.json`
5. 运行 `CUDA_VISIBLE_DEVICES=0 bash start_all.sh` 启动服务
6. 访问 `https://localhost:8006`

详细部署说明请参考 [配置与部署](deployment.md)。
