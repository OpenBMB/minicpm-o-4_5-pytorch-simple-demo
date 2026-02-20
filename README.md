# MiniCPMO45 Web Service

基于 MiniCPM-o 4.5 的多模态推理服务。支持文本/图像/音频输入，提供 Chat、Streaming、Duplex 三种对话模式。

## 架构

```
Frontend (HTML/JS)
    │ HTTPS / WSS
Gateway (:10024, HTTPS)          ← 路由、排队、WS 代理
    │ HTTP / WS (internal)
Worker Pool (:22400+)            ← 每 Worker 独占一张 GPU
    ├── Worker 0 (GPU 0)
    ├── Worker 1 (GPU 1)
    └── ...
```

## 快速开始

**系统要求**：Linux + NVIDIA GPU (≥80GB 显存) + CUDA 12.x + Python 3.10

从零到跑起来，4 步：

```bash
# 1. Clone（含 submodule）
git clone --recurse-submodules <repo_url>
cd minicpmo45_service

# 2. 一键安装环境（自动安装 PyTorch + 依赖 + Flash Attention）
bash install.sh
#    Flash Attention 编译失败会自动跳过，不影响后续使用（降级 SDPA）
#    可选参数：PYTHON=python3.11 bash install.sh    # 指定 Python 版本
#              SKIP_FLASH_ATTN=1 bash install.sh     # 跳过 Flash Attention

# 3. 配置模型路径（唯一必填项）
cp config.example.json config.json
# 编辑 config.json，填入 model.model_path，其余用默认值即可
# 最小配置：{"model": {"model_path": "/path/to/MiniCPM-o-4_5"}}

# 4. 启动（自动检测 GPU，默认 HTTPS）
bash start_all.sh
```

服务启动后访问 https://localhost:10024 即可。自签名证书会触发浏览器警告，点"高级"→"继续访问"。

> **已有仓库但 submodule 为空？** 执行 `git submodule update --init`

默认配置下，所有模式（Chat / Streaming / Duplex）的语音合成均使用 **Token2Wav** vocoder，无需安装 CosyVoice2。模型目录只需包含 `assets/token2wav/` 即可。

### 高级配置：使用 CosyVoice2 作为 Chat vocoder（可选）

CosyVoice2 是另一个可用于 Chat（非流式）模式的 vocoder，音质风格略有不同。如需使用：

```bash
# 1. 安装 CosyVoice2 包
.venv/base/bin/pip install token2wav-cosyvoice==0.1.0

# 2. 确保模型目录下存在 CosyVoice2 模型文件
#    MiniCPM-o-4_5/assets/CosyVoice2-0.5B/

# 3. 修改 config.json
```
```json
{
    "model": {"model_path": "/path/to/MiniCPM-o-4_5"},
    "audio": {"chat_vocoder": "cosyvoice2"}
}
```

| 对比 | Token2Wav（默认） | CosyVoice2 |
|------|-------------------|------------|
| 适用模式 | Chat + Streaming + Duplex | 仅 Chat |
| 额外依赖 | 无（已含在 requirements.txt） | `token2wav-cosyvoice==0.1.0` |
| 额外模型 | 无 | `assets/CosyVoice2-0.5B/`（~0.5GB） |
| 启动耗时 | ~16s | ~112s（需额外加载 CosyVoice2 Qwen2 模型） |
| 额外显存 | 0 | ~0.5GB |

### 启动选项

```bash
CUDA_VISIBLE_DEVICES=0,1 bash start_all.sh          # 指定 GPU
bash start_all.sh --compile                          # torch.compile 加速（实验性）
bash start_all.sh --http                             # 降级 HTTP（不推荐，麦克风/摄像头 API 需要 HTTPS）
```

**手动启动（分步）**：
```bash
# Worker（每张 GPU 一个）
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/base/bin/python worker.py --worker-index 0 --gpu-id 0

# Gateway
PYTHONPATH=. .venv/base/bin/python gateway.py --port 10024 --workers localhost:22400
```

**停止**：
```bash
pkill -f "gateway.py|worker.py"
```

### 访问页面

| 页面 | URL |
|------|-----|
| Chat Demo | https://localhost:10024 |
| Omni Duplex | https://localhost:10024/omni |
| Audio Duplex | https://localhost:10024/audio_duplex |
| Admin Dashboard | https://localhost:10024/admin |
| API Docs | https://localhost:10024/docs |

### 配置

`config.json`（从 `config.example.json` 复制，已 gitignore）。完整字段见下方"配置说明"章节。

**最小配置**：
```json
{"model": {"model_path": "/path/to/MiniCPM-o-4_5"}}
```

**常用配置**（模型 + 微调权重 + 调整播放延迟）：
```json
{
    "model": {
        "model_path": "/path/to/MiniCPM-o-4_5",
        "pt_path": "/path/to/finetuned.pt"
    },
    "audio": {
        "playback_delay_ms": 300
    }
}
```

**模型目录结构要求**：
```
MiniCPM-o-4_5/
├── config.json
├── model*.safetensors
├── tokenizer.json
└── assets/
    ├── token2wav/          # Token2Wav vocoder（必需）
    └── CosyVoice2-0.5B/   # CosyVoice2 vocoder（可选，仅 chat_vocoder=cosyvoice2 时需要）
```

## 配置说明

### config.json — 统一配置文件

所有配置集中在 `config.json`（从 `config.example.json` 复制）。
`config.json` 已 gitignore，不会被提交。

**配置优先级**：CLI 参数 > config.json > Pydantic 默认值

| 分组 | 字段 | 默认值 | 说明 |
|------|------|--------|------|
| **model** | `model_path` | _(必填)_ | HuggingFace 格式模型目录 |
| model | `pt_path` | null | 额外 .pt 权重覆盖 |
| model | `attn_implementation` | `"auto"` | Attention 实现：`"auto"`/`"flash_attention_2"`/`"sdpa"`/`"eager"` |
| **audio** | `ref_audio_path` | `assets/ref_audio/ref_minicpm_signature.wav` | 默认 TTS 参考音频 |
| audio | `playback_delay_ms` | 200 | 前端音频播放延迟（ms），越大越平滑但延迟越高 |
| audio | `chat_vocoder` | `"token2wav"` | Chat 模式 vocoder：`"token2wav"`（默认）或 `"cosyvoice2"` |
| **service** | `gateway_port` | 10024 | Gateway 端口 |
| service | `worker_base_port` | 22400 | Worker 起始端口 |
| service | `max_queue_size` | 100 | 最大排队请求数 |
| service | `request_timeout` | 300.0 | 请求超时（秒） |
| service | `compile` | false | torch.compile 加速 |
| service | `data_dir` | "data" | 数据目录 |
| **duplex** | `pause_timeout` | 60.0 | Duplex 暂停超时（秒） |

**最小配置**（只需模型路径）：
```json
{"model": {"model_path": "/path/to/model"}}
```

### Attention Backend（attn_implementation）

控制模型推理使用的 Attention 实现。默认 `"auto"` 自动检测环境并选择最优方案。

| 值 | 行为 | 适用场景 |
|----|------|----------|
| `"auto"`（默认） | 检测到 flash-attn 包 → `flash_attention_2`；否则 → `sdpa` | 推荐，兼容所有环境 |
| `"flash_attention_2"` | 强制使用 Flash Attention 2，不可用时启动报错 | 确认已安装 flash-attn 且需要锁定 |
| `"sdpa"` | 强制使用 PyTorch 内置 SDPA，不依赖 flash-attn | 无法编译 flash-attn 的环境 |
| `"eager"` | 朴素 Attention 实现 | 仅 debug 用 |

**性能对比**（A100，典型推理场景）：`flash_attention_2` 比 `sdpa` 快约 5-15%，`sdpa` 比 `eager` 快数倍。

**启动日志**：Worker 启动时会明确输出实际使用的 backend，便于确认：

```
# auto 检测到 flash-attn，使用 flash_attention_2：
[Attention] auto → flash_attention_2 (flash-attn 2.6.3 可用，性能最优)

# auto 未检测到 flash-attn，降级到 sdpa：
[Attention] auto → sdpa (flash-attn 不可用，使用 PyTorch 内置 SDPA。如需 flash_attention_2，请安装: ...)

# 用户显式指定：
[Attention] 使用用户指定: sdpa
```

**子模块实际分配**：无论顶层配置什么，Audio（Whisper）子模块始终使用 SDPA（flash_attention_2 与 Whisper 不兼容）。其余子模块（Vision/LLM/TTS）遵循配置。

### CLI 参数覆盖

```bash
# Worker
python worker.py --model-path /alt/model --pt-path /alt/weights.pt --ref-audio-path /alt/ref.wav --compile

# Gateway
python gateway.py --port 10025 --workers localhost:22400,localhost:22401 --http
```

## 对话模式

| 模式 | 特点 | 接口 |
|------|------|------|
| **Chat** | 无状态，一问一答 | POST /api/chat |
| **Streaming** | 有状态，KV Cache 复用，流式输出 | WS /ws/streaming/{session_id} |
| **Duplex** | 全双工实时对话，支持打断 | WS /ws/duplex/{session_id} |

三种模式共享同一个模型实例，毫秒级热切换（< 0.1ms）。

## 项目结构

```
minicpmo45_service/
├── config.json               # 服务配置（从 config.example.json 复制，gitignored）
├── config.example.json       # 配置示例（完整字段 + 默认值）
├── config.py                 # 配置加载逻辑（Pydantic 定义 + JSON 加载）
├── requirements.txt          # Python 依赖
├── start_all.sh              # 一键启动脚本
│
├── gateway.py                # Gateway（路由、排队、WS 代理）
├── worker.py                 # Worker（推理服务）
├── gateway_modules/          # Gateway 业务模块
│
├── core/                     # 核心封装
│   ├── schemas/              # Pydantic Schema（请求/响应）
│   └── processors/           # 推理处理器（UnifiedProcessor）
│
├── MiniCPMO45/               # 模型代码（submodule）
├── static/                   # 前端页面
├── resources/                # 资源文件（参考音频等）
├── tests/                    # 测试
└── tmp/                      # 运行时日志和 PID 文件
```

## 资源消耗

| 资源 | Token2Wav（默认） | CosyVoice2 模式 |
|------|-------------------|-----------------|
| 显存（每 Worker） | ~21.5 GB | ~22 GB |
| 模型加载时间 | ~16s | ~112s |
| 模式切换延迟 | < 0.1ms | < 0.1ms |

> compile 模式首次推理额外 ~60s 编译耗时。

## 测试

```bash
cd minicpmo45_service

# Schema 单元测试（无需 GPU）
PYTHONPATH=. .venv/base/bin/python -m pytest tests/test_schemas.py -v

# Processor 测试（需要 GPU）
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/base/bin/python -m pytest tests/test_chat.py tests/test_streaming.py tests/test_duplex.py -v -s

# API 集成测试（需要先启动服务）
PYTHONPATH=. .venv/base/bin/python -m pytest tests/test_api.py -v -s
```
