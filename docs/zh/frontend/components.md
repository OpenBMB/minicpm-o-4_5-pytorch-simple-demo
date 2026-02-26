# 前端 UI 组件

## 共享组件 (shared/)

### app-nav.js — 动态导航栏

从 `/api/apps` 获取已启用的应用列表，动态渲染导航链接。

- 自动高亮当前页面
- 访问未启用应用时重定向到首页
- 响应式设计（移动端折叠）

### preset-selector.js — 预设选择器

`PresetSelector` 组件管理系统提示词预设。

- 从 `/api/presets` 加载预设列表
- 下拉选择后自动填充系统提示词
- 使用 localStorage 记忆上次选择

### save-share.js — 保存与分享

`SaveShareUI` 管理会话保存和分享：

- 上传录制到 `/api/sessions/{sid}/upload-recording`
- 生成分享链接 `/s/{session_id}`
- 最近会话存储在 localStorage（最多 20 条）

---

## 内容编辑器

### ref-audio-player.js — 参考音频播放器

可复用的音频播放器组件，用于参考音频预览：

- 上传：文件选择或拖放
- 解码：自动解码并重采样到 16kHz 单声道
- 播放：播放/暂停、可拖动进度条、时长显示
- 内部使用 `AudioContext` 解码，输出 Float32 PCM

### system-content-editor.js — 系统内容编辑器

列表式编辑器，管理 system prompt 的多模态内容：

- **文本项**：可编辑文本
- **音频项**：参考音频，集成 `RefAudioPlayer`
- 支持添加/删除/拖拽重排序

### user-content-editor.js — 用户消息编辑器

多模态用户输入编辑器：

| 内容类型 | 输入方式 |
|---------|---------|
| 文本 | 文本框输入 |
| 音频 | 录音 / 文件上传 / 拖放 |
| 图片 | 文件上传 / 拖放 / 粘贴 |
| 视频 | 文件上传 / 拖放（仅 Chat 模式，自动启用 omni_mode） |

**快捷键**：`Space` 录音（点击切换/长按 push-to-talk），`ESC` 取消。

---

## 双工 UI 组件 (duplex/ui/)

### duplex-ui.js — 指标面板 + 设置持久化

**MetricsPanel** 实时指标面板：

| 指标 | 颜色阈值 |
|------|---------|
| Latency（延迟） | 绿 <200ms / 黄 <500ms / 红 |
| TTFS（首音延迟） | 绿 <300ms / 黄 <600ms / 红 |
| Drift（漂移） | 绿 <50ms / 黄 <200ms / 红 |
| Gaps（间隙） | 绿 =0 / 黄 <3 / 红 |
| KV Cache | 绿 <4096 / 黄 <6144 / 红 |

**SettingsPersistence** 设置持久化：声明式字段定义，localStorage 自动存储/恢复。

### ref-audio-init.js — 参考音频初始化

Duplex 页面加载时初始化参考音频：从后端加载默认和自定义参考音频列表，构建下拉选择器。

### tts-ref-controller.js — TTS 参考音频控制器

管理 TTS 参考音频选择、上传、删除。支持 LLM ref 和 TTS ref 使用不同音频。

---

## 通用工具 (lib/)

### countdown-timer.js — 倒计时状态机

UI 无关的倒计时状态机，通过回调报告状态。用于排队等待时间显示。

### chat-eta-estimator.js — ETA 估算器

基于历史响应时间的 EMA 估算，区分 Chat/Streaming 模式独立估算。
