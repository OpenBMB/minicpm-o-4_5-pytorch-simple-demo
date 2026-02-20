
# Attention Backend（attn_implementation）

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
