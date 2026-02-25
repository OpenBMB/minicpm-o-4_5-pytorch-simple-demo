
# Attention Backend (attn_implementation)

Controls the Attention implementation used for model inference. The default `"auto"` mode automatically detects the environment and selects the optimal option.

| Value | Behavior | Use Case |
|----|------|----------|
| `"auto"` (default) | Detects flash-attn package → `flash_attention_2`; otherwise → `sdpa` | Recommended; compatible with all environments |
| `"flash_attention_2"` | Forces Flash Attention 2; raises an error on startup if unavailable | When flash-attn is confirmed installed and you want to lock it in |
| `"sdpa"` | Forces PyTorch built-in SDPA; no flash-attn dependency | Environments where flash-attn cannot be compiled |
| `"eager"` | Naive Attention implementation | Debug only |

**Performance Comparison** (A100, typical inference scenario): `flash_attention_2` is ~5-15% faster than `sdpa`; `sdpa` is several times faster than `eager`.

**Startup Logs**: The Worker explicitly outputs the actual backend in use at startup for easy confirmation:

```
# auto detects flash-attn, uses flash_attention_2:
[Attention] auto → flash_attention_2 (flash-attn 2.6.3 available, best performance)

# auto does not detect flash-attn, falls back to sdpa:
[Attention] auto → sdpa (flash-attn not available, using PyTorch built-in SDPA. For flash_attention_2, install: ...)

# User explicitly specified:
[Attention] Using user-specified: sdpa
```

**Submodule Actual Assignment**: Regardless of the top-level configuration, the Audio (Whisper) submodule always uses SDPA (flash_attention_2 is incompatible with Whisper). The remaining submodules (Vision/LLM/TTS) follow the configuration.
