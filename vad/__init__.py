from vad.vad import (
    SileroVADModel,
    get_vad_model,
    StreamingVAD,
    StreamingVadOptions,
    BatchVadOptions,
    get_speech_timestamps,
    collect_chunks,
    run_vad,
)

# Backward-compatible alias
VadOptions = StreamingVadOptions

__all__ = [
    "SileroVADModel",
    "get_vad_model",
    "StreamingVAD",
    "StreamingVadOptions",
    "VadOptions",
    "BatchVadOptions",
    "get_speech_timestamps",
    "collect_chunks",
    "run_vad",
]
