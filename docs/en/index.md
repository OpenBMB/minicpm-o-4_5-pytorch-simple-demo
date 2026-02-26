# MiniCPM-o 4.5 PyTorch Simple Demo System — Project Documentation

## Project Overview

The MiniCPM-o 4.5 PyTorch Simple Demo System is an official demo system provided by the MiniCPM-o 4.5 model training team. It uses a **PyTorch + CUDA** inference backend, combined with a streamlined frontend and backend design, to comprehensively demonstrate MiniCPM-o 4.5's audio-visual omnimodal full-duplex capabilities in a transparent, concise, and lossless manner.

## Interaction Modes

The system supports three interaction modes, sharing a single model instance with millisecond-level hot-switching (< 0.1ms):

| Mode | Features | Input/Output Modalities | Paradigm |
|------|----------|------------------------|----------|
| **Turn-based Chat** | Low-latency streaming interaction, reply triggered by button or VAD, strong baseline capabilities | Audio + text + video input, audio + text output | Turn-based dialogue |
| **Omnimodal Full-Duplex** | Omnimodal full-duplex with simultaneous vision + voice input and voice output | Vision + voice input, text + voice output | Full-duplex |
| **Audio Full-Duplex** | Voice full-duplex with simultaneous voice input and output | Voice input, text + voice output | Full-duplex |

## Additional Features

- Customizable system prompts
- Customizable reference audio (TTS voice cloning)
- Clean, readable code for easy secondary development
- Can serve as an API backend for third-party applications
- Session recording and playback support

## Documentation Navigation

| Document | Description |
|----------|-------------|
| [System Architecture](architecture.md) | Overall architecture topology, request flow, module dependencies |
| [Gateway Module](gateway.md) | Request routing, WebSocket proxy, queue scheduling, LRU cache |
| [Worker Module](worker.md) | Model inference service, state machine, session recording and cleanup |
| [Core Module](core.md) | Schema definitions, processor architecture, capability declaration system |
| [Model Module](model.md) | Detailed MiniCPMO45 multimodal model architecture |
| [torch.compile](compile.md) | Compilation acceleration, target submodules, warm-up mechanism |
| **Frontend Modules** | |
| [Frontend Overview](frontend/index.md) | Module structure, tech stack, navigation index |
| [Pages & Routing](frontend/pages.md) | Page details, Turn-based Chat state management |
| [Audio Processing](frontend/audio.md) | AudioWorklet capture, player, LUFS measurement |
| [Duplex Session](frontend/duplex-session.md) | DuplexSession class, state machine, recording system |
| [UI Components](frontend/components.md) | Shared components, content editor, metrics panel |
| [API Reference](api.md) | HTTP REST API, WebSocket protocol message format |
| [Configuration & Deployment](deployment.md) | System requirements, configuration guide, startup deployment, testing |

## Project Code Structure Overview

```
minicpm-o-4_5-pytorch-simple-demo/
├── config.json / config.example.json  # Service configuration
├── config.py                          # Configuration loading (Pydantic)
├── requirements.txt                   # Python dependencies
├── install.sh                         # One-click install script
├── start_all.sh                       # One-click start script
│
├── gateway.py                         # Gateway service entry point
├── gateway_modules/                   # Gateway business modules
│   ├── worker_pool.py                 #   Worker pool scheduler
│   ├── app_registry.py                #   Application registry
│   ├── models.py                      #   Data models
│   └── ref_audio_registry.py          #   Reference audio registry
│
├── worker.py                          # Worker inference service entry point
├── session_recorder.py                # Session recorder
├── session_cleanup.py                 # Session cleanup
│
├── core/                              # Core abstraction layer
│   ├── schemas/                       #   Pydantic Schemas
│   │   ├── common.py                  #     Common types (Role, Message, Config)
│   │   ├── chat.py                    #     Chat request/response
│   │   ├── streaming.py               #     Streaming request/response
│   │   └── duplex.py                  #     Duplex request/response
│   ├── processors/                    #   Inference processors
│   │   ├── base.py                    #     Base class & Mixin
│   │   └── unified.py                 #     Unified processor (three Views)
│   ├── capabilities.py                #   Capability declarations
│   └── factory.py                     #   Factory pattern
│
├── MiniCPMO45/                        # Model core inference code
│   ├── configuration_minicpmo.py      #   Model configuration
│   ├── modeling_minicpmo.py           #   Main model implementation
│   ├── modeling_minicpmo_unified.py   #   Unified model (hot-switching)
│   ├── modeling_navit_siglip.py       #   SigLIP vision encoder
│   ├── processing_minicpmo.py         #   Multimodal processor
│   ├── tokenization_minicpmo_fast.py  #   Fast tokenizer
│   └── utils.py                       #   Utility functions
│
├── static/                            # Frontend pages
│   ├── index.html                     #   Home page
│   ├── turnbased.html                 #   Turn-based chat page
│   ├── admin.html                     #   Admin panel
│   ├── session-viewer.html            #   Session playback
│   ├── omni/                          #   Omni full-duplex page
│   ├── audio-duplex/                  #   Audio full-duplex page
│   ├── duplex/                        #   Duplex shared library
│   │   ├── lib/                       #     Core library (audio/video/session)
│   │   └── ui/                        #     UI components
│   ├── shared/                        #   Cross-page shared components
│   └── lib/                           #   Common utility library
│
├── resources/                         # Resource files (reference audio, etc.)
├── tests/                             # Test suite
├── docs/                              # Project documentation
└── tmp/                               # Runtime logs and PID files
```

## Quick Start

1. Ensure you have an NVIDIA GPU (VRAM > 28GB) and a Linux system
2. Install FFmpeg
3. Run `bash install.sh` to install dependencies
4. Copy configuration: `cp config.example.json config.json`
5. Run `CUDA_VISIBLE_DEVICES=0 bash start_all.sh` to start the service
6. Visit `https://localhost:8006`

For detailed deployment instructions, see [Configuration & Deployment](deployment.md).
