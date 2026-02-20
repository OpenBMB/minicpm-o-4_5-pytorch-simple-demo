# MiniCPM-o 4.5 PyTorch Simple Demo System

[中文文档 (Chinese Documentation)](README_zh.md)

This demo system is officially provided by the `MiniCPM-o 4.5` model training team. It uses a PyTorch + CUDA inference backend, combined with a lightweight frontend-backend design, aiming to demonstrate the full audio-video omnimodal full-duplex capabilities of MiniCPM-o 4.5 in a transparent, concise, and lossless manner.

This demo system supports text/image/audio input with real-time voice output. It supports non-streaming, half-duplex, and full-duplex interaction modes.

| Mode | Features | I/O Modalities | Paradigm
|------|----------|------|------
| **Turn-based Chat** | Low-latency streaming interaction; requires button or VAD (Voice Activity Detection) to trigger responses; high response accuracy; strong basic capabilities | Audio + Text input, Audio + Text output | Turn-based
| **Omnimodal Full-Duplex** | Real-time omnimodal full-duplex interaction; visual and voice input with simultaneous voice output; model autonomously decides when to speak; powerful cutting-edge capabilities | Vision + Audio input, Text + Voice output | Full-duplex
| **Audio Full-Duplex** | Real-time audio full-duplex interaction; voice input and voice output happen simultaneously; model autonomously decides when to speak; powerful cutting-edge capabilities | Audio input, Text + Voice output | Full-duplex

The 3 currently supported modes share a single model instance with millisecond-level hot-switching (< 0.1ms). More modes will be supported soon.

**Additional features:**

- Customizable system prompts
- Customizable reference audio
- Simple and readable codebase for continual development
- Serve as API backend for third-party applications

## Architecture

```
Frontend (HTML/JS)
    |  HTTPS / WSS
Gateway (:8006, HTTPS)
    |  HTTP / WS (internal)
Worker Pool (:22400+)
    +-- Worker 0 (GPU 0)
    +-- Worker 1 (GPU 1)
    +-- ...
```

- **Frontend** — Mode selection homepage, Turn-based Chat, Omni / Audio Duplex full-duplex interaction, Admin Dashboard
- **Gateway** — Request routing and dispatching, WebSocket proxy, request queuing and session affinity
- **Worker** — Each Worker occupies one GPU exclusively, supports Turn-based Chat / Duplex protocols, Duplex supports pause/resume (auto-release on timeout)


## Quick Start

### Check System Requirements
1. Make sure you have an NVIDIA GPU with more than 28GB of VRAM.
2. Make sure your machine is running a Linux operating system.

### Deployment Steps
**1. Install Python 3.10**

We recommend using miniconda to install Python 3.10.

```bash
mkdir -p ./miniconda3_install_tmp

# Download the miniconda3 installation script
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_25.11.1-1-Linux-x86_64.sh -O ./miniconda3_install_tmp/miniconda.sh 

# Install miniconda3 into the project directory
bash ./miniconda3_install_tmp/miniconda.sh -b -u -p ./miniconda3 
```

After installation, you will have an empty base environment. Activate this base environment, which uses Python 3.10 by default.

```bash
source ./miniconda3/bin/activate
python --version # Should display 3.10.x
```

**2. Install Dependencies for MiniCPM-o 4.5**

Using the `install.sh` script in the project directory is the fastest way. It creates a venv virtual environment named `base` under `.venv` in the project directory and installs all dependencies.

```bash
source ./miniconda3/bin/activate
bash ./install.sh
```

If you have a good network connection, the entire installation process takes about 5 minutes. If you are in China, consider using a third-party PyPI mirror such as the Tsinghua mirror.

You can also install dependencies manually in 2 steps:

```bash
# First, prepare an empty Python 3.10 environment
source ./miniconda3/bin/activate
python -m venv .venv/base
source .venv/base/bin/activate

# Install PyTorch
pip install "torch==2.8.0" "torchaudio==2.8.0"

# Install the remaining dependencies
pip install -r requirements.txt
```

**3. Create Configuration File**

Copy `config.example.json` to `config.json` in the project directory.

```bash
cp config.example.json config.json
```

The model path (`model_path`) defaults to `openbmb/MiniCPM-o-4_5`. If you have access to Hugging Face, no modification is needed — the model will be automatically pulled from Hugging Face.

<details>
<summary>Click to expand detailed instructions about model path</summary>

(Optional) If you prefer to download model weights to a fixed location, or cannot access Hugging Face, you can modify `model_path` to your local model path.
```bash
# Install huggingface cli
pip install -U huggingface_hub

# Download the model
huggingface-cli download openbmb/MiniCPM-o-4_5 --local-dir /path/to/your/MiniCPM-o-4_5

```

If you cannot access Hugging Face, you can use the following two methods to download the model in advance.

- Download the model using hf-mirror

```bash
pip install -U huggingface_hub

export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download openbmb/MiniCPM-o-4_5 --local-dir /path/to/your/MiniCPM-o-4_5
```

- Download the model using ModelScope

```bash
pip install modelscope

modelscope download --model OpenBMB/MiniCPM-o-4_5 --local_dir /path/to/your/MiniCPM-o-4_5
```


</details>

<br/>

Modify `"gateway_port": 8006` to change the deployment port. The default is 8006.


**4. Start the Service**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash start_all.sh
```

After the service starts, visit https://localhost:8006. The self-signed certificate will trigger a browser warning — click "Advanced" → "Proceed" to continue.

<details>
<summary>Click to expand detailed instructions about startup options</summary>

The following are advanced startup options, currently for developer reference.
```bash
CUDA_VISIBLE_DEVICES=0,1 bash start_all.sh          # Specify GPUs
bash start_all.sh --compile                          # torch.compile acceleration (experimental, unstable)
bash start_all.sh --http                             # Downgrade to HTTP (not recommended, mic/camera APIs require HTTPS)
```

**Manual Startup (step by step):**
```bash
# Worker (one per GPU)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/base/bin/python worker.py --worker-index 0 --gpu-id 0

# Gateway
PYTHONPATH=. .venv/base/bin/python gateway.py --port 10024 --workers localhost:22400
```
</details>

**5. Stop the Service:**
```bash
pkill -f "gateway.py|worker.py"
```

<br/>
<br/>


## Known Issues and Improvement Plans

- In Turn-based Chat mode, image input is temporarily unavailable — only audio and text input are supported. An image Q&A mode will be split out soon.
- Half-duplex voice call (no button required to trigger responses) is under development and will be merged soon.
- In Audio Full-Duplex mode, echo cancellation currently has issues affecting interruption success rate. Using headphones is recommended. A fix is coming soon.
- In voice mode, due to the model's training strategy, Chinese and English calls require corresponding language system prompts.

<br/>

## Project Structure

**Project Code Structure**
```
minicpmo45_service/
├── config.json               # Service config (copied from config.example.json, gitignored)
├── config.example.json       # Config example (full fields + defaults)
├── config.py                 # Config loading logic (Pydantic definition + JSON loading)
├── requirements.txt          # Python dependencies
├── start_all.sh              # One-click startup script
│
├── gateway.py                # Gateway (routing, queuing, WS proxy)
├── worker.py                 # Worker (inference service)
├── gateway_modules/          # Gateway business modules
│
├── core/                     # Core encapsulation
│   ├── schemas/              # Pydantic schemas (request/response)
│   └── processors/           # Inference processors (UnifiedProcessor)
│
├── MiniCPMO45/               # Model core inference code
├── static/                   # Frontend pages
├── resources/                # Resource files (reference audio, etc.)
├── tests/                    # Tests
└── tmp/                      # Runtime logs and PID files
```

**Frontend Routes**

| Page | URL |
|------|-----|
| Non-streaming | https://localhost:8006 |
| Omnimodal Full-Duplex | https://localhost:8006/omni |
| Audio Full-Duplex | https://localhost:8006/audio_duplex |
| Dashboard | https://localhost:8006/admin |
| API Docs | https://localhost:8006/docs |

<br/>
<br/>

## Configuration

### config.json — Unified Configuration File

All configurations are centralized in `config.json` (copied from `config.example.json`).
`config.json` is gitignored and will not be committed.

**Configuration Priority**: CLI arguments > config.json > Pydantic defaults

| Group | Field | Default | Description |
|-------|-------|---------|-------------|
| **model** | `model_path` | _(required)_ | HuggingFace format model directory |
| model | `pt_path` | null | Additional .pt weight override |
| model | `attn_implementation` | `"auto"` | Attention implementation: `"auto"`/`"flash_attention_2"`/`"sdpa"`/`"eager"` |
| **audio** | `ref_audio_path` | `assets/ref_audio/ref_minicpm_signature.wav` | Default TTS reference audio |
| audio | `playback_delay_ms` | 200 | Frontend audio playback delay (ms); higher = smoother but more latency |
| audio | `chat_vocoder` | `"token2wav"` | Chat mode vocoder: `"token2wav"` (default) or `"cosyvoice2"` |
| **service** | `gateway_port` | 8006 | Gateway port |
| service | `worker_base_port` | 22400 | Worker base port |
| service | `max_queue_size` | 100 | Maximum queued requests |
| service | `request_timeout` | 300.0 | Request timeout (seconds) |
| service | `compile` | false | torch.compile acceleration |
| service | `data_dir` | "data" | Data directory |
| **duplex** | `pause_timeout` | 60.0 | Duplex pause timeout (seconds) |

**Minimal Configuration** (only model path required):
```json
{"model": {"model_path": "/path/to/model"}}
```

## CLI Argument Overrides

```bash
# Worker
python worker.py --model-path /alt/model --pt-path /alt/weights.pt --ref-audio-path /alt/ref.wav --compile

# Gateway
python gateway.py --port 10025 --workers localhost:22400,localhost:22401 --http
```


## Resource Consumption

| Resource | Token2Wav (default) |
|----------|---------------------|
| VRAM (per Worker, after initialization) | ~21.5 GB |
| Model loading time | ~16s |
| Mode switching latency | < 0.1ms |

> Compile mode incurs an additional ~60s compilation time on the first inference.

## Testing

```bash

# Schema unit tests (no GPU required)
PYTHONPATH=. .venv/base/bin/python -m pytest tests/test_schemas.py -v

# Processor tests (GPU required)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/base/bin/python -m pytest tests/test_chat.py tests/test_streaming.py tests/test_duplex.py -v -s

# API integration tests (service must be running)
PYTHONPATH=. .venv/base/bin/python -m pytest tests/test_api.py -v -s
```
