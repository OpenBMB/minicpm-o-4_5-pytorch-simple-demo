# Gateway Module Details

The Gateway is the system's request entry point and routing hub. It does not load models and is responsible for request dispatching, WebSocket proxying, queue management, and resource management.

## Module Structure

```
gateway.py                         # Gateway main service
gateway_modules/
├── __init__.py                    # Package identifier
├── worker_pool.py                 # WorkerPool scheduler
├── app_registry.py                # Application registry
├── models.py                      # Data model definitions
└── ref_audio_registry.py          # Reference audio registry
```

## gateway.py — Main Service

The Gateway is built on **FastAPI** and runs with **uvicorn**, providing HTTP REST API and WebSocket endpoints.

### Lifecycle Management

The `lifespan()` async context manager is responsible for:
1. Loading configuration from `config.py`
2. Initializing `WorkerPool` and starting health checks
3. Initializing `RefAudioRegistry` and `AppRegistry`
4. Starting background session cleanup tasks (runs once daily)
5. Configuring HTTPS (self-signed certificate) or HTTP mode

### Core Routes

#### HTTP Endpoints

| Endpoint | Method | Function |
|----------|--------|----------|
| `/health` | GET | Health check |
| `/status` | GET | Global service status |
| `/workers` | GET | Worker list |
| `/api/chat` | POST | Chat inference (stateless) |
| `/api/streaming/stop` | POST | Stop Streaming generation |
| `/api/frontend_defaults` | GET | Frontend default configuration |
| `/api/presets` | GET | System prompt preset list |
| `/api/default_ref_audio` | GET | Default reference audio |
| `/api/assets/ref_audio` | GET | Reference audio list |
| `/api/assets/ref_audio/external` | POST | Upload reference audio |
| `/api/assets/ref_audio/{id}` | DELETE | Delete reference audio |
| `/api/queue` | GET | Queue status |
| `/api/queue/{ticket_id}` | GET/DELETE | Query/cancel queue entry |
| `/api/config/eta` | GET/PUT | ETA configuration management |
| `/api/cache` | GET | KV Cache status |
| `/api/sessions/{sid}` | GET | Session metadata |
| `/api/sessions/{sid}/recording` | GET | Session recording data |
| `/api/sessions/{sid}/assets/{rel}` | GET | Session resource files |
| `/api/sessions/{sid}/download` | GET | Download session |
| `/api/sessions/{sid}/upload-recording` | POST | Upload frontend recording |
| `/api/apps` | GET | Enabled application list |
| `/api/admin/apps` | GET/PUT | Application management (Admin) |

#### WebSocket Endpoints

| Endpoint | Function |
|----------|----------|
| `/ws/chat` | Chat WebSocket proxy |
| `/ws/duplex/{session_id}` | Duplex full-duplex session proxy |

### WebSocket Proxy Mechanism

**Streaming Proxy**:
1. Accepts client WebSocket connection
2. Enqueues the request via `WorkerPool.enqueue("chat")`
3. While queued, pushes `queued` / `queue_update` messages to the client
4. Once a Worker is obtained, establishes a WebSocket connection to the Worker
5. Forwards messages bidirectionally (client ↔ worker)
6. Releases the Worker upon completion

**Duplex Proxy**:
1. Accepts client WebSocket connection
2. Enqueues the request via `WorkerPool.enqueue("omni_duplex" / "audio_duplex")`
3. Once a Worker is obtained, exclusively holds that Worker
4. Starts two parallel tasks: `client_to_worker` and `worker_to_client`
5. Cleans up and releases when the client disconnects or sends `stop`

### Security Measures

- Session IDs are validated via regex `^[a-zA-Z0-9_-]{1,64}$` to prevent path traversal
- Upload file size limit of 200MB
- HTTPS enabled by default (self-signed certificate)

---

## worker_pool.py — WorkerPool Scheduler

WorkerPool is the core scheduling component of the Gateway, managing all Worker connections and request dispatching.

### Core Classes

#### WorkerConnection

A Worker connection data class that maintains the state information for a single Worker.

| Property | Type | Description |
|----------|------|-------------|
| `url` | str | Worker address |
| `index` | int | Worker index |
| `status` | GatewayWorkerStatus | Current status |
| `current_task` | str | Current task type |
| `current_session_id` | str | Current session ID |
| `cached_hash` | str | Cached history hash |
| `busy_since` | datetime | Busy since timestamp |

Key methods:
- `mark_busy(task, session_id)` — Mark as busy
- `mark_idle()` — Mark as idle, update `cached_hash`
- `update_duplex_status(status)` — Update Duplex status
- `to_info()` — Convert to API response model

#### QueueEntry

A queue entry containing `ticket` (queue ticket), `future` (awaitable result), and `history_hash` (message history digest).

#### EtaTracker

An ETA estimation tracker that combines baseline values with EMA (Exponential Moving Average) for dynamic wait time estimation.

- `get_eta(task_type)` — Get estimated duration for a single task
- `record_duration(task_type, duration)` — Record actual duration, update EMA
- `update_config(config)` — Update baseline values

#### WorkerPool

| Method | Description |
|--------|-------------|
| `start()` | Start health check loop |
| `stop()` | Stop and clean up |
| `enqueue(task_type, history_hash)` | Enqueue request, returns (ticket, future) |
| `release_worker(worker_url, cached_hash)` | Release Worker |
| `cancel(ticket_id)` | Cancel queued entry |
| `get_ticket(ticket_id)` | Query queue status |
| `get_queue_status()` | Queue snapshot |
| `get_all_workers()` | Worker list |

### FIFO Queue Mechanism

The queue is implemented using `OrderedDict`. All request types (Chat / Streaming / Duplex) share a single unified queue to ensure fair first-come-first-served ordering.

**Enqueue flow** (`enqueue()`):
1. First attempts immediate assignment — if an idle Worker is available, the `Future` resolves immediately and the request does not enter the queue
2. If no idle Worker is available, checks capacity (default 1000); rejects if full
3. Creates a `QueueEntry` (with `QueueTicket` + `asyncio.Future`) and appends to the tail of the OrderedDict

**Dispatch flow** (`_dispatch_next()`) — the sole entry point for Worker assignment:
1. Takes the head entry from the queue
2. Matches an idle Worker based on request type (Streaming uses LRU routing, others use general routing)
3. If found → immediately calls `mark_busy()` + `future.set_result(worker)` + removes the head
4. Loops until no idle Workers remain or the queue is empty

**Trigger points**: `_dispatch_next()` is triggered when a Worker is released, a queue entry is cancelled, or a health check restores a Worker to IDLE.

**Gateway ↔ Worker communication**: The queue is only responsible for Worker assignment (deciding which Worker handles which request) and does not participate in data transfer. After the Gateway obtains a Worker reference, it communicates directly with the Worker's internal port (22400+) via HTTP (Chat: `POST /chat`) or WebSocket (Chat: `/ws/chat`, Duplex: `/ws/duplex`).

### ETA Estimation

`EtaTracker` combines Admin-configured baseline values with runtime EMA (Exponential Moving Average) for dynamic wait time estimation:

- Each task type maintains independent baseline and EMA dynamic values
- Uses EMA values when sufficient samples are available (default >= 3), otherwise uses baseline values
- `_recalc_positions_and_eta()` uses a min-heap to simulate the dispatch chain, popping the earliest available Worker to assign to the queue head, precisely calculating wait times for each queued entry
- ETA results are pushed to queued clients in real-time via WebSocket

### Health Checks

Polls all Workers' `/health` endpoints every 10 seconds (`_health_check_loop()`), updating status. After health checks, triggers `_dispatch_next()` (Workers may have recovered from OFFLINE to IDLE).

**Gateway scheduling authority**: When the Gateway has dispatched a task to a Worker (`_gateway_dispatched=True`), even if the Worker's `/health` temporarily reports `idle` (transient cleanup state), the Gateway will not downgrade it to IDLE, preventing erroneous assignment. `release_worker()` clears this flag, restoring health check authority.

---

## app_registry.py — Application Registry

Manages the enabled/disabled state of frontend applications, supporting runtime dynamic toggling.

### Default Applications

| ID | Name | Default State |
|----|------|---------------|
| `turnbased` | Turn-based Chat | Enabled |
| `omni` | Omnimodal Full-Duplex | Enabled |
| `audio_duplex` | Audio Full-Duplex | Enabled |

### AppRegistry Class

| Method | Description |
|--------|-------------|
| `is_enabled(app_id)` | Check if an application is enabled |
| `set_enabled(app_id, enabled)` | Set enabled state |
| `get_enabled_apps()` | Get list of enabled applications |
| `get_all_apps()` | Get all applications (for Admin) |

Thread safety is ensured via `threading.Lock`.

---

## models.py — Gateway Data Models

Defines all API data models for the Gateway layer, based on Pydantic.

### Enumerations

- **GatewayWorkerStatus** — Worker status enum: `idle` / `busy_streaming` / `duplex_active` / `duplex_paused` / `offline`

### Data Models

| Model | Description |
|-------|-------------|
| `WorkerInfo` | Worker information (url, index, status, task, session_id, cached_hash, busy_since) |
| `QueueTicket` | Queue ticket (ticket_id, position, eta_seconds, task_type) |
| `QueueTicketSummary` | Brief ticket (ticket_id, position) |
| `RunningTaskInfo` | Running task (worker_url, task_type, session_id, started_at, elapsed_s) |
| `QueueStatus` | Queue snapshot (queue_length, entries, running) |
| `ServiceStatus` | Global status (total_workers, idle, busy, queue_length) |
| `EtaConfig` / `EtaStatus` | ETA configuration and status |
| `WorkersResponse` | Worker list response |

---

## ref_audio_registry.py — Reference Audio Registry

Manages storage and metadata for TTS reference audio.

### RefAudioRegistry Class

| Method | Description |
|--------|-------------|
| `upload(name, audio_data, source)` | Upload reference audio (auto-normalizes to 16kHz mono 16-bit WAV) |
| `get(audio_id)` | Query audio information |
| `get_file_path(audio_id)` | Get file path |
| `get_base64(audio_id)` | Get Base64 encoding |
| `list_all()` | List all reference audio |
| `delete(audio_id)` | Delete reference audio |
| `exists(audio_id)` | Check if exists |

Audio uploads are automatically normalized using `librosa` + `soundfile`. Metadata is persisted to a JSON file.
