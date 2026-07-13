# AI Detector

Watches one or more video streams or files, and sends you an alert the moment something is detected — a person, an animal, a vehicle, whatever your model is trained to find.

Detection works in two stages:
1. **YOLO** — a fast AI model that scans every frame looking for objects.
2. **VLM** *(optional)* — a smarter AI (like Gemini or GPT-5) that double-checks the detection by looking at the footage and answering a question you define, e.g. *"Is there really a cow mounting another cow?"*. This dramatically reduces false alerts.

For individual Holstein-Friesian recognition, a normal YOLO detector can optionally be enriched with cow identities. Segmented cows are encoded with the MIT-licensed MiewID model and matched against identities stored in SQLite.

Confirmed detections can be sent to **Telegram**, saved to **disk**, or posted to a **webhook**.

---

## Getting Started

### Option 1 — Windows Executable (recommended for most users)

👉 **[Download from the Releases page](https://github.com/ESchouten/ai-detector/releases)**

Pick the right file for your hardware:

| File | GPU |
| :--- | :-- |
| `aidetector-winml-<version>.exe` | Windows 11 with any GPU |
| `aidetector-cuda130-<version>.exe` | Windows 10 with NVIDIA RTX 3000 series or newer |
| `aidetector-cuda126-<version>.exe` | Windows 10 with NVIDIA RTX 2000 series or older |
| `aidetector-osx-<version>` | macOS (CPU / Apple Silicon) |

> **Not sure which to pick?** Start with `winml` on Windows. Use a `cuda` build only if you know your NVIDIA setup matches that CUDA version.

**Setup:**
1. Create a folder, e.g. `C:\aidetector`, and move the downloaded `.exe` into it.
2. In that same folder, create a `config.json` file (see [Configuration](#configuration) below).
3. Double-click the `.exe` — a terminal window opens showing detection logs.

On first run with no `config.json` present, a template is generated automatically. Fill it in and run again.

> **Tip:** Keep the terminal window open while the detector is running. If it closes immediately, there is an error in your `config.json` — check for missing quotes `"` or commas `,`.

### Option 2 — Docker

Useful if you are on Linux, a NAS, or want the detector to restart automatically after a reboot. From the `example/` folder:

```bash
cd example
docker compose up -d
docker compose logs -f aidetector web
```

The example Compose stack also starts the web UI on [http://localhost](http://localhost).

> **Don't have Docker?** [Download Docker Desktop](https://www.docker.com/products/docker-desktop/) — it's free.

### Option 3 — Development (advanced)

```bash
# Install dependencies
uv sync --extra default

# Run the detector (config.json must be in the current directory)
uv run --extra default main

# Sync JSON schema with the Pydantic data models
uv run generate-schema

# Run the quality gates
uv run ruff check src tests
uv run ty check src
uv run pytest
```

### Internal architecture

The Python application is organized around a one-way dependency rule:

```text
domain <- pipeline <- adapters <- application
```

- `domain/` contains frames, observations, events, and immutable policies. It has no infrastructure dependencies.
- `pipeline/` performs scheduling, aggregation, cooldown, validation dispatch, filtering, and lifecycle coordination through typed ports.
- `adapters/` contains Ultralytics, video sources, VLM validation, HTTP/SSE, Telegram, webhook, and disk implementations.
- `application.py` is the composition root. It is the only module that translates configuration objects into runtime policies and concrete adapters.
- `media/` renders and caches event artifacts. Equal artifact requests within one event are encoded only once.
- Generic identity candidates, track aggregation, registry sharing, and enrichment live in `domain/` and `pipeline/`. `dazzlecow/` supplies the cow-specific MiewID, enrollment, and SQLite implementations.

Live observations and completed events are sent through the same generic `Sink.send()` contract. Validation and every event sink have small bounded queues. When an external service falls behind, only its oldest pending event is replaced so inference continues. `tests/test_architecture.py` enforces the dependency direction.

---

## Configuration

All settings go in `config.json`. The file supports JSON Schema, so if you use VS Code it will give you autocomplete and describe every option as you type.

You can run multiple independent detectors in the same file — useful if you have several cameras or want different alert rules per camera.

```json
{
  "onnx":   { ... },
  "health": { ... },
  "detectors": [
    {
      "detection": { ... },
      "yolo":      { ... },
      "vlm":       { ... },
      "exporters": { ... }
    }
  ]
}
```

---

### Top-level fields

| Field       | Default      | Description |
| :---------- | :----------- | :---------- |
| `detectors` | **Required** | List of detector definitions. Each detector can watch one or more sources and use its own YOLO/VLM/exporter settings. |
| `onnx`      |              | Optional ONNX Runtime configuration. Lets you pin a provider and control Windows ML registration. |
| `health`    |              | Optional HTTP healthcheck pinger. Useful for watchdogs, uptime tools, or Home Assistant-style monitoring. |

---

### `detection` — What to watch

| Field             | Default      | Description |
| :---------------- | :----------- | :---------- |
| `source`          | **Required** | Path to a video file, or an RTSP/HTTP stream URL. Use a list `[ ]` for multiple sources. |
| `interval`        | `0`          | How many seconds to wait between processed frames. Set to `0` to process every frame. Useful to reduce load on slow machines. |
| `frame_retention` | `15`         | How many recent frames to keep in memory per source so detections can include earlier context. |
| `frames_width`    | `1280`       | Maximum width retained for event images. Larger source frames are resized while preserving aspect ratio. |

**Examples:**
```json
"source": "rtsp://192.168.1.10/stream"
"source": ["rtsp://camera1", "rtsp://camera2"]
"source": "videos/clip.mp4"
```

---

### `yolo` — Object detection model

This is the fast first-pass AI that scans every frame. Without a YOLO model, the detector simply passes all frames through to the VLM or exporters.

| Field                   | Default      | Description |
| :---------------------- | :----------- | :---------- |
| `model`                 | **Required** | URL or local path to a YOLO model file (`.pt` or `.onnx`). |
| `task`                  | `"detect"`   | YOLO task to run: `"detect"` for detection models or `"segment"` for segmentation models. |
| `tracking`              | `false`      | Keep object track IDs between frames. Sources remain isolated while inference is batched. |
| `confidence`            | `0`          | How confident YOLO must be (0–1) before counting something as a detection. `0.8` means 80% sure. You can also set different thresholds per class — see tip below. |
| `time_max`              | `60`         | Maximum duration in seconds to group frames into one event. If a detection runs longer than this, a new event starts. |
| `timeout`               | `5`          | Seconds of no detections before the current event is considered over. |
| `cooldown`              | `0`          | Seconds to wait after finishing one event before starting a new one. Prevents repeat alerts for the same ongoing situation. Can be set per class. |
| `include_trailing_time` | `1`          | Seconds of extra footage to include after the last detected frame so the event does not end too abruptly. |
| `frames_min`            | `3`          | How many frames in a row must match before the event counts. |
| `imgsz`                 | `640`        | The image size fed into the model. Higher values are more accurate but slower. Most models expect `640`. |

> **Tip — per-class confidence thresholds:**
> Instead of a single number, you can give each class its own threshold:
> ```json
> "confidence": { "person": 0.85, "car": 0.6 }
> ```
> Only the classes you list are evaluated — everything else is ignored.

> **Tip — per-class cooldowns:**
> ```json
> "cooldown": { "person": 60, "car": 30 }
> ```

---

### `identity` — Individual cow recognition

Cow identity is an optional enrichment of a normal YOLO detector. MiewID converts segmented cows into normalized embeddings. Five observations from the same YOLO track are averaged, then compared with the best stored view of each identity. By default, an identity is returned only when cosine similarity is at least `0.68` and the lead over the second-best identity is at least `0.05`; ambiguous tracks remain unknown. The pinned 195 MB ONNX model is downloaded from Hugging Face on first use.

When the primary detector already tracks a `cow` class with segmentation masks, those results are reused. For detection-only models such as CowCatcher, the identity pipeline runs `yolo26m-seg.pt` on the same frame and attaches every identified cow inside the event box to detections such as `mounting`.

```json
{
  "yolo": {
    "model": "cowcatcher.pt",
    "confidence": 0.8
  },
  "identity": {
    "segment_model": "yolo26m-seg.pt",
    "database": "identities/cows.sqlite"
  }
}
```

For enrollment, add an `enrollment` object. The detector records averaged tracklets in the same database until enrollment is finalized in the web app. Afterwards it loads the assigned embeddings from SQLite into memory for matching.

Every animal starts without an identity. `identity_count` is therefore not a list of known identities; it is the optional known herd size. Supplying it turns the first scan into a closed-set assignment and is strongly recommended when every animal is expected to participate. Without it, enrollment is open-set and deliberately conservative: one animal may initially appear as multiple provisional identities, which can be reviewed and merged in the web app. This avoids forcing genuinely new animals into an existing identity.

```json
{
  "yolo": {
    "model": "yolo26m-seg.pt",
    "task": "segment",
    "tracking": true,
    "tracker": "bytetrack.yaml",
    "confidence": { "cow": 0.1 },
    "imgsz": 640
  },
  "identity": {
    "database": "identities/cows.sqlite",
    "enrollment": {
      "identity_count": 46
    },
    "match_threshold": 0.68,
    "match_margin": 0.05
  }
}
```

SQLite is the only persistent identity store. It contains track embeddings, previews, assignments, and animal numbers. At startup and after database changes, the detector builds its normalized NumPy search arrays directly in memory.

Identity databases include the pinned embedding model and dimension. Databases created with an earlier experimental encoder are not compatible with MiewID; the detector reports this at startup and requires a fresh enrollment instead of mixing embeddings from different models.

After enrollment is finalized, reliable matches keep improving existing identities. The detector rechecks the complete track embedding, stores at most one sufficiently distinct sample per track, and retains at most 20 learned samples per identity. Unknown or ambiguous tracks are not assigned automatically and remain available for a later enrollment workflow.

Unknown tracks collected after finalization appear as pending evidence in the web app. Finishing another scan clusters only this pending evidence. A mature cluster is first compared conservatively with the existing gallery: confident matches add evidence to that identity, while unmatched clusters become new identities without changing existing assignments or animal numbers. A cluster needs evidence from at least three tracks; immature clusters remain pending.

All detector pipelines in the same application share a short-lived in-memory identity register. An identity-enabled detector publishes normalized cow locations keyed by source, detector, and YOLO track id. Other detectors reading the same source automatically attach those identities by location, so an event detector such as CowCatcher does not need to run a second identity pipeline. Entries expire after five seconds and are never persisted as identity evidence.

Install the optional identity dependencies with `uv sync --extra default --extra identity`.

For benchmark commands, also install `--extra identity-benchmark`. Use `benchmark-cow-identity-video` for annotated end-to-end localization, tracking, and identity regression tests. `benchmark-cow-enrollment` evaluates clustering against prepared public identity crops. Both use the same pinned MiewID ONNX model as the application.

---

### `vlm` *(optional)* — AI double-check

After YOLO flags something, a Vision Language Model looks at the footage and answers a question you write. Only if the answer seems positive does the detection get exported. This step is optional but greatly reduces false alarms.

Can be a single object or a list. If you provide a list, the VLMs are tried in order until one succeeds.

| Field      | Default      | Description |
| :--------- | :----------- | :---------- |
| `prompt`   | **Required** | The question to ask about the footage, e.g. `"Is there a person in this video?"` |
| `model`    | **Required** | The AI model to use, e.g. `"gemini/gemini-2.0-flash"`. Can also be a list of model names for provider fallback. Supports any model from [LiteLLM](https://docs.litellm.ai/docs/providers). |
| `key`      |              | API key for the model provider (Gemini, OpenAI, etc.). |
| `url`      |              | Custom API endpoint, if you're running a local model. |
| `strategy` | `"VIDEO"`    | `"VIDEO"` — sends the full detection clip to the AI. `"IMAGE"` — sends only a single frame. Video is more accurate but costs more tokens. |
| `timeout`  | `30`         | Maximum seconds to wait for a model response. |

---

### `exporters` *(optional)* — Where to send alerts

You can combine multiple exporters. Each exporter key can be either a single object or a list of objects if you want to send to multiple Telegram chats, webhooks, or disk destinations.

`confidence` on any exporter can be either:
- a single number such as `0.7`
- a per-class map such as `{ "person": 0.8, "car": 0.6 }`

#### 💾 Disk (`disk`)

Saves detection images or frames to a folder on your machine.

| Field             | Default      | Description |
| :---------------- | :----------- | :---------- |
| `directory`       |              | Folder path under `detections/` to save files into, e.g. `"mounts"`. If omitted, the exporter uses the best-matching class name as the directory. |
| `strategy`        | `"BEST"`     | `"BEST"` saves only the highest-confidence frame. `"ALL"` saves every frame from the event. |
| `confidence`      |              | Minimum confidence required to save. Leave empty to save everything. |
| `export_rejected` | `true`       | Whether to also save detections that were rejected by the VLM. |

#### 📱 Telegram (`telegram`)

Sends an alert to a Telegram chat. The bot can include images or a video clip.

> **How to get a bot token:** Talk to [@BotFather](https://t.me/BotFather) on Telegram and follow the steps to create a bot. It gives you a token.
>
> **How to get your chat ID:** Add your bot to a chat, send it a message, then open `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` in your browser — the `chat.id` field is your chat ID.

| Field             | Default      | Description |
| :---------------- | :----------- | :---------- |
| `token`           | **Required** | Your Telegram bot token. |
| `chat`            | **Required** | The Telegram chat or user ID to send alerts to. |
| `confidence`      |              | Minimum confidence required to send. Leave empty to always send. |
| `alert_every`     | `1`          | Only send a notification sound every Nth detection. `1` = every time, `5` = every 5th. |
| `timeout`         | `30`         | HTTP request timeout in seconds. |
| `include_plot`    | `false`      | Include the full frame with a detection box drawn on it. |
| `include_crop`    | `false`      | Include a cropped image of just the detected object. |
| `include_video`   | `true`       | Include an MP4 clip of the detection sequence. |
| `video_width`     | `1280`       | Width of the video clip in pixels. Height is calculated automatically. |
| `video_crf`       | `28`         | Video quality (0–51). Lower = better quality, larger file. `28` is a good default. |
| `export_rejected` | `false`      | Whether to also send detections rejected by the VLM. |

#### 🔗 Webhook (`webhook`)

Posts detection data to an HTTP endpoint. Useful for integrating with other systems.
Generated payload fields follow `config/metadata.schema.json`. Multipart requests
JSON-encode nested fields such as `confidences` and `crops`.

| Field             | Default      | Description |
| :---------------- | :----------- | :---------- |
| `url`             | **Required** | The URL to call when a detection occurs. |
| `method`          | `"POST"`     | HTTP method to use: `GET`, `POST`, `PUT`, `PATCH`, `DELETE`, or `HEAD`. |
| `headers`         |              | Optional HTTP headers map. |
| `body`            |              | Optional raw request body. When set, this is sent instead of the generated detection payload. |
| `timeout`         | `30`         | Request timeout in seconds. The runtime uses 30 seconds when omitted. |
| `token`           |              | Authorization token sent in the request headers. |
| `confidence`      |              | Minimum confidence required to trigger. Leave empty to always trigger. |
| `data_type`       | `"binary"`   | How image/video data is sent: `"binary"`, `"base64"`, or `"none"` for no generated body/files. |
| `data_max`        |              | Maximum payload size in bytes. The image is compressed if it exceeds this. |
| `include_plot`    | `false`      | Include the full frame with detection overlay. |
| `include_crop`    | `true`       | Include a cropped image of the detected area. |
| `include_video`   | `false`      | Include an MP4 clip of the detection sequence. |
| `video_width`     | `1280`       | Width of the video clip in pixels. |
| `video_crf`       | `28`         | Video quality (0–51). Lower = better quality, larger file. |
| `export_rejected` | `false`      | Whether to also POST detections rejected by the VLM. |

#### Live stream (`sse`)

Publishes live object detections and completed event summaries as Server-Sent Events (SSE). The web application uses this feed for its live stream view.

| Field             | Default                  | Description |
| :---------------- | :----------------------- | :---------- |
| `port`            | `8765`                   | HTTP port for the SSE server. Exporters using the same port share one server. |
| `endpoint`        | `/events/<detector index>` | Optional endpoint. When omitted, each detector receives a unique endpoint based on its position in `detectors`. |
| `confidence`      |                           | Minimum confidence for completed detection events. Live track messages are not event-filtered. |
| `export_rejected` | `false`                   | Whether to publish completed detections rejected by the VLM. |

The detector binds this internal server to `127.0.0.1` by default. The web application proxies it through the normal web origin. Container deployments should set `SSE_HOST=0.0.0.0` on the detector and `DETECTOR_HOST=aidetector` on the web service; the example Compose files already do this without publishing port `8765` to the host.

---

### `health` *(optional)* — External heartbeat

Sends a simple periodic HTTP request while the detector is running. This is useful if another system wants to verify that the process is still alive.

| Field      | Default      | Description |
| :--------- | :----------- | :---------- |
| `url`      | **Required** | The URL to ping. |
| `method`   | `"GET"`      | HTTP method to use: `GET`, `POST`, `PUT`, `PATCH`, `DELETE`, or `HEAD`. |
| `interval` | `60`         | Seconds between pings. |
| `timeout`  | `5`          | Request timeout in seconds. |
| `headers`  |              | Optional HTTP headers map. |
| `body`     |              | Optional request body sent as raw text. |

---

### `onnx` *(optional)* — ONNX Runtime behavior

These settings control how the executable configures ONNX Runtime before loading a YOLO model.

| Field      | Default  | Description |
| :--------- | :------- | :---------- |
| `provider` |          | Optional provider name to force, e.g. `"CUDAExecutionProvider"` or `"CPUExecutionProvider"`. If omitted, ONNX Runtime uses its normal provider order. |
| `winml`    | `true`   | Only relevant for the `windowsml` build. If `true`, the app tries to register Windows ML execution providers automatically. |
| `opset`    | `20`     | ONNX opset used when exporting a `.pt` model to ONNX. Lower values can improve compatibility with some runtimes. |

---

### Full example

```json
{
  "onnx": {
    "winml": true
  },
  "health": {
    "url": "https://example.com/health/aidetector",
    "interval": 60
  },
  "detectors": [
    {
      "detection": {
        "source": ["rtsp://camera1", "rtsp://camera2"]
      },
      "yolo": {
        "model": "https://github.com/CowCatcherAI/CowCatcherAI/releases/download/model-V16/cowcatcherV15.pt",
        "confidence": 0.8,
        "frames_min": 3
      },
      "vlm": {
        "prompt": "Do you see cows that are mounting each other?",
        "model": [
          "gemini/gemini-3-flash-preview",
          "gemini/gemini-2.5-flash-lite",
          "gemini/gemini-2.5-flash"
        ],
        "key": "<your_api_key>"
      },
      "exporters": {
        "disk": { "directory": "mounts" },
        "telegram": {
          "token": "<your_bot_token>",
          "chat": "<your_chat_id>"
        }
      }
    }
  ]
}
```

---

## Built With

- **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)** — Fast, accurate object detection.
- **[LiteLLM](https://docs.litellm.ai/)** — Connects to any AI model provider (Gemini, OpenAI, Anthropic, and more).
- **[Pydantic](https://docs.pydantic.dev/)** — Validates your config file and gives clear error messages when something is wrong.

## License

This project is licensed under the AGPL (see [LICENSE](LICENSE)).
