# Real-Time Whisper Streaming & Live Translation

This document details the real-time audio streaming architecture, WebSocket protocol, and configuration introduced in FireForm for low-latency Whisper transcription and live English translation.

---

## 1. Architecture Overview

```
 [Browser / Electron UI]
        │
        │  WebSocket: /api/v1/ws/transcribe?translate=true
        ▼
 [FastAPI Gateway: streaming.py]
   ├── Proxies binary 16kHz s16le PCM frames ──► [faster-whisper-server:8000/v1/audio/transcriptions]
   │                                                 (LocalAgreement2 live interim & final hypotheses)
   └── Forwards interim & final text ────────────► [LLM Service] (Debounced Live English Translation)
        │
        └── On "stop" / connection end ──────────► [Database] (Persists canonical Input record)
```

1. **Audio Capture**: The frontend captures audio via the browser `AudioContext` at 16,000 Hz, downmixing to mono 16-bit little-endian linear PCM (`s16le`).
2. **WebSocket Gateway**: Clients connect to `/api/v1/ws/transcribe`.
   - Audio is streamed as binary frames.
   - Upstream connection to `fedirz/faster-whisper-server` is opened over WebSocket.
3. **Live Transcription**: The Whisper engine processes audio using `LocalAgreement2` streaming consensus, returning interim and confirmed transcript updates.
4. **Live Translation (English)**: When enabled (`translate=true`), interim text is periodically translated to English using the configured LLM provider and returned in `interim` and `final` messages.
5. **Persistence**: When the client sends `{"action": "stop"}` (or disconnects after speaking), the final recognized text and translation are automatically saved into the database as a canonical `Input` record (ready for form autofill).

---

## 2. Configuration & Privacy Flags

Environment variables defined in `app/core/config.py`:

| Variable | Default | Description |
|---|---|---|
| `FIREFORM_STREAMING_ENABLED` | `true` | Privacy & consent master switch for live streaming. When `false`, the WebSocket rejects connections with error code `STREAMING_DISABLED`. |
| `WHISPER_ENABLED` | `true` | Master flag indicating if Whisper ASR is enabled (set to `false` in desktop lightweight mode without sidecar). |
| `WHISPER_HOST` | `http://whisper:8000` | HTTP URL of the faster-whisper sidecar container. |
| `WHISPER_WS_URL` | Auto-derived (`ws://whisper:8000/v1/audio/transcriptions`) | Upstream WebSocket endpoint. |

---

## 3. WebSocket Protocol Specification

### Connection URL
`GET /api/v1/ws/transcribe[?language=<iso_code>&translate=<true|false>]`

- `language` *(optional)*: Spoken language code (e.g., `es`, `fr`, `de`, `en`). If omitted, Whisper auto-detects.
- `translate` *(optional, default: `true`)*: Whether to compute live English translations alongside transcription.

### Client Messages

- **Binary Frames**: Raw linear PCM 16-bit signed, little-endian, mono, 16,000 Hz (chunks of 1,024 to 4,096 bytes recommended).
- **Text Frames**:
  - `{"action": "stop"}`: Tells the backend the speaker has finished recording. Flushes final transcription, runs final translation, persists to database, and sends the `final` frame.

### Server Messages

All server messages are JSON objects:

#### 1. Interim Transcript
```json
{
  "type": "interim",
  "text": "The patient is experiencing severe chest pain",
  "translation": "The patient is experiencing severe chest pain"
}
```

#### 2. Final Result
```json
{
  "type": "final",
  "text": "The patient is experiencing severe chest pain since this morning.",
  "translation": "The patient is experiencing severe chest pain since this morning.",
  "input_id": 42
}
```

#### 3. Error
```json
{
  "type": "error",
  "error_code": "WHISPER_UNAVAILABLE",
  "detail": "Whisper sidecar service is not running or unreachable"
}
```

Error codes include:
- `STREAMING_DISABLED`: Live audio streaming is disabled by server policy.
- `WHISPER_UNAVAILABLE`: Whisper container is down or offline.
- `WHISPER_ERROR`: Upstream Whisper failure during streaming.

---

## 4. Docker Deployment

Both `docker/dev/compose.yml` and `docker/prod/compose.yml` run the optimized container:
`fedirz/faster-whisper-server:latest-cpu`

```yaml
  whisper:
    image: fedirz/faster-whisper-server:latest-cpu
    environment:
      - WHISPER__MODEL=Systran/faster-whisper-small
      - WHISPER__DEVICE=cpu
      - WHISPER__COMPUTE_TYPE=int8
    ports:
      - "127.0.0.1:9000:8000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 10s
      timeout: 5s
      retries: 5
```
