import asyncio
import json
import logging
import time
from urllib.parse import parse_qs

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlmodel import Session
import websockets

from app.api.schemas.enums import InputStatus, InputType
from app.core.config import (
    ALLOWED_ORIGINS,
    FIREFORM_STREAMING_ENABLED,
    WHISPER_ENABLED,
    WHISPER_WS_URL,
)
from app.db.database import engine
from app.models import Input
from app.services import llm
from app.services.whisper import call_whisper_asr, check_whisper_available

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_db_session() -> Session:
    from app.api.deps import get_db
    from app.main import app as fastapi_app
    override = fastapi_app.dependency_overrides.get(get_db)
    if override:
        return next(override())
    return Session(engine)


def _is_origin_allowed(origin: str | None) -> bool:
    if not origin:
        return True
    if origin in ALLOWED_ORIGINS:
        return True
    # Allow localhost and 127.0.0.1 variants
    if "localhost" in origin or "127.0.0.1" in origin:
        return True
    return False


async def _translate_text_to_english(text: str) -> str:
    """Translate non-English narrative into English using the configured LLM.

    Returns empty string if translation fails or input is empty.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return ""

    schema = {
        "type": "object",
        "properties": {
            "english_translation": {
                "type": "string",
                "description": "English translation of the provided text.",
            }
        },
        "required": ["english_translation"],
    }
    prompt = (
        "Translate the following speech transcript into English concisely. "
        "Preserve clinical, incident, and technical details accurately:\n\n"
        f"{cleaned}"
    )

    try:
        # Run blocking LLM call in thread pool to avoid blocking the asyncio event loop
        loop = asyncio.get_running_loop()
        registry = llm.get_registry()
        json_str = await loop.run_in_executor(
            None,
            lambda: registry.generate_json(prompt, schema),
        )
        data = json.loads(json_str)
        return str(data.get("english_translation") or "").strip()
    except Exception as exc:
        logger.debug("Live translation skipped or failed: %s", exc)
        return ""


@router.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket) -> None:
    """Bi-directional streaming transcription and live translation gateway.

    Protocol:
    - Client sends query params: ?translate=true&language=auto
    - Client streams binary messages containing raw 16kHz mono 16-bit little-endian PCM audio.
    - Client sends text message '{"action": "stop"}' to finalize recording.
    - Server sends JSON messages:
        {"type": "interim", "text": str, "transcript": str, "translation": str}
        {"type": "final", "text": str, "transcript": str, "translation": str, "input_id": str}
        {"type": "error", "error_code": str, "message": str}
    """
    origin = websocket.headers.get("origin")
    if not _is_origin_allowed(origin):
        await websocket.close(code=1008, reason="Origin not allowed")
        return

    await websocket.accept()

    # 1. Verification of streaming feature flags
    if not FIREFORM_STREAMING_ENABLED:
        await websocket.send_json({
            "type": "error",
            "error_code": "STREAMING_DISABLED",
            "message": "Real-time streaming transcription is disabled in server configuration.",
        })
        await websocket.close(code=1008, reason="Streaming disabled")
        return

    # 2. Check if Whisper service is available
    if not WHISPER_ENABLED or not check_whisper_available():
        await websocket.send_json({
            "type": "error",
            "error_code": "WHISPER_UNAVAILABLE",
            "message": (
                "Speech transcription service is unavailable. "
                "Ensure the Whisper Docker sidecar or local Whisper service is running."
            ),
        })
        await websocket.close(code=1011, reason="Whisper service unavailable")
        return

    # Query params
    query_str = websocket.scope.get("query_string", b"").decode("utf-8")
    query_params = parse_qs(query_str)
    enable_translation = query_params.get("translate", ["true"])[0].lower() in ("true", "1", "yes")

    # Connect to upstream faster-whisper-server WebSocket
    try:
        upstream_ws = await websockets.connect(
            WHISPER_WS_URL,
            ping_interval=20,
            ping_timeout=20,
        )
    except Exception as exc:
        logger.exception("Failed to connect to upstream Whisper WebSocket at %s", WHISPER_WS_URL)
        await websocket.send_json({
            "type": "error",
            "error_code": "UPSTREAM_CONNECT_FAILED",
            "message": f"Could not connect to Whisper engine at {WHISPER_WS_URL}: {exc}",
        })
        await websocket.close(code=1011, reason="Upstream connection failed")
        return

    transcript_state = {
        "latest_transcript": "",
        "latest_translation": "",
        "last_translated_text": "",
        "last_translation_time": 0.0,
        "stopped": False,
    }

    accumulated_audio_bytes = bytearray()

    async def translate_debounced(text: str) -> None:
        """Trigger translation if text changed meaningfully and debounce interval passed."""
        if not enable_translation or not text:
            return
        now = time.monotonic()
        # Debounce: at least 1.5 seconds between LLM translation calls
        if text != transcript_state["last_translated_text"] and (now - transcript_state["last_translation_time"] >= 1.5):
            transcript_state["last_translated_text"] = text
            transcript_state["last_translation_time"] = now
            translation = await _translate_text_to_english(text)
            if translation:
                transcript_state["latest_translation"] = translation
                try:
                    await websocket.send_json({
                        "type": "interim",
                        "text": transcript_state["latest_transcript"],
                        "transcript": transcript_state["latest_transcript"],
                        "translation": transcript_state["latest_translation"],
                    })
                except Exception:
                    pass

    async def receive_from_upstream():
        """Listen to incoming transcription hypotheses from faster-whisper-server."""
        try:
            async for raw_msg in upstream_ws:
                try:
                    data = json.loads(raw_msg)
                except Exception:
                    continue

                text = ""
                if isinstance(data, dict):
                    if "text" in data and data["text"]:
                        text = str(data["text"]).strip()
                    elif "segments" in data and isinstance(data["segments"], list):
                        text = " ".join(s.get("text", "") for s in data["segments"] if isinstance(s, dict)).strip()

                if text:
                    transcript_state["latest_transcript"] = text
                    await websocket.send_json({
                        "type": "interim",
                        "text": text,
                        "transcript": text,
                        "translation": transcript_state["latest_translation"],
                    })
                    if enable_translation:
                        asyncio.create_task(translate_debounced(text))

        except websockets.ConnectionClosed:
            logger.debug("Upstream Whisper WebSocket closed")
        except Exception as exc:
            logger.debug("Error in receive_from_upstream: %s", exc)

    upstream_task = asyncio.create_task(receive_from_upstream())

    try:
        while True:
            msg = await websocket.receive()
            if "bytes" in msg and msg["bytes"]:
                chunk = msg["bytes"]
                accumulated_audio_bytes.extend(chunk)
                if not upstream_task.done():
                    await upstream_ws.send(chunk)

            elif "text" in msg and msg["text"]:
                try:
                    payload = json.loads(msg["text"])
                    if payload.get("action") == "stop":
                        transcript_state["stopped"] = True
                        break
                except json.JSONDecodeError:
                    pass

            elif msg.get("type") == "websocket.disconnect":
                break

    except WebSocketDisconnect:
        logger.debug("Client disconnected from /ws/transcribe")
    except Exception as exc:
        logger.warning("Error reading from client WebSocket: %s", exc)
    finally:
        try:
            await upstream_ws.close()
        except Exception:
            pass
        upstream_task.cancel()

    # Finalize transcription: if streaming produced a hypothesis, use it;
    # otherwise transcribe accumulated PCM audio directly as fallback.
    final_transcript = transcript_state["latest_transcript"].strip()

    if not final_transcript and accumulated_audio_bytes:
        try:
            loop = asyncio.get_running_loop()
            raw_pcm = bytes(accumulated_audio_bytes)
            # Create a simple WAV from the raw 16kHz mono PCM for REST ASR
            import io
            import wave
            wav_io = io.BytesIO()
            with wave.open(wav_io, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(raw_pcm)
            wav_bytes = wav_io.getvalue()
            final_transcript = await loop.run_in_executor(
                None,
                lambda: call_whisper_asr(wav_bytes, "recording.wav", "audio/wav"),
            )
            final_transcript = (final_transcript or "").strip()
        except Exception as exc:
            logger.warning("Fallback transcription failed: %s", exc)

    final_translation = transcript_state["latest_translation"].strip()
    if enable_translation and final_transcript:
        if final_transcript != transcript_state["last_translated_text"] or not final_translation:
            final_translation = await _translate_text_to_english(final_transcript) or final_translation

    # Persist canonical Input record into database
    input_id = None
    if final_transcript:
        try:
            with _get_db_session() as session:
                input_record = Input(
                    input_type=InputType.voice,
                    status=InputStatus.ready,
                    transcript=final_transcript,
                    character_count=len(final_transcript),
                    word_count=len(final_transcript.split()),
                )
                session.add(input_record)
                session.commit()
                session.refresh(input_record)
                input_id = str(input_record.input_id)
        except Exception as exc:
            logger.exception("Failed to persist final streaming input record: %s", exc)

    # Send final packet to client
    try:
        await websocket.send_json({
            "type": "final",
            "text": final_transcript,
            "transcript": final_transcript,
            "translation": final_translation,
            "input_id": input_id,
        })
        await websocket.close(code=1000)
    except Exception:
        pass
