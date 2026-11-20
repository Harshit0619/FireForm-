import requests

from app.core.config import WHISPER_ENABLED, WHISPER_HOST, WHISPER_LOCAL, WHISPER_MODEL
from app.services import whisper_local

_whisper_api_mode: str | None = None


def call_whisper_asr(audio_bytes: bytes, filename: str, content_type: str) -> str:
    """Transcribe audio and return the text.

    With WHISPER_LOCAL the audio is transcribed in this process. Otherwise it is posted
    to the Whisper ASR sidecar.

    Supports both legacy /asr endpoints (onerahmet) and OpenAI-compatible
    endpoints (/v1/audio/transcriptions) for faster-whisper-server.

    Raises ConnectionError if the service is unreachable, RuntimeError for any
    other HTTP failure. Callers map these to their own error codes.
    """
    global _whisper_api_mode

    if WHISPER_LOCAL:
        return whisper_local.transcribe_bytes(audio_bytes)

    # If known to be v1, skip legacy attempt
    if _whisper_api_mode != "v1":
        legacy_url = f"{WHISPER_HOST}/asr"
        legacy_files = {"audio_file": (filename, audio_bytes, content_type)}
        legacy_params = {"task": "transcribe", "output": "json", "encode": "true"}

        try:
            response = requests.post(legacy_url, params=legacy_params, files=legacy_files, timeout=120)
            if response.status_code == 404:
                _whisper_api_mode = "v1"
            else:
                response.raise_for_status()
                _whisper_api_mode = "legacy"
                try:
                    return (response.json().get("text") or "").strip()
                except ValueError:
                    return response.text.strip()
        except requests.exceptions.ConnectionError as exc:
            try:
                requests.get(f"{WHISPER_HOST}/docs", timeout=2)
            except requests.exceptions.RequestException:
                raise ConnectionError(
                    f"Could not connect to the speech-to-text service at {WHISPER_HOST}. "
                    "Please ensure the whisper service is running."
                ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Transcription failed: {exc}") from exc

    # 2. Try OpenAI-compatible endpoint (faster-whisper-server)
    v1_url = f"{WHISPER_HOST}/v1/audio/transcriptions"
    v1_files = {"file": (filename, audio_bytes, content_type)}
    v1_data = {"model": WHISPER_MODEL, "response_format": "json"}

    try:
        response = requests.post(v1_url, files=v1_files, data=v1_data, timeout=120)
        response.raise_for_status()
        _whisper_api_mode = "v1"
        try:
            return (response.json().get("text") or "").strip()
        except ValueError:
            return response.text.strip()
    except requests.exceptions.ConnectionError as exc:
        raise ConnectionError(
            f"Could not connect to the speech-to-text service at {WHISPER_HOST}. "
            "Please ensure the whisper service is running."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Transcription failed: {exc}") from exc


def check_whisper_available() -> bool:
    """Return True if Whisper is enabled and its engine (local or sidecar) can be used."""
    if not WHISPER_ENABLED:
        return False
    if WHISPER_LOCAL:
        return whisper_local.is_available()
    for path in ("/docs", "/health", "/v1/models", ""):
        try:
            resp = requests.get(f"{WHISPER_HOST}{path}", timeout=2)
            if resp.ok:
                return True
        except requests.exceptions.RequestException:
            continue
    return False
