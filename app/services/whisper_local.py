"""In-process speech-to-text for the desktop build.

The desktop binary has no Whisper sidecar, so it runs faster-whisper itself with a
model shipped inside the bundle. Docker deployments keep using the sidecar.
"""

import io
import re
import threading

import numpy as np

from app.core.config import BASE_DIR

SAMPLE_RATE = 16000
MODEL_NAME = "small"
# Fetched by scripts/fetch_whisper_model.py and bundled by fireform-backend.spec.
MODEL_DIR = BASE_DIR / "models" / f"faster-whisper-{MODEL_NAME}"

_model = None
_model_lock = threading.Lock()


def is_available() -> bool:
    """True if the engine is installed and its model is in the bundle."""
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    return (MODEL_DIR / "model.bin").exists()


def _get_model():
    """Load the model once and reuse it."""
    global _model
    with _model_lock:
        if _model is None:
            from faster_whisper import WhisperModel

            _model = WhisperModel(str(MODEL_DIR), device="cpu", compute_type="int8")
        return _model


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _segments(samples: np.ndarray) -> list[tuple[float, str]]:
    """(end time in seconds, text) for each stretch of speech Whisper finds."""
    if samples.size == 0:
        return []
    segments, _info = _get_model().transcribe(samples, beam_size=1, condition_on_previous_text=False)
    return [(seg.end, _clean(seg.text)) for seg in segments if _clean(seg.text)]


def _decode(audio_bytes: bytes) -> np.ndarray:
    """Decode any container PyAV reads (wav, webm, mp3, m4a...) to 16 kHz mono float32."""
    import av

    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    chunks: list[np.ndarray] = []
    with av.open(io.BytesIO(audio_bytes), mode="r") as container:
        for frame in container.decode(audio=0):
            chunks.extend(out.to_ndarray().reshape(-1) for out in resampler.resample(frame))
        chunks.extend(out.to_ndarray().reshape(-1) for out in resampler.resample(None))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def transcribe_bytes(audio_bytes: bytes) -> str:
    """Transcribe an uploaded audio file."""
    try:
        return _clean(" ".join(text for _end, text in _segments(_decode(audio_bytes))))
    except Exception as exc:
        raise RuntimeError(f"Transcription failed: {exc}") from exc


class StreamingTranscriber:
    """Turns a growing buffer of 16 kHz mono 16-bit PCM into interim and final text.

    Each pass transcribes only the audio after the last committed sentence. Once that
    tail passes COMMIT_AFTER_SECONDS, every segment but the last is committed, so a
    pass stays about as fast for a five-minute memo as for a ten-second one.
    """

    COMMIT_AFTER_SECONDS = 15.0

    def __init__(self):
        self._pcm = bytearray()
        self._committed_bytes = 0
        self._committed_text: list[str] = []
        self._lock = threading.Lock()

    def feed(self, chunk: bytes) -> None:
        with self._lock:
            self._pcm.extend(chunk)

    @property
    def seconds_buffered(self) -> float:
        with self._lock:
            return len(self._pcm) / 2 / SAMPLE_RATE

    def transcribe(self) -> str:
        """Transcript of everything received so far. Call from one thread at a time."""
        with self._lock:
            tail = bytes(self._pcm[self._committed_bytes:])
        tail = tail[: len(tail) - len(tail) % 2]
        segments = _segments(np.frombuffer(tail, dtype="<i2").astype(np.float32) / 32768.0)
        text = _clean(" ".join(self._committed_text + [t for _end, t in segments]))

        if len(tail) / 2 / SAMPLE_RATE >= self.COMMIT_AFTER_SECONDS and len(segments) > 1:
            settled = segments[:-1]
            # Cut on a whole 16-bit sample.
            self._committed_bytes += int(settled[-1][0] * SAMPLE_RATE) * 2
            self._committed_text.extend(t for _end, t in settled)
        return text
