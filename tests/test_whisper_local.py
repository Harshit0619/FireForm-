"""Local (in-process) Whisper: routing, audio decoding and the streaming buffer.

The model is faked so these run without the model files.
"""

import io
import json
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import whisper, whisper_local

RATE = whisper_local.SAMPLE_RATE


def _pcm(seconds: float) -> bytes:
    return np.zeros(int(seconds * RATE), dtype="<i2").tobytes()


def _wav(seconds: float) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(_pcm(seconds))
    return buf.getvalue()


def test_local_mode_never_calls_the_sidecar(monkeypatch):
    def _no_network(*args, **kwargs):
        raise AssertionError("The sidecar must not be called in local mode")

    monkeypatch.setattr(whisper, "WHISPER_LOCAL", True)
    monkeypatch.setattr(whisper.requests, "post", _no_network)
    monkeypatch.setattr(whisper.requests, "get", _no_network)
    monkeypatch.setattr(whisper_local, "transcribe_bytes", lambda audio: "engine twelve on scene")

    assert whisper.call_whisper_asr(b"audio", "memo.wav", "audio/wav") == "engine twelve on scene"


def test_local_mode_availability_follows_the_engine(monkeypatch):
    monkeypatch.setattr(whisper, "WHISPER_ENABLED", True)
    monkeypatch.setattr(whisper, "WHISPER_LOCAL", True)
    monkeypatch.setattr(whisper_local, "is_available", lambda: True)
    assert whisper.check_whisper_available() is True
    monkeypatch.setattr(whisper_local, "is_available", lambda: False)
    assert whisper.check_whisper_available() is False


def test_health_reports_the_local_engine(monkeypatch):
    from app.api.routes import system

    monkeypatch.setattr(system, "WHISPER_ENABLED", True)
    monkeypatch.setattr(system, "WHISPER_LOCAL", True)
    monkeypatch.setattr(whisper_local, "is_available", lambda: True)
    assert system._check_whisper().status == "healthy"
    monkeypatch.setattr(whisper_local, "is_available", lambda: False)
    assert system._check_whisper().status == "unhealthy"


def test_uploaded_audio_is_decoded_and_transcribed(monkeypatch):
    seen = []

    def fake(samples):
        seen.append((samples.dtype, len(samples)))
        return [(1.0, " engine  twelve "), (2.0, "on scene")]

    monkeypatch.setattr(whisper_local, "_segments", fake)
    assert whisper_local.transcribe_bytes(_wav(2.0)) == "engine twelve on scene"
    assert seen[0][0] == np.float32
    assert abs(seen[0][1] - 2 * RATE) < 100


def test_unreadable_audio_is_a_runtime_error():
    with pytest.raises(RuntimeError, match="Transcription failed"):
        whisper_local.transcribe_bytes(b"this is not audio")


def test_stream_transcribes_short_recordings_whole(monkeypatch):
    seen = []

    def fake(samples):
        seen.append(len(samples) / RATE)
        return [(len(samples) / RATE, "engine twelve responded")]

    monkeypatch.setattr(whisper_local, "_segments", fake)
    t = whisper_local.StreamingTranscriber()
    t.feed(_pcm(3))
    assert t.transcribe() == "engine twelve responded"
    t.feed(_pcm(2) + b"\x01")  # a stray half-sample is ignored
    assert t.transcribe() == "engine twelve responded"
    assert seen == [3.0, 5.0]


def test_stream_commits_settled_segments_on_long_recordings(monkeypatch):
    seen = []

    def fake(samples):
        seconds = len(samples) / RATE
        seen.append(round(seconds, 1))
        if seconds >= 15:
            return [(6.0, "first sentence."), (12.0, "second sentence."), (seconds, "third so far")]
        return [(seconds, "third sentence done.")]

    monkeypatch.setattr(whisper_local, "_segments", fake)
    t = whisper_local.StreamingTranscriber()
    t.feed(_pcm(16))
    assert t.transcribe() == "first sentence. second sentence. third so far"
    t.feed(_pcm(2))
    # Only the 6 s after the last settled segment is transcribed again.
    assert t.transcribe() == "first sentence. second sentence. third sentence done."
    assert seen == [16.0, 6.0]


def test_local_stream_returns_a_final_transcript(monkeypatch):
    from app.api.routes import streaming

    monkeypatch.setattr(streaming, "FIREFORM_STREAMING_ENABLED", True)
    monkeypatch.setattr(streaming, "WHISPER_ENABLED", True)
    monkeypatch.setattr(streaming, "WHISPER_LOCAL", True)
    monkeypatch.setattr(streaming, "check_whisper_available", lambda: True)
    monkeypatch.setattr(
        whisper_local,
        "_segments",
        lambda samples: [(len(samples) / RATE, "kitchen fire at oak street")] if len(samples) else [],
    )

    with TestClient(app).websocket_connect("/api/v1/ws/transcribe?translate=false") as ws:
        ws.send_bytes(_pcm(1.5))
        ws.send_text(json.dumps({"action": "stop"}))
        while True:
            data = ws.receive_json()
            if data["type"] == "final":
                break
    assert data["text"] == "kitchen fire at oak street"
    assert data["input_id"]
