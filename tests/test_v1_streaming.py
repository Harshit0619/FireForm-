import json
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
import pytest

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_streaming_disabled(client, monkeypatch):
    """When FIREFORM_STREAMING_ENABLED=False, gateway rejects with error."""
    monkeypatch.setattr("app.api.routes.streaming.FIREFORM_STREAMING_ENABLED", False)

    with client.websocket_connect("/api/v1/ws/transcribe") as websocket:
        data = websocket.receive_json()
        assert data["type"] == "error"
        assert data["error_code"] == "STREAMING_DISABLED"


def test_whisper_unavailable(client, monkeypatch):
    """When Whisper is not reachable, gateway rejects with error."""
    monkeypatch.setattr("app.api.routes.streaming.FIREFORM_STREAMING_ENABLED", True)
    monkeypatch.setattr("app.api.routes.streaming.WHISPER_ENABLED", True)
    monkeypatch.setattr("app.api.routes.streaming.check_whisper_available", lambda: False)

    with client.websocket_connect("/api/v1/ws/transcribe") as websocket:
        data = websocket.receive_json()
        assert data["type"] == "error"
        assert data["error_code"] == "WHISPER_UNAVAILABLE"


def test_origin_not_allowed(client, monkeypatch):
    """Disallowed origin headers get closed with 1008."""
    monkeypatch.setattr("app.api.routes.streaming.FIREFORM_STREAMING_ENABLED", True)
    monkeypatch.setattr("app.api.routes.streaming.ALLOWED_ORIGINS", ["http://allowed.local"])

    with pytest.raises(Exception):
        with client.websocket_connect(
            "/api/v1/ws/transcribe",
            headers={"origin": "http://evil.com"},
        ) as websocket:
            websocket.receive()


def test_streaming_flow_success(client, monkeypatch):
    """Simulates active streaming session with interim and final packets."""
    monkeypatch.setattr("app.api.routes.streaming.FIREFORM_STREAMING_ENABLED", True)
    monkeypatch.setattr("app.api.routes.streaming.WHISPER_ENABLED", True)
    monkeypatch.setattr("app.api.routes.streaming.check_whisper_available", lambda: True)

    # Mock upstream websocket
    mock_upstream = AsyncMock()
    # Mock iterator returning an interim text
    async def mock_aiter(self):
        yield json.dumps({"text": "Patient has severe chest pain."})

    mock_upstream.__aiter__ = mock_aiter
    mock_upstream.send = AsyncMock()
    mock_upstream.close = AsyncMock()

    with patch("websockets.connect", AsyncMock(return_value=mock_upstream)), \
         patch("app.api.routes.streaming._translate_text_to_english", AsyncMock(return_value="Patient has severe chest pain.")):

        with client.websocket_connect("/api/v1/ws/transcribe?translate=true") as ws:
            # Send dummy 16-bit PCM chunk
            ws.send_bytes(b"\x00\x00" * 160)

            # Receive interim update
            interim = ws.receive_json()
            assert interim["type"] == "interim"
            assert "chest pain" in interim["transcript"]

            # Stop recording
            ws.send_text(json.dumps({"action": "stop"}))

            # Drain messages until final packet
            final = None
            for _ in range(5):
                msg = ws.receive_json()
                if msg["type"] == "final":
                    final = msg
                    break

            assert final is not None
            assert final["type"] == "final"
            assert "chest pain" in final["transcript"]
            assert final["input_id"] is not None
