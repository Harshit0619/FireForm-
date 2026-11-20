"""Desktop launcher for the FireForm API.

Starts the Uvicorn ASGI server binding exclusively to 127.0.0.1 for local desktop execution.
Enables FIREFORM_DESKTOP mode by default.
"""

import multiprocessing
import os

# In a frozen binary, a library that starts a helper process (the speech engine does)
# re-runs this file. Without this the helper boots a second copy of the whole API.
multiprocessing.freeze_support()

# Ensure FIREFORM_DESKTOP is enabled when running via desktop binary
os.environ.setdefault("FIREFORM_DESKTOP", "true")

import uvicorn  # noqa: E402
from app.main import app  # noqa: E402


def main():
    port = int(os.getenv("PORT", "8000"))
    host = "127.0.0.1"  # Security: strictly loopback only
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
