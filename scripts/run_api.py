"""Start the DeepREAP FastAPI service.

Run with:
    python -m scripts.run_api  -- or --
    uvicorn src.integration.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.environ.get("DEEPREAP_HOST", "0.0.0.0")
    port = int(os.environ.get("DEEPREAP_PORT", "8000"))
    uvicorn.run(
        "src.integration.api:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
