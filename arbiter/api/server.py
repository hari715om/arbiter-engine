"""CLI entry point to start the Arbiter API server."""

import subprocess
import sys


def serve():
    from arbiter.api.config import settings
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "arbiter.api.app:app",
        "--host", settings.api_host,
        "--port", str(settings.api_port),
        "--reload",
    ])


if __name__ == "__main__":
    serve()
