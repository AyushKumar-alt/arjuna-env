"""
FastAPI app for ARJUNA perception OpenEnv (HTTP + WebSocket).

Run locally:
  uvicorn server.app:app --host 0.0.0.0 --port 7860

Or:
  openenv serve .
"""

from __future__ import annotations

from openenv.core.env_server import create_app

from models import ArjunaAction, ArjunaObservation
from server.arjuna_environment import ArjunaEnvironment

app = create_app(
    ArjunaEnvironment,
    ArjunaAction,
    ArjunaObservation,
    env_name="arjuna_perception_env",
)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
