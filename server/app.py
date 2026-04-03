"""
FastAPI app for ARJUNA perception OpenEnv (HTTP + WebSocket).

Run locally:
  uvicorn server.app:app --host 0.0.0.0 --port 7860

Or:
  openenv serve .
"""

from __future__ import annotations

from typing import Any

from fastapi.openapi.utils import get_openapi
from fastapi.responses import RedirectResponse
from openenv.core.env_server import create_app

from models import ArjunaAction, ArjunaObservation
from server.arjuna_environment import ArjunaEnvironment

app = create_app(
    ArjunaEnvironment,
    ArjunaAction,
    ArjunaObservation,
    env_name="arjuna_perception_env",
)


def _patch_step_openapi_examples(schema: dict[str, Any]) -> None:
    """Swagger: replace generic StepRequest example with valid ArjunaAction shapes."""
    task1: dict[str, Any] = {
        "episode_id": "<from POST /reset response observation.episode_id>",
        "action": {"task1_label": "person"},
    }
    task2: dict[str, Any] = {
        "episode_id": "<from POST /reset response observation.episode_id>",
        "action": {"ranked_objects": ["person", "car", "bicycle"]},
    }
    task3: dict[str, Any] = {
        "episode_id": "<from POST /reset response observation.episode_id>",
        "action": {
            "decision": "discard",
            "reasoning": (
                "Confidence below 0.35, object identity unclear, unsafe to log."
            ),
        },
    }
    try:
        app_json = schema["paths"]["/step"]["post"]["requestBody"]["content"][
            "application/json"
        ]
    except KeyError:
        return
    app_json["example"] = task1
    app_json["examples"] = {
        "task1_single_object": {
            "summary": "Task 1 — single object identification",
            "value": task1,
        },
        "task2_multi_object_triage": {
            "summary": "Task 2 — prioritized object list",
            "value": task2,
        },
        "task3_low_confidence_policy": {
            "summary": "Task 3 — low-confidence decision (+ optional reasoning)",
            "value": task3,
        },
    }
    comps = schema.get("components", {}).get("schemas", {})
    step_req = comps.get("StepRequest")
    if isinstance(step_req, dict):
        step_req["example"] = task1


def custom_openapi() -> dict[str, Any]:
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        description=app.description,
        routes=app.routes,
    )
    _patch_step_openapi_examples(openapi_schema)
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi  # type: ignore[method-assign]


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
