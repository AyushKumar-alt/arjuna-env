"""
FastAPI app for ARJUNA perception OpenEnv (HTTP + WebSocket).

Run locally:
  uvicorn server.app:app --host 0.0.0.0 --port 7860

Or:
  openenv serve .
"""

from __future__ import annotations

from typing import Any

import os

from fastapi import Request
from fastapi.openapi.utils import get_openapi
from fastapi.responses import RedirectResponse
from openenv.core.env_server import create_app

try:
    from models import ArjunaAction, ArjunaObservation
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from models import ArjunaAction, ArjunaObservation
from server.arjuna_environment import ArjunaEnvironment
from server.openenv_web_patch import apply as _apply_openenv_web_patch

if os.getenv("ENABLE_WEB_INTERFACE", "false").lower() in ("true", "1", "yes"):
    _apply_openenv_web_patch()

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
def root(request: Request) -> Any:
    # HF Space preview and plain browser loads should show the Swagger UI.
    # Keep the JSON summary available for API clients via `/?format=json`
    # (or any explicit JSON Accept header).
    accept = request.headers.get("accept", "").lower()
    want_json = (
        request.query_params.get("format") == "json"
        or request.query_params.get("json") == "1"
        or "application/json" in accept
        or "text/json" in accept
    )
    if not want_json:
        return RedirectResponse(url="/docs")

    return {
        "name": "arjuna-perception-env",
        "description": (
            "Simulated robot perception environment for ARJUNA built on OpenEnv; "
            "use /reset and /step or the WebSocket client for episodes."
        ),
        "tasks": 3,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/curriculum", tags=["curriculum"])
async def get_curriculum_endpoint():
    """
    Auto-curriculum status (only active when 
    ENABLE_DYNAMIC_SCENES=true).
    """
    dynamic_enabled = os.environ.get(
        "ENABLE_DYNAMIC_SCENES", "false"
    ).lower() == "true"
    
    if not dynamic_enabled:
        return {
            "enabled": False,
            "message": (
                "Auto-curriculum is disabled. "
                "Set ENABLE_DYNAMIC_SCENES=true to enable."
            ),
            "current_difficulty": "medium",
            "note": (
                "Environment runs fully offline "
                "using synthetic_data.py scenes."
            )
        }
    
    try:
        from server.curriculum import get_curriculum_stats
        stats = get_curriculum_stats()
        return {
            "enabled": True,
            "current_difficulty": stats["current_difficulty"],
            "override": stats.get("override"),
            "recent_mean_reward": stats["recent_mean"],
            "total_episodes": stats["total_episodes"],
            "promotions": stats["promotions"],
            "demotions": stats["demotions"],
            "window_rewards": stats["window"],
            "thresholds": {
                "promote_above": stats["promote_threshold"],
                "demote_below": stats["demote_threshold"]
            }
        }
    except Exception as e:
        return {
            "enabled": True,
            "error": str(e),
            "message": "Curriculum module error"
        }

@app.post("/curriculum/difficulty", tags=["curriculum"])
async def set_difficulty_endpoint(difficulty: str | None = None):
    """
    Manually override the curriculum difficulty tier.
    difficulty: 'easy', 'medium', 'hard', or null (to return to auto).
    """
    try:
        from server.curriculum import set_curriculum_override
        # Handle "auto" or empty string as None
        if difficulty in ("auto", ""):
            difficulty = None
        set_curriculum_override(difficulty)
        return {"status": "success", "difficulty": difficulty or "auto"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
