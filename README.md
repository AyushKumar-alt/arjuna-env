---
title: Arjuna Perception Env
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# ARJUNA Perception Environment

**Name:** Arjuna Perception Environment (`arjuna-perception-env` in OpenEnv metadata)  
**One-liner:** A simulated robot **vision / perception** environment where an agent reads YOLO-style scene descriptions and is scored on identification, triage, and low-confidence decisions—built on **OpenEnv**.

## What does this environment do?

** ARJUNA is an autonomous robot whose “eyes” are simulated here. Each **episode** is a **3-step sequence** (identify → triage → decide) over one **themed bundle** of scenes (see `EPISODE_BUNDLES` in `server/synthetic_data.py`). The agent receives **natural-language observations** with fake detections and must emit a structured **action** per step. The **grader** returns a **per-step reward in \[0, 1\]**, and after step 3 an **`overall_reward`** (mean of the three steps) plus **feedback**—without real cameras, cloud databases, or (for the env itself) any external API.

---

## Quick Start — No API Key Required

The environment runs **fully offline**. 
No API key needed to run the environment itself.

### Run with Docker (recommended)
```bash
docker build -t arjuna-env .
docker run -p 7860:7860 arjuna-env
```

### Run locally
```bash
pip install -r requirements.txt
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Test offline with demo.py
```bash
# No API key needed — uses heuristic policy
python demo.py
```

---

## Core vs Optional Features

### Core (offline, no API key needed)
| Feature | File | Description |
|---|---|---|
| Environment server | server/app.py | HTTP endpoints |
| Episode logic | server/arjuna_environment.py | reset/step/state |
| Task graders | server/tasks.py, server/grader.py | reward logic |
| Synthetic scenes | server/synthetic_data.py | 8 episode bundles |
| Data models | models.py | typed actions/observations |
| Offline demo | demo.py | heuristic agent, no LLM |

### Optional (requires API key + network)
| Feature | File | How to enable |
|---|---|---|
| LLM baseline agent | inference.py | Set API_BASE_URL + HF_TOKEN |
| Dynamic scene generation | server/scene_generator.py | ENABLE_DYNAMIC_SCENES=true |
| Auto-curriculum | server/curriculum.py | ENABLE_DYNAMIC_SCENES=true |
| AutoRL loop | autorl.py | Set API_BASE_URL + HF_TOKEN |

The environment **always falls back to synthetic_data.py** 
if dynamic scene generation is disabled or fails.

---

## Offline Execution Guarantee

All of these work with **zero network calls**:
- `POST /reset` → picks from synthetic_data.py bundles
- `POST /step` → grades using local tasks.py logic
- `GET /state` → returns local session state
- `GET /health` → returns healthy
- `GET /schema` → returns typed schemas
- `GET /metadata` → returns environment info
- `python demo.py` → full 3-step episode, heuristic policy

---

## Table of contents

1. [Hackathon / Round 1: quick answers](#hackathon--round-1-quick-answers)  
2. [Why 3-step episodes?](#why-3-step-episodes)  
3. [**AutoRL approach — how it all fits together**](#autorl-approach--how-it-all-fits-together)  
4. [Dynamic Scene Generation (Level 1)](#dynamic-scene-generation-level-1)  
5. [Auto-Curriculum Learning (Level 2)](#auto-curriculum-learning-level-2)  
6. [Environment overview (observations, actions, tasks)](#environment-overview-observations-actions-tasks)  
7. [Prerequisites](#prerequisites)  
8. [Setup: `requirements.txt` and venv](#setup-requirementstxt-and-venv)  
9. [Run with Docker](#run-with-docker)  
10. [Run locally without Docker (uvicorn)](#run-locally-without-docker-uvicorn)  
11. [Run the demo (offline)](#run-the-demo-offline)  
12. [Gradio Playground (`/web`)](#gradio-playground-web)  
13. [How grading works](#how-grading-works)  
14. [OpenEnv compliance and key files](#openenv-compliance-and-key-files)  
15. [Project structure](#project-structure)  
16. [Example interaction (reset → three steps)](#example-interaction-reset--three-steps)  
17. [Testing and validation](#testing-and-validation)  
18. [Offline execution](#offline-execution)  
19. [Optional: LLM baseline (`inference.py`)](#optional-llm-baseline-inferencepy)  
20. [Design notes](#design-notes)  
21. [Troubleshooting](#troubleshooting)  
22. [FAQ](#faq)  
23. [Future improvements](#future-improvements)  
24. [Visuals & architecture](#visuals--architecture)  
25. [Credits and acknowledgements](#credits-and-acknowledgements)  
26. [License](#license)  
27. [Maintainer / contact](#maintainer--contact)  

---

## Why 3-step episodes?

A **single-step** environment gives RL agents one reward signal per reset — limiting the training signal and making it impossible to model sequential decision-making. ARJUNA solves this with **3-step episodes**:

| Benefit | Detail |
|---------|--------|
| **Denser reward signal** | Agents receive a reward after **every step** (not just at episode end), enabling faster credit assignment and learning. |
| **Sequential difficulty** | Steps escalate: easy identification → ordered triage → ambiguous low-confidence call. Agents must adapt within the same episode. |
| **Thematic coherence** | All 3 steps draw scenes from the same **location bundle** (e.g. "Warehouse"), so context carries across steps — closer to real-world perception pipelines. |
| **Overall episode signal** | `overall_reward` = mean of 3 step rewards gives a clean episode-level metric for leaderboard comparison. |

The 8 themed bundles (Urban Street, Warehouse, Parking Lot, School Zone, Airport, Hospital Entrance, Construction Site, Night Street) ensure diverse training distributions across resets.

---

## AutoRL approach — how it all fits together

ARJUNA implements a **closed-loop, self-improving training environment** inspired by Automatic Reinforcement Learning (AutoRL) principles. The two subsystems — **Dynamic Scene Generation** and **Auto-Curriculum** — work together in a feedback loop:

```mermaid
flowchart TD
  subgraph AutoRL Loop
    A["Agent submits actions"] --> B["Grader scores episode"]
    B --> C["Auto-Curriculum records reward"]
    C --> D{"Mean reward vs thresholds"}
    D -- "> 0.85" --> E["PROMOTE difficulty"]
    D -- "< 0.60" --> F["DEMOTE difficulty"]
    D -- "0.60-0.85" --> G["STAY at current level"]
    E --> H["Scene Generator uses new difficulty"]
    F --> H
    G --> H
    H --> I["LLM generates fresh scene at difficulty tier"]
    I --> J["Agent receives new observation"]
    J --> A
  end
```

### Key design principles

| Principle | Implementation |
|-----------|----------------|
| **No memorization** | The LLM generates a **unique scene every `reset()`** — the agent can never memorize fixed scenarios |
| **Adaptive difficulty** | A sliding-window curriculum automatically **promotes/demotes** difficulty based on recent performance |
| **Graceful degradation** | If the LLM is unavailable, the environment **falls back** to 8 hardcoded episode bundles — it always works offline |
| **Stateless scalability** | The `episode_id` + `SESSIONS` pattern lets the autoRL loop run across **stateless HTTP workers** (e.g., HF Spaces) |
| **Environment variables** | `ENABLE_DYNAMIC_SCENES`, `API_BASE_URL`, `HF_TOKEN` toggle the full autoRL loop on/off without code changes |

### Files implementing autoRL

| File | Role in AutoRL |
|------|----------------|
| `server/scene_generator.py` | LLM-powered scene generation with difficulty-aware prompts (easy/medium/hard) |
| `server/curriculum.py` | `AutoCurriculum` class: sliding-window reward tracker with promote/demote logic |
| `server/arjuna_environment.py` | Orchestrator: calls `generate_episode_bundle()` on `reset()`, calls `record_episode()` after final `step()` |
| `server/app.py` | Exposes `GET /curriculum` endpoint for real-time monitoring |

---

## Environment overview (observations, actions, tasks)

### What the agent sees

After **`reset`**, **`ArjunaObservation`** includes:

- **`task_type`**: `1` for step 1 (then `2`, then `3` after each graded step)
- **`step_number`**: `1`, `2`, or `3` — which step you are on
- **`bundle_name`**: human-readable theme (e.g. “Urban Street”) shared across the episode
- **`scene_id`**: id for the **current** task’s scene (e.g. `t1_bnd_urban`)
- **`observation_text`**: instructions + scene description + simulated YOLO lines
- **`episode_id`**: must be sent back on **stateless HTTP** `POST /step` (see below)
- After each **`step`**: **`reward`** for that step, **`done`** (false until step 3), **`feedback`**
- After the **third** **`step`**: **`overall_reward`** (mean of the three step rewards), **`done: true`**

Definitions live in **`models.py`**.

### What actions it can take

Single Pydantic model **`ArjunaAction`** — set fields that match the active **`task_type`**:

| Field | Task | Role |
|--------|------|------|
| **`task1_label`** | 1 | Predicted class label (string) |
| **`ranked_objects`** | 2 | Ordered list of labels (most important first) |
| **`decision`** | 3 | One of `discard`, `request_rescan`, `log_and_continue` |
| **`reasoning`** | 3 | Optional; affects partial credit on task 3 |

### The three tasks (summary)

- **Task 1 — Single-object identification:** One primary detection; agent fills **`task1_label`**. Grading uses **exact match** plus **semantic partial credit** (same broad category, etc.). See [How grading works](#how-grading-works).
- **Task 2 — Multi-object triage:** Several detections; agent fills **`ranked_objects`**. Score is **fraction of positions matching** the ground-truth order (confidence first; tie-break: person > vehicle > other).
- **Task 3 — Low-confidence decision:** One low-confidence primary detection; agent fills **`decision`** and optional **`reasoning`**. Bands: **&lt; 0.35 → discard**, **0.35–&lt;0.50 → request_rescan**, **≥ 0.50 → log_and_continue**. Grading uses **adjacent-band** partial credit and **reasoning quality** (numeric confidence in text → “strong” reasoning).

---

## Prerequisites

| Requirement | Notes |
|-------------|--------|
| **Python** | **3.11+** (matches `Dockerfile`; 3.12+ usually fine locally) |
| **pip** | For `requirements.txt` |
| **Docker** | **Docker Desktop** (or compatible engine) to build/run the image—optional if you use uvicorn only |
| **Git** | To clone / push the repo |

No cloud database, no Redis, no external service is required to **run the environment** itself.

---

## Setup: `requirements.txt` and venv

From the repository root:

```bash
python -m venv .venv
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux:**

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins **OpenEnv** (`openenv-core`), **FastAPI**, **uvicorn**, **pydantic**, and **openai** (used only by the optional **`inference.py`** baseline).

---

## Run with Docker

From the repository root:

```bash
docker build -t arjuna-env .
docker run -p 7860:7860 arjuna-env
```

Defaults (the **`Dockerfile`** sets **`ENABLE_WEB_INTERFACE=true`**):

- **Playground:** `http://127.0.0.1:7860/web`  
- **Swagger:** `http://127.0.0.1:7860/docs`  
- **Health:** `http://127.0.0.1:7860/health`

To turn the Playground **off** in a local run: **`docker run -p 7860:7860 -e ENABLE_WEB_INTERFACE=false arjuna-env`**

---

## Run locally without Docker (uvicorn)

```bash
# optional: Gradio UI at /web
export ENABLE_WEB_INTERFACE=true   # Windows PowerShell: $env:ENABLE_WEB_INTERFACE="true"

python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Use another port if **7860** is busy, e.g. `--port 8000`.

---

## Run the demo (offline)

The **demo does not call Hugging Face or any LLM API**. It uses a small heuristic policy and talks to your running server over HTTP.

**Terminal 1 — start the server** (Docker or uvicorn as above). Example with Docker:

```bash
docker build -t arjuna-env .
docker run -p 7860:7860 arjuna-env
```

**Terminal 2 — from repo root, with dependencies installed:**

```bash
# default base URL is http://127.0.0.1:7860
export ARJUNA_ENV_BASE_URL=http://127.0.0.1:7860   # PowerShell: $env:ARJUNA_ENV_BASE_URL="http://127.0.0.1:7860"
python demo.py
```

You will see one **full 3-step episode** (bundle name, per-step rewards, overall reward). **`demo.py`** is the canonical **offline-friendly** “try the env” entrypoint for reviewers (no LLM API key).

---

## Gradio Playground (`/web`)

When **`ENABLE_WEB_INTERFACE=true`**, OpenEnv mounts a browser UI at **`/web`**.

This repo applies **`server/openenv_web_patch.py`** so the Playground **Step** passes **`episode_id`** into **`env.step`**, consistent with HTTP **`POST /step`** (important on stateless workers and HF Spaces).

**Usage:**

1. Click **Reset** — note **`step_number`**, **`task_type`**, and **`bundle_name`**.  
2. Fill the fields for **that** step only, then click **Step** ( **`done`** is **false** after steps 1 and 2 ).  
3. Repeat until **`task_type`** is **3** and you’ve submitted the third action — then **`done`** becomes **true** and **`overall_reward`** appears.

**Task 2 input:** `ranked_objects` can be a **JSON array string** (e.g. `["person","car"]`) or **comma-separated** labels; **`models.py`** coerces strings into `list[str]` for the Playground.

---

## Dynamic Scene Generation (Level 1)

Instead of fixed hardcoded scenes, the environment generates **infinite unique episodes** using an LLM. This is the core of the autoRL approach — the environment itself is **non-stationary**, forcing the agent to generalize rather than memorize.

### How it works

1. On every `reset()`, `arjuna_environment.py` calls `generate_episode_bundle()` from `scene_generator.py`
2. The generator sends **difficulty-specific prompts** to the LLM (via HF Router / OpenAI-compatible API)
3. Each prompt template enforces structural constraints (COCO classes, confidence ranges, JSON schema)
4. Generated JSON is **validated** (schema checks + band-rule consistency for Task 3)
5. Valid scenes are converted into `EpisodeBundle` dataclasses — **identical interface** to hardcoded scenes
6. On failure (no creds, quota, malformed response), the system **silently falls back** to `synthetic_data.py`

### Difficulty-aware generation

| Difficulty | Task 1 (Confidence) | Task 2 (Objects) | Task 3 (Bands) |
|-----------|--------------------|--------------------|----------------|
| `easy` | 0.85–0.98 (clear) | 3 objects, no ties | Deep in one band |
| `medium` | 0.72–0.84 (partial occlusion) | 4 objects, 1 tie | Near boundary |
| `hard` | 0.60–0.71 (fog/night/blur) | 5 objects, multiple ties | Within 0.005 of boundary |

```python
# Environment generates a fresh scene on every reset
obs = await env.reset(seed=42)
# obs.observation_text contains a brand new LLM-generated scene
# difficulty is driven by the auto-curriculum (see Level 2)
```

**Environment variables required:** `ENABLE_DYNAMIC_SCENES=true`, `API_BASE_URL`, `HF_TOKEN`

---

## Auto-Curriculum Learning (Level 2)

The environment **automatically adjusts difficulty** based on the agent's recent performance, completing the autoRL feedback loop. The `AutoCurriculum` class in `server/curriculum.py` uses a **sliding window** of the last 5 episode rewards:

```
Agent mean reward ≥ 0.85 → PROMOTE to harder difficulty  (easy→medium→hard)
Agent mean reward < 0.60 → DEMOTE  to easier difficulty  (hard→medium→easy)
Otherwise                → STAY    at current difficulty
```

### Curriculum internals

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `WINDOW_SIZE` | 5 | Number of recent episodes to average |
| `MIN_EPISODES` | 3 | Minimum data before any adjustment |
| `PROMOTE_THRESHOLD` | 0.85 | Mean reward above which difficulty increases |
| `DEMOTE_THRESHOLD` | 0.60 | Mean reward below which difficulty decreases |

After a promotion or demotion, the **window is cleared** to give the agent a fresh start at the new difficulty level.

### Monitoring

Check current curriculum status via the `/curriculum` endpoint:
```bash
curl http://127.0.0.1:7860/curriculum
```
Example response:
```json
{
  "current_difficulty": "medium",
  "total_episodes": 12,
  "recent_mean": 0.883,
  "promotions": 1,
  "demotions": 0,
  "promote_threshold": 0.85,
  "demote_threshold": 0.60,
  "window": [0.90, 0.88, 0.85, 0.92]
}
```

The curriculum persists across episodes within a server process but **resets on restart** (stateless container recovery).

---

## How grading works

Plain-language summary (details in **`server/tasks.py`**; discoverable re-exports in **`server/grader.py`**):

- **Task 1:** Compares normalized **`task1_label`** to the scene’s expected label. **1.0** exact match; **0.7** same semantic **category group** (vehicles / people / animals); **0.2** agent label is a known category but wrong group; **0.0** otherwise.  
- **Task 2:** Compares your ordering to **`expected_priority`**. **All correct → 1.0**; **n−1 positions → 0.85**; **n−2 → 0.65**; **exactly one → 0.33**; **none → 0.0** (length must match).  
- **Task 3:** Maps detector confidence to gold action (`discard` / `request_rescan` / `log_and_continue`). **Correct + strong reasoning** (text mentions a numeric confidence) → **1.0**; **correct + weak** → **0.85**; **correct + no reasoning** → **0.7**; **one band off** with strong/weak reasoning → **0.5** / **0.3**; **two bands off or invalid** → **0.0**.

Feedback strings after **`step`** are produced in **`server/arjuna_environment.py`** from these scores (including **“Step k/3 complete…”** banners).

---

## OpenEnv compliance and key files

This project targets **OpenEnv** conventions:

- **HTTP:** `POST /reset`, `POST /step`, state routes as provided by **`openenv.core.env_server.create_app`**.  
- **Models:** **`ArjunaAction`**, **`ArjunaObservation`**, **`ArjunaState`** extend OpenEnv base types in **`models.py`**.  
- **Environment:** **`server/arjuna_environment.py`** — implements **`reset` / `step` / `state` / `get_metadata`**, Gymnasium-style episode flow, **`EnvironmentMetadata`** for docs.

| Concern | Primary file(s) |
|--------|------------------|
| Episodes, sessions, orchestration | `server/arjuna_environment.py` (`SESSIONS` + `episode_id` for HTTP; **3 steps** per episode) |
| Rubric / reward logic | `server/tasks.py`, **`server/grader.py`** (re-exports) |
| Scenes and gold labels | `server/synthetic_data.py` (includes **`EPISODE_BUNDLES`** for themed 3-step episodes) |
| FastAPI / OpenEnv app | `server/app.py`, **`server/openenv_web_patch.py`** (Gradio `episode_id`) |
| CLI / Space config | `openenv.yaml`, `Dockerfile` |

---

## Project structure

```
arjuna_env/
├── README.md                 # This file (HF Space frontmatter at top)
├── LICENSE                   # MIT
├── openenv.yaml              # OpenEnv / Space metadata
├── requirements.txt
├── Dockerfile
├── docs/
│   ├── images/               # See docs/images/README.md (no image binaries in repo)
│   └── PUSH_TO_HF_SPACE.md   # How to push to HF without binary-blob history errors
├── demo.py                   # Offline-friendly heuristic demo (no LLM)
├── inference.py              # Optional LLM baseline (uses HF router + API key)
├── client.py                 # Thin HTTP client helper
├── models.py                 # ArjunaAction, ArjunaObservation, ArjunaState
├── __init__.py
└── server/
    ├── app.py                # OpenEnv FastAPI app + /curriculum endpoint
    ├── arjuna_environment.py # Core Environment: reset / step / state + autoRL orchestration
    ├── scene_generator.py    # ★ Level 1: LLM-powered dynamic scene generation
    ├── curriculum.py         # ★ Level 2: AutoCurriculum sliding-window tracker
    ├── tasks.py              # Grading + prompt formatting
    ├── grader.py             # Explicit re-exports of grading functions
    ├── synthetic_data.py     # Fallback scenes (100% local, 8 bundles)
    ├── openenv_web_patch.py  # Gradio: pass episode_id into step
    └── __init__.py
```

---

## Example interaction (reset → three steps)

This shows a complete episode using the **"Urban Street"** bundle (seed 0).

**1. Reset** — start a new episode:

```bash
curl -X POST http://127.0.0.1:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"seed": 0}'
```

Response: `task_type: 1`, `step_number: 1`, `bundle_name: "Urban Street"`, `scene_id`, `episode_id`, `observation_text` (YOLO-style scene), `reward: 0.0`, `done: false`.

**2. Step 1 — single-object identification** (`task_type: 1`):

```json
{
  "episode_id": "<from reset response>",
  "action": { "task1_label": "person" }
}
```

Response: `reward` for step 1 (e.g. `1.0`), `done: false`, `task_type: 2`, new `observation_text` for triage, `step_number: 2`, `feedback`: `"Step 1/3 complete. Reward: 1.000. Move to next step."`

**3. Step 2 — multi-object triage** (`task_type: 2`):

```json
{
  "episode_id": "<same episode_id>",
  "action": { "ranked_objects": ["person", "car", "bicycle"] }
}
```

Response: `reward` for step 2 (e.g. `1.0`), `done: false`, `task_type: 3`, new `observation_text` for low-confidence decision, `step_number: 3`, `feedback`: `"Step 2/3 complete. Reward: 1.000. Move to next step."`

**4. Step 3 — low-confidence decision** (`task_type: 3`):

```json
{
  "episode_id": "<same episode_id>",
  "action": {
    "decision": "discard",
    "reasoning": "confidence 0.31 is below the 0.35 threshold, object identity unclear."
  }
}
```

Final response: `done: true`, `overall_reward: 1.0` (mean of all 3 step rewards), `step_number: 3`, `feedback`: `"Step 3/3 complete. Reward: 1.000. Episode done. Overall: 1.000"`. See Swagger **`/docs`** and [Legacy cURL](#legacy-quick-reference-curl) below.

---

## Testing and validation

**OpenEnv CLI validate** (network required; checks deployed Space):

```bash
python -m openenv.cli validate https://calpol500mg-arjuna-env.hf.space
```

**Manual smoke test (local or Space):**

1. `POST /reset` with `{"seed": 42}`  
2. Copy `observation.episode_id`  
3. `POST /step` three times with `episode_id` + actions for **`task_type` 1, then 2, then 3**  
4. Expect `done: false` after steps 1–2, then **`done: true`** and **`overall_reward`** after step 3; `feedback` non-empty each time

**Playground:** `Reset` → **Step** three times (adjust fields as `task_type` changes); expect graded JSON each time.

*(Automated `pytest` suite is not bundled; add `tests/` if you want CI-style checks.)*

---

## Offline execution

- **Environment, scenes, and grader:** run entirely **on your machine** (or in Docker). Data is **local** in **`server/synthetic_data.py`**. **No cloud database** and **no external HTTP** are required for **`reset` / `step` / `demo.py` / Playground**.  
- **`inference.py`** is **optional**: it calls a **remote LLM** via Hugging Face Inference / OpenAI-compatible router and is **not** needed to verify the environment or grading.

---

## Optional: LLM baseline (`inference.py`)

Uses **network + API key** (HF token). Example (**PowerShell**):

```powershell
$env:API_BASE_URL="https://router.huggingface.co/v1"
$env:MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
$env:HF_TOKEN="your_hf_token_here"
$env:ARJUNA_ENV_BASE_URL="http://127.0.0.1:7860"
# Optional quota savers:
$env:N_SEEDS="3"
$env:MAX_TOKENS="80"
$env:ENABLE_RETRY="0"

python inference.py
```

---

## Design notes

- **AutoRL feedback loop:** The scene generator and curriculum form a closed loop — the agent's performance directly influences the difficulty of future scenes. This eliminates manual curriculum tuning.  
- **Difficulty-aware prompting:** `scene_generator.py` uses **3 distinct prompt templates per task** (easy/medium/hard), controlling confidence ranges, object counts, and ambiguity levels at the prompt level.  
- **Graceful fallback chain:** `LLM → validate → convert` or fall back to `synthetic_data.py`. The environment **never errors** due to LLM unavailability.  
- **Stateless HTTP:** `episode_id` + module-level **`SESSIONS`** keeps **multi-step** episode state across requests (critical on HF Spaces).  
- **Single action schema:** one **`ArjunaAction`** for all tasks; agents pick fields based on **`task_type`** on each observation.  
- **Synthetic diversity:** **`EPISODE_BUNDLES`** tie three scenes to one theme; standalone **`TASK*_SCENES`** remain as fallback variety.  
- **Sliding window curriculum:** Window clears on difficulty change, preventing stale data from influencing future promotions/demotions.  
- **Playground UX:** `ranked_objects` string coercion and **`openenv_web_patch`** reduce friction for reviewers using **`/web`**.

---

## Troubleshooting

| Issue | What to try |
|--------|-------------|
| `Call reset() before step()` | Call **`/reset`** (or **Reset** in UI) before **`step`**; send **`episode_id`** on HTTP; run **three** graded **Step** calls per episode (or **`Call reset()`** if the episode id expired). |
| `only one usage of each socket address` | Another process uses the port; pick another `--port` or `taskkill` the old listener (see `netstat` / `findstr` on Windows). |
| Docker `npipe` / cannot connect | Start **Docker Desktop** and wait until the engine is running. |
| Playground `ranked_objects` validation | Use JSON array string or comma-separated labels (see [Gradio](#gradio-playground-web)). |
| `/web` returns **404** | Start with **`ENABLE_WEB_INTERFACE=true`** **before** importing/running the app; rebuild/restart container if needed. |
| HF `402` on `inference.py` | Inference quota exhausted; use smaller **`N_SEEDS`** / **`MAX_TOKENS`** or run **`demo.py`** offline. |
| **`git push space` rejected (binary files)** | History may still contain binary blobs; use a **snapshot push** (see **`docs/PUSH_TO_HF_SPACE.md`**) or **[Hub Xet](https://huggingface.co/docs/hub/xet)**. |
| **`sdk` must be one of [gradio, docker, …]** | README frontmatter must use **`sdk: docker`** (lowercase), with keys like **`title`**, **`colorFrom`**, not `Title` / `Sdk: Docker`. |

---

## FAQ

**Q: Does the Space need my API key?**  
A: The **environment** does not. Only **`inference.py`** (external LLM) needs a token.

**Q: Where is the “source of truth” for correct answers?**  
A: **`server/synthetic_data.py`** (expected labels, orderings, task 3 bands).

**Q: Why is `episode_id` required on HTTP step?**  
A: Workers may be stateless; **`SESSIONS[episode_id]`** ties **reset** and **step** together.

**Q: Is WebSocket supported?**  
A: Yes via OpenEnv’s stack; **`inference.py`** can pass metadata consistently. HTTP remains the primary “curl-friendly” path.

---

## Future improvements

- Add a **`tests/`** package with parametrized grading and HTTP contract tests.  
- Optional **`gradio_builder`** “Custom” tab with task-aware field visibility.

---

## Visuals & architecture

**Try the live UI** (no installation required):

| What | Live URL |
|------|----------|
| **Gradio Playground** | [calpol500mg-arjuna-env.hf.space/web](https://calpol500mg-arjuna-env.hf.space/web) |
| **Swagger / OpenAPI** | [calpol500mg-arjuna-env.hf.space/docs](https://calpol500mg-arjuna-env.hf.space/docs) |
| **Curriculum Status** | [calpol500mg-arjuna-env.hf.space/curriculum](https://calpol500mg-arjuna-env.hf.space/curriculum) |

**Architecture (with autoRL loop):**

```mermaid
flowchart LR
  subgraph client [Clients]
    Web["Gradio /web"]
    HTTP["HTTP /reset /step"]
    Demo["demo.py"]
    Inf["inference.py optional"]
  end
  subgraph server [Server — AutoRL Core]
    App["server.app FastAPI"]
    Env["ArjunaEnvironment"]
    SceneGen["scene_generator.py — LLM Scenes"]
    Curriculum["curriculum.py — Auto-Curriculum"]
    Data["synthetic_data.py — Fallback"]
    Grade["tasks.py / grader.py"]
  end
  subgraph llm [External LLM]
    HF["HF Router / OpenAI API"]
  end
  Web --> App
  HTTP --> App
  Demo --> App
  Inf --> App
  App --> Env
  Env -- "reset()" --> SceneGen
  SceneGen -- "difficulty" --> Curriculum
  SceneGen --> HF
  SceneGen -- "fallback" --> Data
  Env -- "step() grade" --> Grade
  Grade -- "episode reward" --> Curriculum
  Curriculum -- "promote/demote" --> SceneGen
```

---

## Credits and acknowledgements

- **[OpenEnv](https://github.com/meta-pytorch/OpenEnv)** — Meta & Hugging Face, HTTP/WebSocket environment interface.  
- **FastAPI**, **Pydantic**, **uvicorn**, **Gradio** (via OpenEnv UI).  
- Synthetic scenes and grading authored for this **ARJUNA Perception** hackathon project.

---

## License

This project is released under the **MIT License** — see the [`LICENSE`](LICENSE) file in the repository root.

---

## Maintainer / contact

- **Author:** Ayush Kumar  
- **HF Space:** [Calpol500mg/arjuna-env](https://huggingface.co/spaces/Calpol500mg/arjuna-env)  
- **Live app:** [calpol500mg-arjuna-env.hf.space](https://calpol500mg-arjuna-env.hf.space)

For reviewer questions, use the Space **Community** tab or GitHub **Issues**: [github.com/AyushKumar-alt/arjuna-env/issues](https://github.com/AyushKumar-alt/arjuna-env/issues).

---

## Baseline results (historical, LLM-dependent)

**`inference.py`** runs **full 3-step episodes** per seed and reports **per-task mean rewards** plus **overall mean reward**. Output format:

```
=== Episode 1 (seed=42) ===
  Step 1/3: task1 | scene=t1_bnd_urban | reward=1.000
  Step 2/3: task2 | scene=t2_bnd_urban | reward=0.850
  Step 3/3: task3 | scene=t3_bnd_urban | reward=1.000
  Episode reward: 0.950
---
task 1 mean reward: 0.950
task 2 mean reward: 0.820
task 3 mean reward: 0.867
overall mean reward: 0.879
```

Numbers **vary by model and quota**; typical ranges with a capable LLM:

| Task | Typical range | Note |
|------|--------------|------|
| Task 1 — Identification | 0.90–1.00 | Single label; exact or semantic match |
| Task 2 — Triage | 0.70–0.90 | Sensitive to list ordering accuracy |
| Task 3 — Low-confidence | 0.75–1.00 | Strongly improved by mentioning confidence value in reasoning |
| **Overall episode** | **0.80–0.95** | Mean of all 3 step rewards per episode |

Prefer **`demo.py`** (deterministic heuristic, no API key) or manual **Playground** checks for **repeatable** smoke tests.

---

## Legacy quick reference (cURL)

**Reset**

```bash
curl -X POST https://calpol500mg-arjuna-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d "{\"seed\": 42}"
```

**Step — Task 1** (paste `episode_id` from reset)

```bash
curl -X POST https://calpol500mg-arjuna-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d "{\"episode_id\": \"<paste-from-reset>\", \"action\": {\"task1_label\": \"person\"}}"
```

**Step — Task 2**

```bash
curl -X POST https://calpol500mg-arjuna-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d "{\"episode_id\": \"<paste-from-reset>\", \"action\": {\"ranked_objects\": [\"person\", \"car\", \"bicycle\"]}}"
```

**Step — Task 3**

```bash
curl -X POST https://calpol500mg-arjuna-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d "{\"episode_id\": \"<paste-from-reset>\", \"action\": {\"decision\": \"discard\", \"reasoning\": \"confidence below 0.35, unsafe to log\"}}"
```

On Spaces, **always** pass the **same** **`episode_id`** on **each** **`POST /step`** until **`done`** is **true** (three steps per episode) for stateless HTTP workers.
