---
title: Arjuna Perception Env
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# ARJUNA Perception Environment

## Overview

This repository implements a **simulated robot perception** reinforcement-learning environment for the OpenEnv framework. An AI agent acts as the **decision brain** of **ARJUNA**, an autonomous robot. The agent receives **simulated camera scene descriptions** (YOLO-style detections in natural language) and must produce the correct action for the active task. The environment scores responses with rewards in **[0.0, 1.0]**.

Built with **OpenEnv** (Meta & Hugging Face): standardized `reset` / `step` / `state` over HTTP and WebSocket clients.

## Environment Details

- **Framework:** OpenEnv (`openenv-core`)
- **API:** `reset()` / `step()` / `state()` (HTTP + WebSocket; for HTTP, pass **`episode_id`** from **`reset`** into **`step`**)
- **Deployed on:** Hugging Face Spaces — [Calpol500mg/arjuna-env](https://huggingface.co/spaces/Calpol500mg/arjuna-env)
- **Docker:** Yes — self-contained image via `Dockerfile`

## Action Space

Unified Pydantic model **`ArjunaAction`** (set fields matching `task_type` on the observation):

- **`task1_label`** (`str | null`): Task 1 — predicted YOLO class for the single visible object
- **`ranked_objects`** (`list[str] | null`): Task 2 — ordered list of class labels, most important first
- **`decision`** (`str | null`): Task 3 — one of: `log_and_continue` | `discard` | `request_rescan`
- **`reasoning`** (`str | null`): Task 3 optional explanation (used for partial credit)

## Observation Space

**`ArjunaObservation`** fields:

- **`episode_id`** (`str | null`): Set on **`reset`**; required on **`POST /step`** over plain HTTP (see below) so the server can load the same scene across requests
- **`task_type`** (`int`): Which task is active (`1`, `2`, or `3`)
- **`scene_id`** (`str`): Synthetic scene identifier
- **`observation_text`** (`str`): Natural-language scene + YOLO detections for the agent
- **`feedback`** (`str`): Grader message after `step`
- **`reward`** (`float`): Last reward in `[0, 1]`
- **`done`** (`bool`): `true` when the episode ends

## The 3 Tasks

### Task 1 — Single Object Identification (Easy)

- Scene has **one** clearly visible object with **confidence > 0.85**
- Agent must identify the object type via **`task1_label`**
- **Reward:** `1.0` if correct, `0.0` if wrong

### Task 2 — Multi-Object Triage (Medium)

- Scene has **3–5** objects at varying confidences
- Agent must return a prioritized list via **`ranked_objects`** (most important first)
- **Priority rules:** higher confidence first; ties: **person > vehicle > others**
- **Reward:** (number of correctly ranked positions) / total objects, range **0.0–1.0**

### Task 3 — Low-Confidence Decision (Hard)

- YOLO returns **low-confidence** primary detection (**0.25–0.55**)
- Agent chooses via **`decision`** (optional **`reasoning`**):
  - confidence **< 0.35** → correct: **`discard`** → reward **1.0**
  - **0.35 ≤** confidence **< 0.50** → correct: **`request_rescan`** → reward **1.0**
  - confidence **≥ 0.50** → correct: **`log_and_continue`** → reward **1.0**
  - wrong decision → reward **0.0**
  - wrong decision but strong uncertainty reasoning → reward **0.5** (partial credit)

## Setup & Installation

### Requirements

- Python **3.11+**
- **Docker Desktop** (for local container runs)
- `pip install -r requirements.txt` (includes `openenv-core`)

### Run Locally

```bash
docker build -t arjuna-env .
docker run -p 7860:7860 arjuna-env
```

Then open `http://127.0.0.1:7860/docs` or `http://127.0.0.1:7860/health`.

### Run Inference Script

PowerShell (Windows):

```powershell
$env:API_BASE_URL="https://router.huggingface.co/v1"
$env:MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
$env:HF_TOKEN="your_hf_token_here"
$env:ARJUNA_ENV_BASE_URL="http://127.0.0.1:7860"

python inference.py
```

Use a Hugging Face token with permission to call **Inference Providers** via the router.

### Validate Deployed Space

```bash
python -m openenv.cli validate https://calpol500mg-arjuna-env.hf.space
```

## Example API Usage

On **Hugging Face Spaces** (and any deployment where each HTTP request may hit a new process), **`POST /reset`** and **`POST /step`** do not share the same in-memory environment instance. After **`reset`**, read **`observation.episode_id`** from the response and send that same value on **`step`** as a top-level JSON field alongside **`action`**. Omitting it produces “Call reset() before step().” The **WebSocket** client keeps one environment per connection, so it can omit **`episode_id`** on **`step`**; **`inference.py`** still passes **`episode_id`** in **`action.metadata`** for consistency.

### Reset

```bash
curl -X POST https://calpol500mg-arjuna-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d "{\"seed\": 42}"
```

The JSON response includes **`observation.episode_id`** — use it in the next request.

### Step — Task 1

```bash
curl -X POST https://calpol500mg-arjuna-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d "{\"episode_id\": \"<paste-from-reset-observation>\", \"action\": {\"task1_label\": \"person\"}}"
```

### Step — Task 2

```bash
curl -X POST https://calpol500mg-arjuna-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d "{\"episode_id\": \"<paste-from-reset-observation>\", \"action\": {\"ranked_objects\": [\"person\", \"car\", \"bicycle\"]}}"
```

### Step — Task 3

```bash
curl -X POST https://calpol500mg-arjuna-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d "{\"episode_id\": \"<paste-from-reset-observation>\", \"action\": {\"decision\": \"discard\", \"reasoning\": \"confidence below 0.35, unsafe to log\"}}"
```

## Baseline Results

Baseline LLM agent (`inference.py`), representative run:

- **Task 1:** 1.000  
- **Task 2:** 0.778  
- **Task 3:** 0.667  
- **Overall:** 0.815  

## Project Structure

```
arjuna_env/
├── inference.py
├── Dockerfile
├── requirements.txt
├── openenv.yaml
├── README.md
├── models.py
├── client.py
├── __init__.py
└── server/
    ├── app.py
    ├── arjuna_environment.py
    ├── tasks.py
    ├── synthetic_data.py
    └── __init__.py
```

## Author

- **Author:** Ayush Kumar 
- **HF Space:** [https://huggingface.co/spaces/Calpol500mg/arjuna-env](https://huggingface.co/spaces/Calpol500mg/arjuna-env)  
- **Live app:** [https://calpol500mg-arjuna-env.hf.space](https://calpol500mg-arjuna-env.hf.space/docs)  
- **Framework:** [OpenEnv](https://github.com/meta-pytorch/OpenEnv) by Meta & Hugging Face  
