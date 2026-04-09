import base64
import json
import zlib

def encode_pako(text):
    # This is a basic recreation of what mermaid.ink expects:
    # A base64 encoding of a JSON payload that looks like {"code":"....","mermaid":"..."}
    # Actually, mermaid.ink has an API that accepts base64url(json({"code": text, "mermaid": {"theme": "default"}}))
    payload = json.dumps({
        "code": text,
        "mermaid": {"theme": "default"}
    }).encode("utf-8")
    # Base64url encoding
    encoded = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    return f"https://mermaid.ink/img/{encoded}"

text1 = """flowchart TD
  subgraph AutoRL_Loop [AutoRL Loop]
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
  end"""

text2 = """flowchart LR
  subgraph client [Clients]
    Web["Gradio /web"]
    HTTP["HTTP /reset /step"]
    Demo["demo.py"]
    Inf["inference.py optional"]
  end
  subgraph server [Server - AutoRL Core]
    App["server.app FastAPI"]
    Env["ArjunaEnvironment"]
    SceneGen["scene_generator.py - LLM Scenes"]
    Curriculum["curriculum.py - Auto-Curriculum"]
    Data["synthetic_data.py - Fallback"]
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
  Env -->|"reset()"| SceneGen
  SceneGen -->|"difficulty"| Curriculum
  SceneGen --> HF
  SceneGen -->|"fallback"| Data
  Env -->|"step() grade"| Grade
  Grade -->|"episode reward"| Curriculum
  Curriculum -->|"promote/demote"| SceneGen"""

print("DIAGRAM 1:\\n" + encode_pako(text1) + "\\n")
print("DIAGRAM 2:\\n" + encode_pako(text2) + "\\n")
