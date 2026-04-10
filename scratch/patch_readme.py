import re

with open('README.md', 'r', encoding='utf-8') as f:
    text = f.read()

mermaid_chart = """
### Auto-Curriculum Architecture
```mermaid
graph TD
    classDef primary fill:#1ABC9C,stroke:#16A085,stroke-width:2px,color:white,font-weight:bold;
    classDef secondary fill:#34495E,stroke:#2C3E50,stroke-width:2px,color:white;
    classDef logic fill:#F39C12,stroke:#E67E22,stroke-width:2px,color:white,font-weight:bold;

    A[LLM Agent / User Interface] -->|Submits Action| B(ArjunaEnv Step)
    B --> C{Tasks.py Grader}
    C -->|Reward < 0.60| D[Curriculum: Demote]
    C -->|Reward > 0.85| E[Curriculum: Promote]
    D --> F[Synthetic_Data Scene Loader]
    E --> F
    F -->|Yields Next Dynamic Scene| A

    class A secondary;
    class B primary;
    class C logic;
    class D secondary;
    class E secondary;
    class F primary;
```

---
"""

text = text.replace("---", mermaid_chart, 1)

# Update integers
text = text.replace("12 offline episode bundles", "14 offline episode bundles")
text = text.replace("12 Offline Scenarios", "14 Offline Scenarios")

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(text)

print("Patched README.md successfully.")
