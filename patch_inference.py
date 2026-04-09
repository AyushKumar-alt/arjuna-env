with open('inference.py', 'r') as f:
    code = f.read()

# Increase max_tokens
code = code.replace(
    'max_tokens = int(os.environ.get("MAX_TOKENS", "80"))',
    'max_tokens = int(os.environ.get("MAX_TOKENS", "250"))'
)

# Update TASK2_SYSTEM
task2_old = """Return ONLY a JSON array of label strings:
["label_a", "label_b", "label_c"]

MANDATORY: Answer with the list ONLY. No preamble, no explanation, no "To solve this..." text. Just the array."""

task2_new = """First, print a brief 1-sentence Chain-of-Thought reasoning about the confidences.
Then, output the JSON array of label strings:
["label_a", "label_b", "label_c"]"""

code = code.replace(task2_old, task2_new)

# Update TASK3_SYSTEM
task3_old = """Return JSON only:
{"decision": "<choice>", "reasoning": "<short sentence mentioning the numeric value>"}
No other text."""

task3_new = """Return JSON only:
{"decision": "<choice>", "reasoning": "<mathematical Chain-of-Thought proving the confidence matches the threshold>"}
No other text."""

code = code.replace(task3_old, task3_new)

with open('inference.py', 'w') as f:
    f.write(code)

print("Updated inference.py successfully.")
