import re

with open('inference.py', 'r') as f:
    code = f.read()

helpers = """
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error=None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True
    )

def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True
    )

"""

# 1. Insert helpers
code = code.replace("def main() -> None:\n", helpers + "def main() -> None:\n")

# 2. Extract run_episode body and indent
m = re.search(r'(    def run_episode\(env: ArjunaEnv, llm_client: OpenAI, seed: int, ep_idx: int\) -> float:\n)(.*?)(\n    n_seeds = int\(os.environ.get\("N_SEEDS", "3"\)\))', code, re.DOTALL)
if m:
    header = m.group(1)
    body = m.group(2)
    footer = m.group(3)
    
    # Prefix
    prefix = """        log_start(
            task="arjuna-perception",
            env="arjuna-perception-env",
            model=MODEL_NAME
        )
        step_rewards: list[float] = []
        try:
"""
    
    # Body modifications
    body = body.replace('        step_rewards: list[float] = []\n\n', '')
    
    new_body = ""
    for line in body.splitlines(True):
        if line.strip() == "":
            new_body += line
        else:
            new_body += "    " + line
            
    # Add the log_step call
    log_step_code = """
                # Build a short action string for logging
                if task == 1:
                    action_str = f"task1_label={action.task1_label}"
                elif task == 2:
                    action_str = f"ranked_objects={action.ranked_objects}"
                else:
                    action_str = f"decision={action.decision}"
                
                log_step(
                    step=sub,
                    action=action_str,
                    reward=rw,
                    done=step_out.done,
                    error=None
                )
"""
    new_body = new_body.replace('                print(f"STEP {sub}/3 reward={rw:.3f}")\n', '                print(f"STEP {sub}/3 reward={rw:.3f}")\n' + log_step_code)

    finally_code = """        finally:
            steps_done = len(step_rewards)
            score = overall if 'overall' in locals() else (mean(step_rewards) if step_rewards else 0.0)
            success_val = score >= 0.5
            log_end(
                success=success_val,
                steps=steps_done,
                score=score,
                rewards=step_rewards
            )"""

    new_func = header + prefix + new_body + "\n" + finally_code + footer
    code = code[:m.start()] + new_func + code[m.end():]
    
    with open('inference.py', 'w') as f:
        f.write(code)
    print("Updated inference.py successfully.")
else:
    print("Could not find run_episode pattern.")
