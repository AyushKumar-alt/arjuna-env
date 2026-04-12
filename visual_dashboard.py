
import streamlit as st
import plotly.graph_objects as go
import requests
import json
import re
import os
import time
from openai import OpenAI
from statistics import mean
import pandas as pd

# Global Production Prompts (imported logic style)
TASK1_SYSTEM = "You are ARJUNA, an autonomous robot vision system. Identify the object and respond with its class label only. One word or short phrase. No explanation."
TASK2_SYSTEM = """You are ARJUNA, an autonomous robot at an industrial or urban site. Rank detected objects by importance. Rules: (1) higher confidence first; (2) ties -> person > vehicle > other. Output JSON array of label strings only: ["label_a", "label_b"]""".strip()
TASK3_SYSTEM = """You are ARJUNA. Decide next action for low-confidence detection. Options: discard (<0.35), request_rescan (0.35-0.50), log_and_continue (>=0.50). MANDATORY: 0.42 and 0.46 are < 0.50, thus "request_rescan". Return JSON only: {"decision": "...", "reasoning": "..."}""".strip()

# Set Page Config
st.set_page_config(
    page_title="ARJUNA | Agent Command Center",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #e0e6ed;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    .status-pulse {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: #00ff00;
        box-shadow: 0 0 8px #00ff00;
        margin-right: 8px;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { opacity: 0.4; }
        50% { opacity: 1; }
        100% { opacity: 0.4; }
    }
    .log-entry {
        font-family: 'Courier New', Courier, monospace;
        padding: 5px;
        border-bottom: 1px solid #333;
    }
    .reasoning { color: #888; font-style: italic; }
    .action { color: #00d4ff; font-weight: bold; }
    .reward { color: #00ff00; }
</style>
""", unsafe_allow_html=True)

# Helper: Parse Bounding Boxes from Observation Text
def parse_bboxes(text):
    # Pattern: label='...', confidence=..., bbox_xyxy=[x1, y1, x2, y2]
    pattern = r"label='(.*?)', confidence=(.*?), bbox_xyxy=\[(.*?)\]"
    matches = re.finditer(pattern, text)
    objects = []
    for m in matches:
        label = m.group(1)
        conf = float(m.group(2))
        bbox = [int(x.strip()) for x in m.group(3).split(',')]
        # Center point for radar visualization
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        objects.append({
            "label": label,
            "confidence": conf,
            "x": cx,
            "y": cy,
            "width": bbox[2] - bbox[0],
            "height": bbox[3] - bbox[1]
        })
    return objects

# NEW: Robot Stage Navigator Visualization
def render_robot_navigator(current_diff):
    levels = ["easy", "medium", "hard"]
    colors = ["#28a745", "#ffc107", "#dc3545"] # Green, Yellow, Red
    
    fig = go.Figure()
    
    # Draw the zones
    for i, (lvl, color) in enumerate(zip(levels, colors)):
        fig.add_shape(
            type="rect", x0=i, y0=0, x1=i+1, y1=1,
            fillcolor=color, opacity=0.15, line=dict(width=0),
        )
        # Zone Label
        fig.add_annotation(
            x=i+0.5, y=0.1, text=lvl.upper(),
            showarrow=False, font=dict(color=color, size=14, family="Outfit")
        )

    # Calculate Robot Position (Discrete center of the current level)
    stage_idx = levels.index(current_diff.lower()) if current_diff.lower() in levels else 0
    robot_x = stage_idx + 0.5
    
    # Add the Robot Icon
    fig.add_trace(go.Scatter(
        x=[robot_x], y=[0.5],
        mode="markers+text",
        marker=dict(size=40, color="#ffffff", symbol="square"),
        text=["🤖"], textposition="middle center",
        textfont=dict(size=30),
        name="Agent Position"
    ))

    fig.update_layout(
        height=180, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 3]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1]),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    return fig

# Sidebar Configuration
st.sidebar.title("🛠 Mission Control")
st.sidebar.markdown("Configure the agent session parameters below.")

api_provider = st.sidebar.selectbox("API Provider", ["Hugging Face", "Groq", "Custom"])

if api_provider == "Hugging Face":
    default_url = "https://router.huggingface.co/v1"
elif api_provider == "Groq":
    default_url = "https://api.groq.com/openai/v1"
else:
    default_url = os.getenv("API_BASE_URL", "")

api_key = st.sidebar.text_input(f"{api_provider} API Key", type="password", value=os.getenv("HF_TOKEN", "") if api_provider == "Hugging Face" else "")
model_name = st.sidebar.text_input("Model Name", value="meta-llama/Llama-3.3-70B-Instruct" if api_provider == "Hugging Face" else "llama-3.3-70b-versatile")
base_url_input = st.sidebar.text_input("Inference Base URL", value=default_url)
env_url = st.sidebar.text_input("Arjuna Environment URL", value="http://127.0.0.1:7860")

# NEW: Difficulty Control Toggle
st.sidebar.divider()
st.sidebar.subheader("🕹️ Learning Engine Mode")
learning_mode = st.sidebar.radio("Engine Logic", ["AutoRL (Autonomous)", "Manual (Force Tier)"])

target_difficulty = "auto"
if learning_mode == "Manual (Force Tier)":
    target_difficulty = st.sidebar.selectbox("Force Difficulty", ["easy", "medium", "hard"])

# Sync override with server
if st.sidebar.button("💾 Apply Engine Settings"):
    try:
        requests.post(f"{env_url}/curriculum/difficulty", params={"difficulty": target_difficulty}, timeout=2)
        st.sidebar.success(f"Locked to: {target_difficulty.upper()}")
        st.rerun() # Force UI refresh to show new status board
    except:
        st.sidebar.error("Failed to sync with server.")

st.sidebar.divider()
num_episodes = st.sidebar.slider("Episodes to Play", 1, 50, 5)
seed_start = st.sidebar.number_input("Starting Seed", value=42)

start_button = st.sidebar.button("🚀 ENGAGE AGENT", use_container_width=True)

# Main App Layout
col_left, col_right = st.columns([2, 1])

with col_left:
    # Hero Section: Robot Stage Navigator
    nav_placeholder = st.empty()
    st.markdown("---") # Visual separation
    
    # --- TOP STATUS Intel Banner ---
    try:
        curr_intel = requests.get(f"{env_url}/curriculum", timeout=1).json()
        current_diff = curr_intel['current_difficulty'].upper()
        if curr_intel.get('override'):
            status_color = "🔴" if current_diff == "HARD" else "🟡" if current_diff == "MEDIUM" else "🟢"
            st.markdown(f"""
                <div style="background-color:rgba(255,0,0,0.1); padding:20px; border-radius:10px; border: 2px solid {status_color == '🔴' and 'red' or 'orange'}">
                    <h2 style="color:{'red' if status_color == '🔴' else 'orange'}; margin:0;">{status_color} {current_diff} TIER ACTIVE (LOCKED)</h2>
                    <p style="margin:0; opacity:0.8;">Environment is manually locked to high-complexity scenes for demonstration.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"### 🤖 MISSION TIER: {current_diff} (AUTONOMOUS)")
    except Exception as e:
        st.markdown("### ⚠️ SYSTEM STATUS: DISCONNECTED")

    st.title("🤖 ARJUNA | Agent Command Center")
    st.markdown("Monitor real-time perception and decision logic in the simulated environment.")
    
    radar_placeholder = st.empty()
    
    # Session Metrics HUD
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        current_ep_metric = st.empty()
    with m_col2:
        avg_reward_metric = st.empty()
    with m_col3:
        status_metric = st.empty()
    
    # NEW: Curriculum Stats
    st.divider()
    curr_col1, curr_col2, curr_col3 = st.columns(3)
    with curr_col1:
        difficulty_metric = st.empty()
    with curr_col2:
        recent_mean_metric = st.empty()
    with curr_col3:
        progress_metric = st.empty()

with col_right:
    st.subheader("📋 Mission Log")
    log_container = st.container(height=500)
    
    st.subheader("📊 Analytics")
    chart_placeholder = st.empty()
    
    # Copy Log Area
    st.divider()
    if st.button("📋 Copy Full Mission Log"):
        # Convert HTML logs to plain text for clipboard
        full_text = "\n".join([re.sub('<[^<]+?>', '', log) for log in st.session_state.logs])
        st.code(full_text, language="text")
        st.success("Log formatted for copying above!")

# Initialization of State & Initial Hydration
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'stats' not in st.session_state:
    st.session_state.stats = {"rewards": [], "difficulty": {}}

# Render Initial Navigator
try:
    init_curr = requests.get(f"{env_url}/curriculum", timeout=1).json()
    nav_fig = render_robot_navigator(init_curr['current_difficulty'])
    nav_placeholder.plotly_chart(nav_fig, width='stretch', key="init_nav")
except:
    pass

# Logic for Running the Agent
if start_button:
    if not api_key:
        st.error("Please provide an API Key to start the mission.")
    else:
        # Client Setup
        llm = OpenAI(base_url=base_url_input, api_key=api_key)
        
        # Performance Tracking
        all_rewards = []
        
        for ep_idx in range(num_episodes):
            seed = seed_start + ep_idx
            status_metric.markdown(f'<div class="status-pulse"></div> Agent Playing (Seed {seed})', unsafe_allow_html=True)
            current_ep_metric.metric("Current Episode", f"{ep_idx + 1} / {num_episodes}")
            
            try:
                # 1. Reset Environment
                r = requests.post(f"{env_url}/reset", json={"seed": seed}, timeout=5)
                if r.status_code != 200:
                    st.error(f"Failed to connect to environment at {env_url}. Is the server running?")
                    break
                
                data = r.json()["observation"]
                eid = data["episode_id"]
                bundle = data["bundle_name"]
                
                st.session_state.logs.append(f"<b>[NEW EPISODE] Seed {seed} - {bundle}</b>")
                
                ep_rewards = []
                
                # 2. Play 3-Step Episode
                for step in range(1, 4):
                    obs_text = data["observation_text"]
                    task_type = data["task_type"]
                    
                    # Update Radar View
                    found_objects = parse_bboxes(obs_text)
                    
                    # Create Plotly Radar
                    fig = go.Figure()
                    # Boundary
                    fig.add_shape(type="rect", x0=0, y0=0, x1=640, y1=480, line=dict(color="#333", width=2))
                    
                    for obj in found_objects:
                        color = "#00d4ff" if task_type == 2 else "#ffaa00"
                        fig.add_trace(go.Scatter(
                            x=[obj['x']], y=[480 - obj['y']], # Invert Y for UI
                            mode="markers+text",
                            marker=dict(size=15, color=color, symbol="square"),
                            name=obj['label'],
                            text=[obj['label']],
                            textposition="top center",
                            hovertemplate=f"<b>{obj['label']}</b><br>Conf: {obj['confidence']:.2f}"
                        ))
                    
                    fig.update_layout(
                        title=f"Scene Visualization: {bundle} (Tier: {current_diff} | Objects: {num_objs} | Step {step}/3)",
                        xaxis=dict(range=[0, 640], showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(range=[0, 480], showgrid=False, zeroline=False, showticklabels=False),
                        width=800, height=450,
                        template="plotly_dark",
                        showlegend=False,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    radar_placeholder.plotly_chart(fig, width='stretch', key=f"radar_chart_{ep_idx}_{step}")
                    
                    # Use the "Hardened" production-validated prompts from inference.py
                    if task_type == 1:
                        sys = TASK1_SYSTEM + " CRITICAL: Use the EXACT label provided in the YOLO detection text (e.g., if it says 'worker', respond 'worker', NOT 'person')."
                    elif task_type == 2:
                        sys = TASK2_SYSTEM
                    else:
                        sys = TASK3_SYSTEM
                    
                    resp = llm.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "system", "content": sys}, {"role": "user", "content": obs_text}],
                        temperature=0.3
                    )
                    reply = resp.choices[0].message.content
                    
                    # Prepare Action with robust parsing
                    action = {}
                    if task_type == 1:
                        # Improved Task 1 Parser: Handle multi-word labels like "shopping bag"
                        label_clean = reply.strip().lower().strip(".,'\"")
                        # If it's a sentence: "It is a forklift", just take the last part
                        if " " in label_clean and not any(x in label_clean for x in ["shopping", "traffic", "fire hydrant"]):
                             label_clean = label_clean.split()[-1]
                        action["task1_label"] = label_clean
                    elif task_type == 2:
                        # Use the same robust list extraction as inference.py
                        array_match = re.search(r'\[.*?\]', reply, re.DOTALL)
                        if array_match:
                            try:
                                action["ranked_objects"] = json.loads(array_match.group(0))
                            except:
                                action["ranked_objects"] = []
                        else:
                            action["ranked_objects"] = [lbl.strip().lower() for lbl in re.findall(r'"(.*?)"', reply)]
                    else:
                        # Decision logic for Task 3
                        norm = reply.lower().replace("-", "_")
                        if "request_rescan" in norm: action["decision"] = "request_rescan"
                        elif "log_and_continue" in norm: action["decision"] = "log_and_continue"
                        else: action["decision"] = "discard"
                    
                    # 3. Post Step
                    step_req = {"episode_id": eid, "action": action}
                    step_resp = requests.post(f"{env_url}/step", json=step_req).json()
                    
                    reward = step_resp["reward"]
                    ep_rewards.append(reward)
                    
                    # Log Update
                    with log_container:
                        st.markdown(f'<div class="log-entry"><span class="action">Step {step}:</span> {reply[:100]}...</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="log-entry">Reward: <span class="reward">{reward:.3f}</span></div>', unsafe_allow_html=True)
                    
                    if step_resp["done"]:
                        all_rewards.append(mean(ep_rewards))
                        avg_reward_metric.metric("Average Reward", f"{mean(all_rewards):.3f}")
                        break
                    
                    data = step_resp["observation"]
                    
                    # UPDATE CURRICULUM HUD & NAVIGATOR
                    try:
                        curr_resp = requests.get(f"{env_url}/curriculum", timeout=1).json()
                        diff = curr_resp['current_difficulty']
                        difficulty_metric.metric("Current Difficulty", diff.upper())
                        recent_mean_metric.metric("AutoRL Recent Mean", f"{curr_resp['recent_mean']:.3f}")
                        
                        # Render Robot Navigator
                        nav_fig = render_robot_navigator(diff)
                        nav_placeholder.plotly_chart(nav_fig, width='stretch', key=f"nav_{ep_idx}_{step}")

                        # Promotion Progress
                        window_len = len(curr_resp.get('window', []))
                        progress_metric.metric("Window Depth", f"{window_len}/5 eps")
                        
                        # Promotion Alert in Sidebar for focus
                        recent_mean = curr_resp.get('recent_mean', 0)
                        if recent_mean >= 0.80 and recent_mean < 0.85:
                            st.sidebar.warning(f"🚀 Promotion Imminent: {recent_mean:.3f}")
                        elif recent_mean >= 0.85:
                            st.sidebar.success(f"🔥 Excellence: {recent_mean:.3f}")
                    except:
                        pass
                        
                    time.sleep(0.5) # Animation speed
                    
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
                break

        status_metric.markdown("✅ Mission Complete", unsafe_allow_html=True)
        st.balloons()

        # Final Analytics Chart
        if all_rewards:
            df = pd.DataFrame({"Episode": list(range(1, len(all_rewards)+1)), "Reward": all_rewards})
            chart = go.Figure(data=[go.Scatter(x=df["Episode"], y=df["Reward"], mode='lines+markers', line=dict(color="#00d4ff"))])
            chart.update_layout(title="Agent Learning Curve", template="plotly_dark", xaxis_title="Episode", yaxis_title="Mean Reward")
            chart_placeholder.plotly_chart(chart, width='stretch', key="performance_summary")
