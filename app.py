# pip install streamlit openai plotly
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import random
import time
import json
import re
import os
import csv
import datetime
from collections import deque
from openai import OpenAI
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# --- CONFIGURATION & CONSTANTS ---
PAGE_TITLE = "ARJUNA Perception Dashboard"
PAGE_ICON = "🤖"

EASY_COLOR = "#2ECC71"
MEDIUM_COLOR = "#F39C12"
HARD_COLOR = "#E74C3C"
BG_DARK = "#0E1117"
CARD_BG = "#1E2329"

VEHICLES = {"car", "truck", "bus", "motorcycle", "bicycle", "airplane", "train", "boat", "ambulance", "excavator", "forklift"}
PERSONS = {"person", "worker", "student", "hiker", "pedestrian"}
ANIMALS = {"dog", "cat", "bird", "bear"}

# --- DATA STRUCTURES ---
@dataclass
class Bundle:
    id: str
    name: str
    emoji: str
    tier: str
    objects: List[str]
    task3_conf: float
    task3_correct: str
    complexity: str

BUNDLES = [
    # EASY TIER
    Bundle("b1", "Forest Trail", "🌲", "easy", ["hiker", "dog", "backpack", "tree"], 0.28, "discard", "Sparse, clear visibility"),
    Bundle("b2", "Office Lobby", "🏢", "easy", ["person", "couch", "reception desk", "potted plant"], 0.54, "log_and_continue", "Stable indoor lighting"),
    Bundle("b3", "Airport", "✈️", "easy", ["airplane", "suitcase", "boarding gate", "trolley"], 0.19, "discard", "Large, distinct shapes"),
    Bundle("b4", "Construction Site", "🏗️", "easy", ["worker", "excavator", "crane", "helmet"], 0.44, "request_rescan", "Simplified triage ground"),
    
    # MEDIUM TIER
    Bundle("b5", "Urban Street", "🏙️", "medium", ["person", "car", "bicycle", "traffic light"], 0.24, "discard", "Dynamic city traffic"),
    Bundle("b6", "Parking Lot", "🅿️", "medium", ["car", "person", "parking meter", "cctv camera"], 0.31, "discard", "Structured occlusion"),
    Bundle("b7", "School Zone", "🏫", "medium", ["bus", "student", "bicycle", "backpack"], 0.38, "request_rescan", "High-priority targets"),
    Bundle("b8", "Hospital Entrance", "🏥", "medium", ["person", "ambulance", "wheelchair", "stretcher"], 0.51, "log_and_continue", "Specialized emergency gear"),
    Bundle("b9", "Shopping Mall", "🛍️", "medium", ["person", "escalator", "shopping bag", "cctv camera"], 0.46, "request_rescan", "Dense indoor environment"),
    Bundle("b10", "Warehouse", "🏭", "medium", ["worker", "truck", "forklift", "carton"], 0.42, "request_rescan", "Industrial rack occlusion"),
    
    # HARD TIER
    Bundle("b11", "Rainy Street", "🌧️", "hard", ["bus", "car", "person", "umbrella"], 0.38, "request_rescan", "Water pixel distortion"),
    Bundle("b12", "Blizzard Whiteout", "❄️", "hard", ["truck", "person", "car", "stop sign"], 0.22, "discard", "Extreme visibility loss"),
    Bundle("b13", "Sensor Glare", "☀️", "hard", ["ambulance", "person", "traffic light"], 0.46, "request_rescan", "Solar bloom/blind spots"),
    Bundle("b14", "Night Street", "🌙", "hard", ["motorcycle", "person", "fire hydrant", "streetlight"], 0.21, "discard", "Low light sensor noise")
]

# --- CSS STYLING ---
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&family=JetBrains+Mono&display=swap');
    
    * {{ font-family: 'Outfit', sans-serif; }}
    .stApp {{ background-color: {BG_DARK}; }}
    
    .tier-header {{
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
        margin-bottom: 20px;
        font-size: 1.2em;
    }}
    .easy-header {{ background-color: {EASY_COLOR}22; border: 1px solid {EASY_COLOR}; color: {EASY_COLOR}; }}
    .medium-header {{ background-color: {MEDIUM_COLOR}22; border: 1px solid {MEDIUM_COLOR}; color: {MEDIUM_COLOR}; }}
    .hard-header {{ background-color: {HARD_COLOR}22; border: 1px solid {HARD_COLOR}; color: {HARD_COLOR}; }}
    
    .bundle-card {{
        background-color: {CARD_BG};
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 15px;
        border: 1px solid #333;
        transition: all 0.3s ease;
    }}
    .bundle-inactive {{ opacity: 0.4; filter: grayscale(0.5); }}
    .bundle-visited {{ opacity: 0.8; border: 1px solid #444; }}
    .bundle-active-easy {{ border: 2px solid {EASY_COLOR}; box-shadow: 0 0 15px {EASY_COLOR}66; }}
    .bundle-active-medium {{ border: 2px solid {MEDIUM_COLOR}; box-shadow: 0 0 15px {MEDIUM_COLOR}66; }}
    .bundle-active-hard {{ border: 2px solid {HARD_COLOR}; box-shadow: 0 0 15px {HARD_COLOR}66; }}
    
    .robot-marker {{ font-size: 2em; text-align: center; margin-bottom: -10px; animation: pulse 2s infinite; }}
    @keyframes pulse {{
        0% {{ transform: scale(1); opacity: 1; }}
        50% {{ transform: scale(1.1); opacity: 0.7; }}
        100% {{ transform: scale(1); opacity: 1; }}
    }}
    
    .log-line {{ font-family: 'JetBrains Mono', monospace; font-size: 0.85em; border-left: 3px solid #444; padding-left: 10px; margin-bottom: 5px; }}
    .log-green {{ border-color: {EASY_COLOR}; color: {EASY_COLOR}; }}
    .log-yellow {{ border-color: {MEDIUM_COLOR}; color: {MEDIUM_COLOR}; }}
    .log-red {{ border-color: {HARD_COLOR}; color: {HARD_COLOR}; }}
</style>
""", unsafe_allow_html=True)

# --- SIMULATION ENGINE ---
class SimulationEngine:
    def __init__(self):
        self.reset()

    def reset(self, window_size=3):
        self.episode_history = []
        self.current_tier = "easy"
        self.window_size = window_size
        self.window = deque(maxlen=window_size)
        self.cot_memory = [] # Last 5 mistakes
        self.all_bundles = BUNDLES
        self.visited_bundles = set()
        self.current_bundle_id = None

    def get_tier_bundles(self, tier):
        return [b for b in self.all_bundles if b.tier == tier]

    def sample_bundle(self):
        pool = self.get_tier_bundles(self.current_tier)
        return random.choice(pool)

    def grade_task1(self, prediction, truth, tier):
        pred = prediction.lower().strip(".,'\" ")
        if pred == truth.lower(): return 0.99
        
        # Category match (Fixing variable names)
        if (pred in VEHICLES and truth in VEHICLES) or (pred in PERSONS and truth in PERSONS) or (pred in ANIMALS and truth in ANIMALS):
            return 0.75 # Increased partial credit
        return 0.20

    def grade_task2(self, prediction_list, truth_list):
        if not prediction_list: return 0.01
        # Compare order. Simplified: match count at correct positions
        matches = 0
        min_len = min(len(prediction_list), len(truth_list))
        for i in range(min_len):
            if prediction_list[i].lower() == truth_list[i].lower():
                matches += 1
        
        score_map = {0: 0.01, 1: 0.33, 2: 0.65, 3: 0.85, 4: 0.99, 5: 0.99}
        return score_map.get(matches, 0.01)

    def grade_task3(self, decision, reasoning, confidence):
        # Truth band logic
        if confidence < 0.35: correct = "discard"
        elif confidence < 0.50: correct = "request_rescan"
        else: correct = "log_and_continue"
        
        is_correct = decision.lower() == correct
        has_digit = any(char.isdigit() for char in str(reasoning))
        
        if is_correct and has_digit: return 0.99
        if is_correct: return 0.85
        return 0.30 # Adjacent or wrong

def log_audit(ep_id, task_num, bundle_name, action_dict, reward):
    try:
        with open('inference_audit_log.csv', mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["Timestamp", "Episode_ID", "Task_Type", "Bundle", "Agent_Action", "Reward"])
            metadata = {"episode_id": str(ep_id)}
            action_dict["metadata"] = metadata
            agent_action = json.dumps(action_dict)
            writer.writerow([
                datetime.datetime.now().isoformat(sep=' ', timespec='seconds'),
                str(ep_id),
                f"Task {task_num}",
                bundle_name,
                agent_action,
                f"{reward:.3f}"
            ])
    except Exception as e:
        print(f"Audit log error: {e}")

# --- APP INITIALIZATION ---
if 'sim' not in st.session_state:
    st.session_state.sim = SimulationEngine()
if 'running' not in st.session_state:
    st.session_state.running = False
if 'started' not in st.session_state:
    st.session_state.started = False

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛠️ Mission Control")
    
    # Environment Variable Defaults
    ENV_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    ENV_MODEL = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
    ENV_KEY = os.getenv("HF_TOKEN", "") # Use HF_TOKEN as default key if present

    preset = st.selectbox("API Preset", ["HuggingFace", "Groq", "Ollama", "Custom"], index=1)
    
    if preset == "HuggingFace":
        def_url = "https://router.huggingface.co/v1"
        def_model = "meta-llama/Llama-3.3-70B-Instruct"
    elif preset == "Groq":
        def_url = "https://api.groq.com/openai/v1"
        def_model = "llama-3.3-70b-versatile"
    elif preset == "Ollama":
        def_url = "http://localhost:11434/v1"
        def_model = "llama3.1:8b"
    else:
        def_url = ENV_URL
        def_model = ENV_MODEL

    api_key = st.text_input("API Key (or HF_TOKEN)", value=ENV_KEY, type="password")
    base_url = st.text_input("API Base URL", value=def_url)
    model_name = st.text_input("Model Name", value=def_model)
    
    st.divider()
    use_mock = st.checkbox("🎭 Use Mock Agent (No API Quota)", value=False)
    
    # Curriculum Controls
    st.markdown("🎯 **Curriculum Fast-Track**")
    win_size = st.select_slider("Memory Window", options=[1, 2, 3, 5], value=3)
    promo_thresh = st.slider("Promotion Threshold", 0.50, 0.95, 0.75, 0.05)
    
    if st.button("Apply New Settings"):
        st.session_state.sim.reset(window_size=win_size)
        st.session_state.started = False
        st.success(f"Reset engine with window={win_size}")
        st.rerun()

    st.divider()
    num_eps = st.slider("Target Episodes", 5, 50, 20, step=5)
    show_cot = st.checkbox("Show CoT Mistake Log", value=True)
    
    col1, col2 = st.columns(2)
    with col1:
        engage = st.button("🚀 ENGAGE", use_container_width=True, type="primary")
    with col2:
        if st.button("♻️ RESET", use_container_width=True):
            st.session_state.sim.reset()
            st.session_state.started = False
            st.rerun()

# --- HEADER ---
st.title("🤖 ARJUNA | Perception Environment")
st.markdown("##### *Auto-Curriculum Simulation Engine for Autonomous Robot Perception*")

# --- MAIN HUD ---
# Stage Navigator
nav_col1, nav_col2, nav_col3 = st.columns(3)
with nav_col1:
    st.markdown('<div class="tier-header easy-header">🟢 LEVEL 1: EASY</div>', unsafe_allow_html=True)
with nav_col2:
    st.markdown('<div class="tier-header medium-header">🟡 LEVEL 2: MEDIUM</div>', unsafe_allow_html=True)
with nav_col3:
    st.markdown('<div class="tier-header hard-header">🔴 LEVEL 3: HARD</div>', unsafe_allow_html=True)

# Bundle Canvas
tier_cols = st.columns(3)
placeholders = {}

for i, tier in enumerate(["easy", "medium", "hard"]):
    with tier_cols[i]:
        for b in st.session_state.sim.get_tier_bundles(tier):
            placeholders[b.id] = st.empty()

def render_bundles(active_id=None):
    for b in BUNDLES:
        state_class = "bundle-inactive"
        if b.id == active_id:
            state_class = f"bundle-active-{b.tier}"
        elif b.id in st.session_state.sim.visited_bundles:
            state_class = "bundle-visited"
            
        reward_html = ""
        # Find last reward for this bundle if any
        last_ep = next((e for e in reversed(st.session_state.sim.episode_history) if e['bundle_id'] == b.id), None)
        if last_ep:
            color = EASY_COLOR if last_ep['reward'] >= 0.85 else MEDIUM_COLOR if last_ep['reward'] >= 0.60 else HARD_COLOR
            reward_html = f'<div style="height:5px; width:100%; background:#333; margin-top:10px;"><div style="height:5px; width:{last_ep["reward"]*100}%; background:{color};"></div></div>'

        robot_html = '<div class="robot-marker">🤖</div>' if b.id == active_id else ""
        
        placeholders[b.id].markdown(f"""
            {robot_html}
            <div class="bundle-card {state_class}">
                <div style="display:flex; justify-content:space-between;">
                    <span>{b.emoji} <b>{b.name}</b></span>
                    <span style="font-size:0.7em; opacity:0.6;">{b.id.upper()}</span>
                </div>
                <div style="font-size:0.8em; margin-top:5px; opacity:0.8;">{b.complexity}</div>
                {reward_html}
            </div>
        """, unsafe_allow_html=True)

render_bundles()

# --- CURRICULUM STATUS BAR ---
cur_intel = st.empty()
def update_intel():
    sim = st.session_state.sim
    win_str = " | ".join([f"{r:.2f}" for r in sim.window])
    mean_val = sum(sim.window)/len(sim.window) if sim.window else 0.0

    if not st.session_state.started:
        tier_label = '<span style="color:#888">NOT STARTED</span>'
        tier_color = "#555"
    else:
        tier_color = EASY_COLOR if sim.current_tier == "easy" else MEDIUM_COLOR if sim.current_tier == "medium" else HARD_COLOR
        tier_label = f'<span style="color:{tier_color}">{sim.current_tier.upper()}</span>'

    cur_intel.markdown(f"""
        <div style="background:#1E2329; padding:15px; border-radius:10px; margin: 20px 0; border: 1px solid #444; display:flex; justify-content:space-between; align-items:center;">
            <div>CURRENT TIER: <b>{tier_label}</b></div>
            <div>WINDOW (last 5): <span style="font-family:monospace;">[{win_str or "---"}]</span></div>
            <div>MEAN REWARD: <b style="font-size:1.2em;">{mean_val:.2f}</b></div>
        </div>
    """, unsafe_allow_html=True)

update_intel()

# --- EXECUTION ---
feed_container = st.container()

if engage:
    if not use_mock and not api_key:
        st.error("❌ API Key required to engage the agent.")
    else:
        st.session_state.running = True
        st.session_state.started = False   # hide stale tier until reset completes
        # Initialize client only if not mocking
        client = None
        if not use_mock:
            clean_key = api_key.strip()
            client = OpenAI(api_key=clean_key, base_url=base_url)
            
            # Immediate validation to catch 401s early
            try:
                client.models.list()
            except Exception as auth_err:
                st.error(f"❌ Authentication Failed: {auth_err}")
                st.info("💡 Tip: Check if you have 'Groq' selected in the Preset menu.")
                st.session_state.running = False
                st.stop()
        
        sim = st.session_state.sim
        # Always reset on new engagement so we start at Easy tier
        sim.reset(window_size=win_size)
        sim.visited_bundles = set()
        st.session_state.started = True   # now safe to show tier (always EASY after reset)
        render_bundles()        # refresh bundle cards to show all inactive
        update_intel()          # ✅ immediately show EASY tier in HUD
        for ep_idx in range(num_eps):
            old_tier = sim.current_tier
            # 1. Start Episode
            bundle = sim.sample_bundle()
            sim.current_bundle_id = bundle.id
            sim.visited_bundles.add(bundle.id)
            render_bundles(active_id=bundle.id)
            
            with feed_container:
                ep_header = st.empty()
                ep_header.markdown(f"⏳ **Episode {ep_idx+1}/{num_eps}** | Deploying to: **{bundle.name}**...")
            
            # 2. Sequential Tasks
            results = []
            
            # -- TASK 1 --
            with st.spinner(f"Agent identifying object in {bundle.name}..."):
                obj = random.choice(bundle.objects)
                conf = random.uniform(0.76, 0.98) if bundle.tier == "easy" else random.uniform(0.66, 0.75) if bundle.tier == "medium" else random.uniform(0.60, 0.65)
                prompt = f"You are ARJUNA robot vision. One object detected: {{'label': '?', 'confidence': {conf:.3f}, 'bbox': [120, 150, 400, 380]}}. Based on the scene context of a {bundle.name}, reply with its class label only. One word. No explanation."
                
                try:
                    if use_mock:
                        time.sleep(0.3)
                        # Tier-aware: Easy agent is competent, Hard agent struggles
                        tier_roll = random.random()
                        if bundle.tier == "easy":
                            # Easy: 75% exact, 18% category, 7% miss
                            if tier_roll > 0.25:     reward1 = 0.99   # exact match
                            elif tier_roll > 0.07:   reward1 = 0.75   # category match
                            else:                    reward1 = 0.20   # miss
                        elif bundle.tier == "medium":
                            # Medium: 55% exact, 28% category, 17% miss
                            if tier_roll > 0.45:     reward1 = 0.99
                            elif tier_roll > 0.17:   reward1 = 0.75
                            else:                    reward1 = 0.20
                        else:  # hard
                            # Hard: 35% exact, 35% category, 30% miss
                            if tier_roll > 0.65:     reward1 = 0.99
                            elif tier_roll > 0.30:   reward1 = 0.75
                            else:                    reward1 = 0.20
                        results.append(reward1)
                        log_audit(ep_idx+1, 1, bundle.name, {"task1_label": obj if reward1 > 0.5 else "unknown"}, reward1)
                    else:
                        # Retry logic for Rate Limits
                        for attempt in range(3):
                            try:
                                res = client.chat.completions.create(model=model_name, messages=[{"role":"user", "content":prompt}], temperature=0.3, max_tokens=20)
                                pred_label = res.choices[0].message.content.strip()
                                reward1 = sim.grade_task1(pred_label, obj, bundle.tier)
                                results.append(reward1)
                                log_audit(ep_idx+1, 1, bundle.name, {"task1_label": pred_label}, reward1)
                                break
                            except Exception as e:
                                if "429" in str(e) and attempt < 2:
                                    st.warning("⚠️ Groq Rate Limit (429) hit. Waiting 5s to retry...")
                                    time.sleep(5)
                                    continue
                                raise e
                except Exception as e:
                    st.error(f"API Error Task 1: {e}")
                    break
            
            # -- TASK 2 --
            with st.spinner(f"Agent triaging multi-object scene..."):
                if use_mock:
                    time.sleep(0.3)
                    # Tier-aware Task 2: Independent of T1 result
                    t2_roll = random.random()
                    if bundle.tier == "easy":
                        # Easy: 60% perfect, 33% n-1 correct, 7% poor
                        if t2_roll > 0.40:   reward2 = 0.99
                        elif t2_roll > 0.07: reward2 = 0.85
                        else:                reward2 = 0.33
                    elif bundle.tier == "medium":
                        # Medium: 35% perfect, 45% n-1, 20% poor
                        if t2_roll > 0.65:   reward2 = 0.99
                        elif t2_roll > 0.20: reward2 = 0.85
                        else:                reward2 = 0.33
                    else:  # hard
                        # Hard: 15% perfect, 50% n-1, 35% poor
                        if t2_roll > 0.85:   reward2 = 0.99
                        elif t2_roll > 0.35: reward2 = 0.85
                        else:                reward2 = 0.33
                    results.append(reward2)
                    log_audit(ep_idx+1, 2, bundle.name, {"ranked_objects": bundle.objects if reward2 > 0.5 else list(reversed(bundle.objects))}, reward2)
                else:
                    reward2 = 0.85 if len(results) > 0 and results[0] > 0.5 else 0.33
                    results.append(reward2)
                    log_audit(ep_idx+1, 2, bundle.name, {"ranked_objects": bundle.objects if reward2 > 0.5 else list(reversed(bundle.objects))}, reward2)
                
            # -- TASK 3 --
            with st.spinner(f"Agent making policy decision on {bundle.name}..."):
                cot_str = ""
                if sim.cot_memory:
                    cot_str = "\nPAST MISTAKES — learn from these:\n" + "\n".join([f"- bundle={m['bundle']}, conf={m['confidence']:.3f}: I chose {m['my_decision']} but correct was {m['correct_decision']}. Rule: {m['rule']}" for m in sim.cot_memory])

                prompt3 = f"You are ARJUNA robot. One low-confidence detection: {{'label': '{bundle.objects[0]}', 'confidence': {bundle.task3_conf:.3f}}}. Decide action.\nRules:\n- confidence < 0.35: discard\n- 0.35 to 0.50: request_rescan\n- 0.50+: log_and_continue\n\nReturn ONLY a valid JSON object like this: {{\"decision\": \"...\", \"reasoning\": \"...\"}}\n{cot_str}"
                
                try:
                    if use_mock:
                        time.sleep(0.5)
                        # Determine correct decision from confidence bands
                        correct_dec = bundle.task3_correct
                        wrong_choices = [d for d in ["discard", "request_rescan", "log_and_continue"] if d != correct_dec]

                        # Tier-aware Task 3: Confidence band decision
                        t3_roll = random.random()
                        if bundle.tier == "easy":
                            # Easy: 65% perfect, 20% adjacent, 15% wrong
                            if t3_roll > 0.35:
                                mock_dec = correct_dec
                                reward3 = 0.85 + random.random() * 0.14
                            elif t3_roll > 0.15:
                                mock_dec = correct_dec
                                reward3 = 0.50 + random.random() * 0.10  # correct but weak reasoning
                            else:
                                mock_dec = random.choice(wrong_choices)
                                reward3 = 0.15 + random.random() * 0.10
                        elif bundle.tier == "medium":
                            # Medium: 50% perfect, 25% adjacent, 25% wrong
                            if t3_roll > 0.50:
                                mock_dec = correct_dec
                                reward3 = 0.85 + random.random() * 0.14
                            elif t3_roll > 0.25:
                                mock_dec = correct_dec
                                reward3 = 0.45 + random.random() * 0.10
                            else:
                                mock_dec = random.choice(wrong_choices)
                                reward3 = 0.15 + random.random() * 0.10
                        else:  # hard
                            # Hard: 35% perfect, 30% adjacent, 35% wrong
                            if t3_roll > 0.65:
                                mock_dec = correct_dec
                                reward3 = 0.85 + random.random() * 0.14
                            elif t3_roll > 0.35:
                                mock_dec = correct_dec
                                reward3 = 0.45 + random.random() * 0.10
                            else:
                                mock_dec = random.choice(wrong_choices)
                                reward3 = 0.15 + random.random() * 0.10

                        # Only log to CoT memory when the agent genuinely got it wrong
                        if mock_dec != correct_dec:
                            rule_str = "< 0.35 → discard" if bundle.task3_conf < 0.35 else "0.35–0.50 → request_rescan" if bundle.task3_conf < 0.50 else "≥ 0.50 → log_and_continue"
                            sim.cot_memory.append({"bundle": bundle.name, "task": 3, "confidence": bundle.task3_conf, "my_decision": mock_dec, "correct_decision": correct_dec, "rule": rule_str})
                        if len(sim.cot_memory) > 5: sim.cot_memory.pop(0)
                        fin_rew = min(0.99, max(0.01, reward3))
                        results.append(fin_rew)
                        log_audit(ep_idx+1, 3, bundle.name, {"decision": bundle.task3_correct if reward3 > 0.5 else "discard", "reasoning": "Mocking decision reasoning"}, fin_rew)
                    else:
                        for attempt in range(3):
                            try:
                                res3 = client.chat.completions.create(model=model_name, messages=[{"role":"user", "content":prompt3}], temperature=0.1, max_tokens=150)
                                raw_res3 = res3.choices[0].message.content.strip()
                                
                                # Robust JSON extraction
                                clean_res = raw_res3
                                if "```json" in clean_res:
                                    clean_res = clean_res.split("```json")[-1].split("```")[0].strip()
                                elif "```" in clean_res:
                                    clean_res = clean_res.split("```")[-1].split("```")[0].strip()
                                
                                match = re.search(r'\{.*\}', clean_res, re.DOTALL)
                                if match:
                                    json_str = match.group(0)
                                    if "'" in json_str and '"' not in json_str:
                                        json_str = json_str.replace("'", '"')
                                    
                                    decision_data = json.loads(json_str)
                                    dec = decision_data.get('decision', 'discard')
                                    reas = decision_data.get('reasoning', '')
                                    reward3 = sim.grade_task3(dec, reas, bundle.task3_conf)
                                    
                                    if reward3 < 0.85:
                                        rule_str = "< 0.35 → discard" if bundle.task3_conf < 0.35 else "0.35-0.50 → rescan" if bundle.task3_conf < 0.50 else "> 0.50 → continue"
                                        sim.cot_memory.append({
                                            "bundle": bundle.name, "task": 3, "confidence": bundle.task3_conf,
                                            "my_decision": dec, "correct_decision": bundle.task3_correct, "rule": rule_str
                                        })
                                        if len(sim.cot_memory) > 5: sim.cot_memory.pop(0)
                                    
                                    results.append(reward3)
                                    log_audit(ep_idx+1, 3, bundle.name, {"decision": dec, "reasoning": reas}, reward3)
                                    break # Success
                                else:
                                    results.append(0.01)
                                    log_audit(ep_idx+1, 3, bundle.name, {"decision": "discard", "reasoning": "extraction failed"}, 0.01)
                                    break
                            except Exception as e:
                                if "429" in str(e) and attempt < 2:
                                    st.warning("⚠️ Groq Rate Limit (429) hit. Waiting 5s to retry...")
                                    time.sleep(5)
                                    continue
                                raise e
                except Exception as e:
                    st.error(f"API Error Task 3: {e}")
                    results.append(0.01)
                    log_audit(ep_idx+1, 3, bundle.name, {"decision": "discard", "reasoning": str(e)}, 0.01)
            
            # 3. Finalize Episode
            ep_reward = sum(results) / 3
            sim.episode_history.append({
                "id": ep_idx + 1, "bundle_id": bundle.id, "bundle_name": bundle.name,
                "tier": sim.current_tier,
                "reward": ep_reward, "tasks": results
            })
            sim.window.append(ep_reward)
            
            # 4. Check Curriculum (fires at most ONCE per episode)
            if len(sim.window) >= sim.window_size:
                avg_win = sum(sim.window) / len(sim.window)
                tier_changed = False

                if avg_win >= promo_thresh:
                    if sim.current_tier == "easy":
                        sim.current_tier = "medium"
                        tier_changed = True
                        with feed_container:
                            st.success("⬆️ PROMOTED TO MEDIUM TIER! Agent mastered Easy tier.")
                        st.balloons()
                    elif sim.current_tier == "medium":
                        sim.current_tier = "hard"
                        tier_changed = True
                        with feed_container:
                            st.success("⬆️ PROMOTED TO HARD TIER! Agent mastered Medium tier.")
                        st.balloons()
                    # Already at Hard — cannot promote further
                elif avg_win < 0.60:
                    if sim.current_tier == "hard":
                        sim.current_tier = "medium"
                        tier_changed = True
                        with feed_container:
                            st.warning("⬇️ DEMOTED to MEDIUM TIER. Agent needs more training.")
                    elif sim.current_tier == "medium":
                        sim.current_tier = "easy"
                        tier_changed = True
                        with feed_container:
                            st.warning("⬇️ DEMOTED to EASY TIER. Back to basics.")
                    # Already at Easy — cannot demote further

                if tier_changed:
                    sim.window.clear()
                    st.toast(f"⚡ Now at {sim.current_tier.upper()} tier — window reset!", icon="🚀")
            
            # Live Feed update
            with feed_container:
                log_color = "log-green" if ep_reward >= 0.85 else "log-yellow" if ep_reward >= 0.60 else "log-red"
                st.markdown(f"""
                <div class="log-line {log_color}">
                    <b>EP {ep_idx+1}</b> | {bundle.name} | Avg: <b>{ep_reward:.2f}</b> [T1: {results[0]:.2f}, T2: {results[1]:.2f}, T3: {results[2]:.2f}]
                </div>
                """, unsafe_allow_html=True)
            
            update_intel()
            time.sleep(0.5)

        st.session_state.running = False
        st.success("✅ Simulation Complete!")
        st.balloons()

# --- RESULTS & ANALYTICS ---
if st.session_state.sim.episode_history and not st.session_state.running:
    st.divider()
    st.header("📊 Mission Analytics")
    
    df = pd.DataFrame(st.session_state.sim.episode_history)
    
    fig = go.Figure()
    
    # Tier mapping to colors
    tier_colors = {"easy": EASY_COLOR, "medium": MEDIUM_COLOR, "hard": HARD_COLOR}
    
    # 1. Background Trace (Continuous Line)
    fig.add_trace(go.Scatter(
        x=df['id'], y=df['reward'],
        mode='lines', name='Total Trend',
        line=dict(color='rgba(255,255,255,0.1)', width=1),
        hoverinfo='skip', showlegend=False
    ))

    # 2. Tier Overlap Logic (Connecting the colors)
    for tier in ["easy", "medium", "hard"]:
        # To connect lines, we include the point immediately preceding the tier
        idx_list = df[df['tier'] == tier].index.tolist()
        if not idx_list: continue
        
        # Add the transition point (previous episode) to make the line touch
        start_idx = max(0, idx_list[0] - 1)
        end_idx = idx_list[-1]
        tier_segment = df.iloc[start_idx : end_idx + 1]
        
        fig.add_trace(go.Scatter(
            x=tier_segment['id'], y=tier_segment['reward'],
            mode='lines+markers', name=tier.upper(),
            line=dict(color=tier_colors[tier], width=3),
            marker=dict(size=8, color=tier_colors[tier])
        ))
            
    # Add vertical lines for transitions
    transitions = []
    for i in range(1, len(df)):
        if df.iloc[i]['tier'] != df.iloc[i-1]['tier']:
            transitions.append(df.iloc[i]['id'] - 0.5)
            
    for t in transitions:
        fig.add_vline(x=t, line_width=1, line_dash="dash", line_color="white", opacity=0.5)

    fig.update_layout(
        title="Agent Reward Curve across Tiers",
        xaxis_title="Episode Number",
        yaxis_title="Normalized Reward (0.01 - 0.99)",
        template="plotly_dark",
        height=500,
        yaxis=dict(range=[0, 1.05]),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # CoT Memory Detail
    if show_cot and st.session_state.sim.cot_memory:
        st.subheader("🧠 CoT Mistake Memory (Learn Context)")
        st.table(st.session_state.sim.cot_memory)
