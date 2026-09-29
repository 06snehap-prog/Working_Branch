import json
import os
from pathlib import Path

import boto3
from dotenv import dotenv_values, load_dotenv
import pandas as pd
import streamlit as st
from supabase import Client, create_client

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="WanderMind AI — Hyper-Personalized Travel Discovery",
    page_icon="🌍",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0b0f19; color: #f1f5f9; }
    .stButton>button { width: 100%; background-color: #0d9488; color: white; border-radius: 8px; font-weight: bold; border: none; padding: 0.6rem; }
    .stButton>button:hover { background-color: #0f766e; color: #ffffff; }
    .card-box { background-color: #1e293b; border-radius: 12px; padding: 1.25rem; border: 1px solid #334155; margin-bottom: 1rem; }
    .badge { background-color: #0f766e; color: white; padding: 0.25rem 0.6rem; border-radius: 12px; font-size: 0.8rem; font-weight: 600; margin-right: 0.4rem; }
    .badge-warning { background-color: #b45309; color: white; padding: 0.25rem 0.6rem; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem !important; color: #2dd4bf; }
    </style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# 2. CONFIGURATION & CLIENT INITIALIZATION
# ==========================================
REGION = "us-east-1"
# AWS Bedrock Model ID (Using Nova Micro for fast, cost-effective inference)
MODEL_ID = "us.amazon.nova-micro-v1:0"
SSO_PROFILE = "AdministratorAccess-715248855541"


@st.cache_resource
def get_supabase_client() -> Client:
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path, override=True)
    env_vars = dotenv_values(dotenv_path=env_path)

    supabase_url = os.getenv("SUPABASE_URL") or env_vars.get("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY") or env_vars.get("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        st.error(f"Missing SUPABASE_URL or SUPABASE_KEY in {env_path}")
        st.stop()

    return create_client(supabase_url, supabase_key)


@st.cache_resource
def get_bedrock_client():
    try:
        session = boto3.Session(profile_name=SSO_PROFILE, region_name=REGION)
        return session.client("bedrock-runtime")
    except Exception:
        try:
            return boto3.client("bedrock-runtime", region_name=REGION)
        except Exception as inner_e:
            st.error(f"Failed to initialize AWS Bedrock Client: {inner_e}")
            st.stop()


supabase = get_supabase_client()
bedrock_runtime = get_bedrock_client()

# ==========================================
# 3. TRAVEL DISCOVERY ENGINE AGENT
# ==========================================
class TravelDiscoveryEngine:

    def __init__(self, db: Client, bedrock_client):
        self.db = db
        self.bedrock = bedrock_client
        self.model_id = MODEL_ID

    def fetch_user_travel_history(self, user_name: str):
        """Fetches past feedback to enable continuous learning."""
        try:
            res = (
                self.db.table("travel_feedback")
                .select("*")
                .eq("user_name", user_name)
                .order("created_at", desc=True)
                .limit(5)
                .execute()
            )
            return res.data or []
        except Exception:
            return []

    def save_user_feedback(self, user_name: str, destination: str, reaction: str, notes: str):
        """Saves explicitly learned preferences into Supabase memory."""
        try:
            payload = {
                "user_name": user_name,
                "destination_name": destination,
                "reaction": reaction,
                "user_notes": notes,
            }
            self.db.table("travel_feedback").insert(payload).execute()
            return True
        except Exception as e:
            st.error(f"Failed to save feedback: {e}")
            return False

    def generate_curated_destinations(self, context: dict) -> dict:
        """Deeply models traveler intent and generates structured recommendations."""
        system_prompt = (
            "You are an AI Travel Intent Specialist and Exploration Engine. "
            "Analyze the traveler's context (mood, life stage, trip pace, budget, seasonality, and past feedback). "
            "Surface 3 highly curated, non-generic destinations with custom experiential itineraries. "
            "Respond ONLY with a single valid JSON object (no markdown, no extra prose) matching this format:\n"
            "{\n"
            '  "intent_archetype": "Summary of modeled traveler mindset (e.g. Seeking Serene Restoration)",\n'
            '  "context_analysis": "2-sentence breakdown of how budget, seasonality, and mood balance out",\n'
            '  "destinations": [\n'
            "     {\n"
            '       "name": "City, Country",\n'
            '       "tagline": "Evocative summary",\n'
            '       "match_score": "95%",\n'
            '       "why_it_fits": "Deep reasoning aligned with mood and life stage",\n'
            '       "seasonality_status": "Optimal / Shoulder / Off-peak context",\n'
            '       "est_budget_breakdown": "$X flights + $Y/night stay",\n'
            '       "anchor_experience": "A unique experience matching their intent",\n'
            '       "trade_offs": "Honest cons or friction points (e.g. 12hr flight, rainy afternoons)"\n'
            "     }\n"
            "  ]\n"
            "}"
        )

        user_content = f"""
        Traveler Genome:
        - Name: {context['user_name']}
        - Current Mood/Energy: {context['mood']}
        - Life Stage Context: {context['life_stage']}
        - Desired Trip Pace: {context['pace']}
        - Month of Travel: {context['travel_month']}
        - Duration: {context['duration_days']} days
        - Total Budget: ${context['budget_usd']} USD
        - Open Text Intent: "{context['open_intent']}"
        - Historical Preference Memory: {context['past_history']}
        """

        try:
            response = self.bedrock.converse(
                modelId=self.model_id,
                system=[{"text": system_prompt}],
                messages=[{"role": "user", "content": [{"text": user_content}]}],
                inferenceConfig={"maxTokens": 1500, "temperature": 0.4},
            )

            response_text = response["output"]["message"]["content"][0]["text"].strip()
            if response_text.startswith("```"):
                response_text = (
                    response_text.replace("```json", "").replace("```", "").strip()
                )

            return json.loads(response_text)
        except Exception as e:
            st.error(f"Engine Generation Error: {e}")
            return {
                "intent_archetype": "Adaptive Explorer",
                "context_analysis": "Fallback generated due to timeout.",
                "destinations": [
                    {
                        "name": "Kyoto, Japan",
                        "tagline": "Tranquil temples & timeless gardens",
                        "match_score": "90%",
                        "why_it_fits": "Ideal for restorative mindsets and cultural immersion.",
                        "seasonality_status": "Optimal",
                        "est_budget_breakdown": "$800 flights + $150/night stay",
                        "anchor_experience": "Private morning walk through Arashiyama Bamboo Grove",
                        "trade_offs": "Popular sites can get crowded during midday.",
                    }
                ],
            }


engine = TravelDiscoveryEngine(supabase, bedrock_runtime)

# Initialize Session State
if "curated_results" not in st.session_state:
    st.session_state.curated_results = None

# ==========================================
# 4. USER INTERFACE & DASHBOARD
# ==========================================
st.title("🌍 WanderMind AI")
st.caption("AI-Native Travel Discovery Engine — Moving from Keyword Search to Intent Exploration")

st.markdown("---")

# MAIN SIDEBAR: CONTEXTUAL INTENT GENOME
with st.sidebar:
    st.header("🧬 Traveler Context Genome")
    
    user_name = st.text_input("Traveler Name", value="Alex Rivera")
    
    mood = st.selectbox(
        "Current Mood / Mindset",
        [
            "Burnt Out (Needs Pure Unplugging & Relaxation)",
            "Curious & Energetic (Cultural & Culinary Immersion)",
            "Restless (High-Adrenaline Outdoor Exploration)",
            "Reflective (Quiet, Scenic & Inspiring Solitude)",
        ],
    )
    
    life_stage = st.selectbox(
        "Life Stage & Companions",
        [
            "Solo Traveler",
            "Couple (Romantic & Shared Discovery)",
            "Young Family with Toddlers (Low Transit Friction)",
            "Group of Close Friends (Vibrant & Social)",
        ],
    )
    
    pace = st.select_slider(
        "Desired Trip Pace",
        options=["Slow & Mindful", "Balanced Exploration", "Packed & High-Octane"],
        value="Slow & Mindful",
    )
    
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        travel_month = st.selectbox(
            "Month",
            ["October", "November", "December", "January", "February", "March", "April"],
        )
    with col_b2:
        duration_days = st.number_input("Days", min_value=3, max_value=30, value=7)

    budget_usd = st.slider("Total Budget (USD)", 500, 15000, 3500, step=250)
    
    open_intent = st.text_area(
        "Vague Intent / Sensory Cues",
        placeholder="e.g., I want warm ocean air, fresh seafood, and no crowded tourist traps.",
    )
    
    discover_btn = st.button("✨ Proactively Surface Destinations")

# TOP METRICS & USER HISTORY
past_logs = engine.fetch_user_travel_history(user_name)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Profile Memory", user_name)
m2.metric("Saved Feedback Items", len(past_logs))
m3.metric("Current Budget Cap", f"${budget_usd}")
m4.metric("Target Month", travel_month)

st.markdown("---")

# DISCOVERY GENERATION EXECUTION
if discover_btn:
    with st.spinner("AI Engine is modeling intent vectors & calculating constraints..."):
        context_payload = {
            "user_name": user_name,
            "mood": mood,
            "life_stage": life_stage,
            "pace": pace,
            "travel_month": travel_month,
            "duration_days": duration_days,
            "budget_usd": budget_usd,
            "open_intent": open_intent,
            "past_history": past_logs,
        }
        st.session_state.curated_results = engine.generate_curated_destinations(context_payload)

# MAIN DISPLAY AREA
if st.session_state.curated_results:
    res = st.session_state.curated_results
    
    st.subheader("🧠 Modeled Traveler Archetype")
    st.info(f"**{res.get('intent_archetype')}**: {res.get('context_analysis')}")
    
    st.subheader("🎯 Proactively Curated Destinations")
    
    for idx, dest in enumerate(res.get("destinations", [])):
        with st.container():
            st.markdown(
                f"""
                <div class="card-box">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h2 style="margin: 0; color: #38bdf8;">📍 {dest.get('name')}</h2>
                        <span class="badge">Match: {dest.get('match_score')}</span>
                    </div>
                    <p style="font-style: italic; color: #94a3b8; margin-top: 0.2rem;">"{dest.get('tagline')}"</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            c1, c2 = st.columns([2, 1])
            
            with c1:
                st.markdown(f"**Why This Fits Your Context:**\n{dest.get('why_it_fits')}")
                st.markdown(f"**⚓ Key Anchor Experience:**\n{dest.get('anchor_experience')}")
            
            with c2:
                st.markdown(f"**Seasonality:** `{dest.get('seasonality_status')}`")
                st.markdown(f"**Est. Cost:** `{dest.get('est_budget_breakdown')}`")
                st.markdown(f"**⚠️ Trade-offs:**\n{dest.get('trade_offs')}")
            
            # Interactive Feedback Loop to update Supabase
            with st.expander(f"💬 Rate this recommendation (Refines future memory for {user_name})"):
                fb_col1, fb_col2 = st.columns([3, 1])
                with fb_col1:
                    fb_notes = st.text_input(
                        "What do you like or dislike about this suggestion?",
                        key=f"fb_notes_{idx}",
                        placeholder="e.g., Love the quiet vibe, but flight is too long.",
                    )
                with fb_col2:
                    rx = st.selectbox("Action", ["Liked", "Disliked", "Saved"], key=f"rx_{idx}")
                    if st.button("Save Feedback", key=f"btn_{idx}"):
                        if engine.save_user_feedback(user_name, dest.get("name"), rx, fb_notes):
                            st.success("Preference logged into Supabase context memory!")

            st.markdown("---")

else:
    st.info(
        "👈 Configure your **Traveler Context Genome** in the sidebar and click **Proactively Surface Destinations** to generate hyper-personalized choices."
    )