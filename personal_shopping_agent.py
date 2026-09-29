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
    page_title="StyleMind AI — Personal Shopping Agent",
    page_icon="🛍️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0d1117; color: #f0f6fc; }
    .stButton>button { width: 100%; background-color: #8b5cf6; color: white; border-radius: 8px; font-weight: bold; border: none; padding: 0.65rem; }
    .stButton>button:hover { background-color: #7c3aed; color: #ffffff; }
    .card-box { background-color: #161b22; border-radius: 12px; padding: 1.25rem; border: 1px solid #30363d; margin-bottom: 1rem; }
    .badge-match { background-color: #059669; color: white; padding: 0.25rem 0.6rem; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
    .badge-tradeoff { background-color: #d97706; color: white; padding: 0.25rem 0.6rem; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem !important; color: #a78bfa; }
    </style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# 2. CONFIGURATION & CLIENT INITIALIZATION
# ==========================================
REGION = "us-east-1"
MODEL_ID = "us.amazon.nova-micro-v1:0"
SSO_PROFILE = "AdministratorAccess-715248855541"


def get_secret(key_name: str, default=None):
    """Safely fetch secrets without throwing StreamlitSecretNotFoundError locally."""
    try:
        return st.secrets.get(key_name, default)
    except Exception:
        return default


@st.cache_resource
def get_supabase_client() -> Client:
    # 1. Load local .env first if available
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)

    # 2. Check environment variables first, then Streamlit secrets
    supabase_url = os.getenv("SUPABASE_URL") or get_secret("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY") or get_secret("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        st.error(
            "Missing SUPABASE_URL or SUPABASE_KEY in .env or Streamlit Secrets."
        )
        st.stop()

    return create_client(supabase_url, supabase_key)


@st.cache_resource
def get_bedrock_client():
    # Streamlit Cloud deployment check via secrets
    aws_secrets = get_secret("aws")
    if aws_secrets:
        return boto3.client(
            "bedrock-runtime",
            aws_access_key_id=aws_secrets["aws_access_key_id"],
            aws_secret_access_key=aws_secrets["aws_secret_access_key"],
            region_name=aws_secrets.get("aws_default_region", REGION),
        )

    # Local development fallback using SSO Profile or standard AWS CLI auth
    try:
        session = boto3.Session(profile_name=SSO_PROFILE, region_name=REGION)
        return session.client("bedrock-runtime")
    except Exception:
        return boto3.client("bedrock-runtime", region_name=REGION)


supabase = get_supabase_client()
bedrock_runtime = get_bedrock_client()

# ==========================================
# 3. PERSONAL SHOPPING AGENT ENGINE
# ==========================================
class PersonalShoppingAgent:

    def __init__(self, db: Client, bedrock):
        self.db = db
        self.bedrock = bedrock
        self.model_id = MODEL_ID

    def fetch_user_shopping_memory(self, shopper_name: str):
        """Fetches historical feedback to enable long-term preference learning."""
        try:
            res = (
                self.db.table("shopping_feedback")
                .select("*")
                .eq("shopper_name", shopper_name)
                .order("created_at", desc=True)
                .limit(5)
                .execute()
            )
            return res.data or []
        except Exception:
            return []

    def save_shopping_feedback(self, shopper_name: str, item_name: str, reaction: str, notes: str):
        """Saves explicitly learned preferences into Supabase memory."""
        try:
            payload = {
                "shopper_name": shopper_name,
                "item_name": item_name,
                "reaction": reaction,
                "user_notes": notes,
            }
            self.db.table("shopping_feedback").insert(payload).execute()
            return True
        except Exception as e:
            st.error(f"Failed to save feedback: {e}")
            return False

    def generate_curated_decisions(self, context: dict) -> dict:
        """Translates vague shopping intent into a small, highly explainable decision set."""
        system_prompt = (
            "You are an Unbiased AI Personal Shopping & Decision Assistant. "
            "Your objective is to solve choice overload by curating exactly 3 highly relevant options "
            "and explaining trade-offs neutrally without sponsored bias. "
            "Respond ONLY with a single valid JSON object (no markdown, no backticks, no extra prose):\n"
            "{\n"
            '  "intent_archetype": "Modeled Shopper Intent (e.g. Resort Casual with High Breathability)",\n'
            '  "decision_framework_summary": "1-2 sentence evaluation strategy based on constraints",\n'
            '  "curated_items": [\n'
            "     {\n"
            '       "item_name": "Product Name & Brand",\n'
            '       "category": "Outfit / Footwear / Accessory",\n'
            '       "price_usd": 120,\n'
            '       "confidence_score": "94%",\n'
            '       "why_it_fits": "Specific alignment with user intent and weather/vibe",\n'
            '       "key_trade_off": "Transparent downside (e.g. Dry clean only, higher price, delicate fabric)",\n'
            '       "durability_and_quality": "High / Medium / Premium summary"\n'
            "     }\n"
            "  ]\n"
            "}"
        )

        user_content = f"""
        Shopper Context Genome:
        - Shopper Name: {context['shopper_name']}
        - Vague Intent Request: "{context['raw_intent']}"
        - Occasion / Destination: {context['occasion']}
        - Target Style Vibe: {context['style_vibe']}
        - Maximum Total Budget: ${context['budget_usd']} USD
        - Priority Trade-off Focus: {context['priority_focus']}
        - Past Learned Preferences: {context['past_memory']}
        """

        try:
            response = self.bedrock.converse(
                modelId=self.model_id,
                system=[{"text": system_prompt}],
                messages=[{"role": "user", "content": [{"text": user_content}]}],
                inferenceConfig={"maxTokens": 1600, "temperature": 0.3},
            )

            raw_text = response["output"]["message"]["content"][0]["text"].strip()
            if raw_text.startswith("```"):
                raw_text = (
                    raw_text.replace("```json", "").replace("```", "").strip()
                )

            return json.loads(raw_text)
        except Exception as e:
            st.error(f"Engine Generation Error: {e}")
            return {
                "intent_archetype": "Versatile Beach & Evening Look",
                "decision_framework_summary": "Prioritized breathable fabrics and multi-purpose styling.",
                "curated_items": [
                    {
                        "item_name": "Linen Relaxed-Fit Shirt & Shorts Set",
                        "category": "Apparel",
                        "price_usd": 85,
                        "confidence_score": "92%",
                        "why_it_fits": "Ultra-breathable 100% linen ideal for warm coastal weather.",
                        "key_trade_off": "Prone to natural creasing during travel.",
                        "durability_and_quality": "Medium - Requires gentle cold washing.",
                    }
                ],
            }


agent = PersonalShoppingAgent(supabase, bedrock_runtime)

# Initialize Session State
if "shopping_decisions" not in st.session_state:
    st.session_state.shopping_decisions = None

# ==========================================
# 4. USER INTERFACE & DASHBOARD
# ==========================================
st.title("🛍️ StyleMind AI — Intent-Based Personal Shopping")
st.caption(
    "Decision-Driven E-Commerce: Shifting from Endless Search to High-Confidence Selections"
)

st.markdown("---")

# SIDEBAR: SHOPPER CONTEXT GENOME
with st.sidebar:
    st.header("🧬 Shopper Intent Genome")

    shopper_name = st.text_input("Shopper Profile", value="Alex Rivera")

    raw_intent = st.text_area(
        "Vague Goal / Intent",
        value="I need outfits for a 5-day Goa trip with sunset beach parties and casual daytime exploring.",
        placeholder="e.g. I need comfortable business casual clothing for humid summer conference calls.",
    )

    occasion = st.selectbox(
        "Occasion & Setting",
        [
            "Beach & Coastal Vacation",
            "Business Casual Workwear",
            "Outdoor Adventure & Hiking",
            "Formal Evening Event / Wedding",
            "Capsule Wardrobe Refresh",
        ],
    )

    style_vibe = st.selectbox(
        "Preferred Style Vibe",
        [
            "Minimalist & Functional",
            "Boho & Relaxed Resortwear",
            "Smart Executive & Polished",
            "Trendy & High-Street",
        ],
    )

    budget_usd = st.slider("Total Budget Cap ($)", 50, 2000, 450, step=25)

    priority_focus = st.radio(
        "Decision Priority (Trade-Off Shield)",
        [
            "Maximize Comfort & Fabric Breathability",
            "Maximize Versatility (Rewearability)",
            "Maximize Premium Quality & Material Longevity",
            "Maximize Value for Money",
        ],
    )

    curate_btn = st.button("✨ Curate Decision Set (Max 3)")

# METRICS DISPLAY
memory_logs = agent.fetch_user_shopping_memory(shopper_name)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Shopper Profile", shopper_name)
m2.metric("Learned Preference Items", len(memory_logs))
m3.metric("Budget Limit", f"${budget_usd}")
m4.metric("Target Occasion", occasion.split(" ")[0])

st.markdown("---")

# GENERATION TRIGGER
if curate_btn:
    with st.spinner("AI Shopping Agent is evaluating trade-offs & eliminating noise..."):
        ctx = {
            "shopper_name": shopper_name,
            "raw_intent": raw_intent,
            "occasion": occasion,
            "style_vibe": style_vibe,
            "budget_usd": budget_usd,
            "priority_focus": priority_focus,
            "past_memory": memory_logs,
        }
        decisions = agent.generate_curated_decisions(ctx)
        st.session_state.shopping_decisions = decisions

# MAIN DASHBOARD CONTENT
if st.session_state.shopping_decisions:
    res = st.session_state.shopping_decisions

    st.subheader("🎯 Modeled Decision Strategy")
    st.info(f"**Archetype:** {res.get('intent_archetype')}\n\n**Framework:** {res.get('decision_framework_summary')}")

    st.subheader("📦 Curated High-Confidence Selections")

    for idx, item in enumerate(res.get("curated_items", [])):
        with st.container():
            st.markdown(
                f"""
                <div class="card-box">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h3 style="margin: 0; color: #c084fc;">🏷️ {item.get('item_name')}</h3>
                        <div>
                            <span class="badge-match">Match: {item.get('confidence_score')}</span>
                            <span style="font-weight: bold; margin-left: 0.5rem; color: #34d399;">${item.get('price_usd')}</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            c1, c2 = st.columns([2, 1])

            with c1:
                st.markdown(f"**Why This Fits Your Intent:**\n{item.get('why_it_fits')}")
                st.markdown(f"**Fabric & Build Quality:** `{item.get('durability_and_quality')}`")

            with c2:
                st.markdown(f"**⚠️ Transparent Trade-Off:**\n{item.get('key_trade_off')}")

            # Interactive Preference Feedback Loop to update Supabase
            with st.expander(f"💬 Save Preference / Feedback for {item.get('item_name')}"):
                fb_col1, fb_col2 = st.columns([3, 1])
                with fb_col1:
                    fb_notes = st.text_input(
                        "What do you like or dislike about this option?",
                        key=f"shop_notes_{idx}",
                        placeholder="e.g. Perfect style, but I prefer a brighter color palette.",
                    )
                with fb_col2:
                    rx = st.selectbox("Action", ["Liked", "Disliked", "Shortlisted"], key=f"shop_rx_{idx}")
                    if st.button("Save Preference", key=f"shop_btn_{idx}"):
                        if agent.save_shopping_feedback(
                            shopper_name, item.get("item_name"), rx, fb_notes
                        ):
                            st.success("Preference recorded into Supabase shopping memory!")

            st.markdown("---")

else:
    st.info(
        "👈 Define your vague shopping intent in the sidebar and click **Curate Decision Set** to see your high-confidence recommendations."
    )