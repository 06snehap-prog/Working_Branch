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
    page_title="BuildMind AI — Renovation Planner", page_icon="🏡", layout="wide"
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0f172a; color: #f8fafc; }
    .stButton>button { width: 100%; background-color: #2563eb; color: white; border-radius: 8px; font-weight: bold; border: none; padding: 0.65rem; }
    .stButton>button:hover { background-color: #1d4ed8; color: #ffffff; }
    .card-box { background-color: #1e293b; border-radius: 12px; padding: 1.25rem; border: 1px solid #334155; margin-bottom: 1rem; }
    .metric-card { background-color: #1e293b; padding: 1rem; border-radius: 10px; border-left: 4px solid #38bdf8; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem !important; color: #38bdf8; }
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

    # 2. Check environment variables first, then fallback to Streamlit secrets
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
# 3. RENOVATION PLANNER AGENT ENGINE
# ==========================================
class RenovationPlannerAgent:

    def __init__(self, db: Client, bedrock):
        self.db = db
        self.bedrock = bedrock
        self.model_id = MODEL_ID

    def save_project_to_db(self, project_data: dict):
        """Persists generated renovation plans into Supabase."""
        try:
            self.db.table("renovation_projects").insert(project_data).execute()
            return True
        except Exception:
            return False

    def fetch_saved_projects(self, homeowner_name: str):
        """Retrieves past renovation roadmaps from Supabase."""
        try:
            res = (
                self.db.table("renovation_projects")
                .select("*")
                .eq("homeowner_name", homeowner_name)
                .order("created_at", desc=True)
                .execute()
            )
            return res.data or []
        except Exception:
            return []

    def generate_renovation_roadmap(self, context: dict) -> dict:
        system_prompt = (
            "You are an AI Master Construction Project Manager & Renovation Architect. "
            "Model project dependencies, sequencing, material lead times, permit risks, and budgets. "
            "Respond ONLY with a single valid JSON object (no markdown formatting, no backticks, no prose):\n"
            "{\n"
            '  "project_title": "Descriptive Project Title",\n'
            '  "contingency_buffer_pct": 15,\n'
            '  "recommended_buffer_usd": 4500,\n'
            '  "phases": [\n'
            "     {\n"
            '       "step": 1,\n'
            '       "phase_name": "Phase Name",\n'
            '       "contractor_trade": "General / Electrician / Plumber",\n'
            '       "duration_weeks": 2,\n'
            '       "estimated_cost_usd": 3500,\n'
            '       "prerequisite_tasks": "Permit Approval & Demo",\n'
            '       "alignment_deliverable": "Inspection sign-off on rough wiring"\n'
            "     }\n"
            "  ],\n"
            '  "risk_matrix": [\n'
            "     {\n"
            '       "risk_category": "Permit / Material / Trade Collision / Budget",\n'
            '       "risk_description": "Description of potential bottleneck",\n'
            '       "mitigation_strategy": "Actionable preventative step"\n'
            "     }\n"
            "  ],\n"
            '  "alignment_protocol": "Key rules to keep homeowners and contractors synced"\n'
            "}"
        )

        user_content = f"""
        Renovation Context:
        - Homeowner: {context['homeowner_name']}
        - Scope: {context['scope']}
        - Property Age: {context['property_age']}
        - Total Allocated Budget: ${context['budget_usd']} USD
        - Target Duration: {context['target_weeks']} Weeks
        - DIY vs Pro Ratio: {context['labor_type']}
        - Special Constraints / Notes: "{context['notes']}"
        """

        try:
            response = self.bedrock.converse(
                modelId=self.model_id,
                system=[{"text": system_prompt}],
                messages=[{"role": "user", "content": [{"text": user_content}]}],
                inferenceConfig={"maxTokens": 1800, "temperature": 0.3},
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
                "project_title": "Standard Space Remodel",
                "contingency_buffer_pct": 15,
                "recommended_buffer_usd": context["budget_usd"] * 0.15,
                "phases": [
                    {
                        "step": 1,
                        "phase_name": "Demolition & Site Prep",
                        "contractor_trade": "General Contractor",
                        "duration_weeks": 1,
                        "estimated_cost_usd": 2000,
                        "prerequisite_tasks": "Permits Secured",
                        "alignment_deliverable": "Clean site check",
                    }
                ],
                "risk_matrix": [
                    {
                        "risk_category": "Permit Delay",
                        "risk_description": "City inspection backlogs.",
                        "mitigation_strategy": "Submit plans 4 weeks ahead.",
                    }
                ],
                "alignment_protocol": "Weekly Friday standups with contractor.",
            }


agent = RenovationPlannerAgent(supabase, bedrock_runtime)

# Initialize Session State
if "active_renovation_plan" not in st.session_state:
    st.session_state.active_renovation_plan = None

# ==========================================
# 4. USER INTERFACE & DASHBOARD
# ==========================================
st.title("🏡 BuildMind AI — Home Renovation Planner")
st.caption(
    "AI-Native Project Management: Sequencing, Budget Guardrails & Risk Prevention"
)

st.markdown("---")

# SIDEBAR: PROJECT SETUP
with st.sidebar:
    st.header("⚙️ Project Specifications")

    homeowner_name = st.text_input("Homeowner Name", value="Alex Rivera")

    scope = st.selectbox(
        "Renovation Scope",
        [
            "Full Kitchen Remodel & Layout Change",
            "Master Bathroom & Wet Room Overhaul",
            "Basement Conversion to Living Suite",
            "Whole Home Interior & Mechanical Update",
            "Outdoor Living Space, Deck & Kitchen",
        ],
    )

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        budget_usd = st.number_input(
            "Budget ($)",
            min_value=5000,
            max_value=250000,
            value=35000,
            step=2500,
        )
    with col_s2:
        target_weeks = st.number_input(
            "Target (Wks)", min_value=2, max_value=52, value=8
        )

    property_age = st.selectbox(
        "Property Age",
        [
            "Newer Construction (< 10 yrs)",
            "Mid-Age (10-30 yrs)",
            "Older Home (30-70 yrs)",
            "Historic (> 70 yrs)",
        ],
    )

    labor_type = st.select_slider(
        "Contractor Execution Model",
        options=[
            "100% General Contractor",
            "Hybrid (GC + Subbed Direct)",
            "DIY Heavy + GC Oversight",
        ],
        value="100% General Contractor",
    )

    notes = st.text_area(
        "Specific Constraints / Concerns",
        placeholder="e.g. Load-bearing wall removal required, custom Italian tiles ordered with 6-week lead time.",
    )

    generate_btn = st.button("🚀 Build Renovation Roadmap")

# METRICS DISPLAY
saved_projects = agent.fetch_saved_projects(homeowner_name)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Homeowner Profile", homeowner_name)
m2.metric("Saved Projects", len(saved_projects))
m3.metric("Allocated Budget", f"${budget_usd:,}")
m4.metric("Target Timeline", f"{target_weeks} Weeks")

st.markdown("---")

# GENERATION TRIGGER
if generate_btn:
    with st.spinner(
        "AI Project Manager is mapping trade dependencies & analyzing risk vectors..."
    ):
        ctx = {
            "homeowner_name": homeowner_name,
            "scope": scope,
            "property_age": property_age,
            "budget_usd": budget_usd,
            "target_weeks": target_weeks,
            "labor_type": labor_type,
            "notes": notes,
        }
        plan = agent.generate_renovation_roadmap(ctx)
        st.session_state.active_renovation_plan = plan

        # Save to DB
        db_payload = {
            "homeowner_name": homeowner_name,
            "project_title": plan.get("project_title", scope),
            "scope": scope,
            "budget_usd": budget_usd,
            "plan_json": json.dumps(plan),
        }
        agent.save_project_to_db(db_payload)

# MAIN DASHBOARD CONTENT
if st.session_state.active_renovation_plan:
    p = st.session_state.active_renovation_plan

    st.subheader(f"📋 Project Roadmap: {p.get('project_title')}")

    # Financial Contingency Recommendation
    c_buffer = p.get("recommended_buffer_usd", budget_usd * 0.15)
    effective_budget = budget_usd - c_buffer

    bc1, bc2, bc3 = st.columns(3)
    bc1.metric("Total Project Fund", f"${budget_usd:,}")
    bc2.metric(
        "Recommended Risk Reserve",
        f"${c_buffer:,.0f}",
        f"{p.get('contingency_buffer_pct', 15)}% Buffer",
    )
    bc3.metric("Working Build Budget", f"${effective_budget:,.0f}")

    st.markdown("### 🔄 Task Sequencing & Dependency Map")

    phases = p.get("phases", [])
    if phases:
        df_phases = pd.DataFrame(phases)
        # Format table columns for clean presentation
        df_display = df_phases[
            [
                "step",
                "phase_name",
                "contractor_trade",
                "duration_weeks",
                "estimated_cost_usd",
                "prerequisite_tasks",
                "alignment_deliverable",
            ]
        ].rename(
            columns={
                "step": "Step",
                "phase_name": "Phase",
                "contractor_trade": "Trade",
                "duration_weeks": "Duration (Wks)",
                "estimated_cost_usd": "Est. Cost ($)",
                "prerequisite_tasks": "Prerequisites",
                "alignment_deliverable": "Sign-Off Criteria",
            }
        )
        st.dataframe(df_display, use_container_width=True, hide_index=True)

    st.markdown("---")

    col_r1, col_r2 = st.columns([1, 1])

    with col_r1:
        st.markdown("### 🚨 AI Risk Detector & Mitigation")
        for r in p.get("risk_matrix", []):
            with st.container():
                st.markdown(
                    f"**[{r.get('risk_category')}] {r.get('risk_description')}**"
                )
                st.info(f"💡 **Mitigation:** {r.get('mitigation_strategy')}")

    with col_r2:
        st.markdown("### 🤝 Contractor-Homeowner Alignment Protocol")
        st.success(
            p.get(
                "alignment_protocol",
                "Establish bi-weekly walk-throughs before payment milestones.",
            )
        )

else:
    st.info(
        "👈 Enter project scope in the sidebar and click **Build Renovation Roadmap** to generate your AI-managed build plan."
    )