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
    page_title="AI Fitness Coach", page_icon="🏋️‍♂️", layout="wide"
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0f172a; color: #f8fafc; }
    .stButton>button { width: 100%; background-color: #2563eb; color: white; border-radius: 8px; font-weight: bold; }
    .stButton>button:hover { background-color: #1d4ed8; }
    div[data-testid="stMetricValue"] { font-size: 2rem !important; color: #38bdf8; }
    </style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# 2. CONFIGURATION & CLIENT INITIALIZATION
# ==========================================
REGION = "us-east-1"
#MODEL_ID = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
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
# 3. AI FITNESS COACH AGENT CLASS
# ==========================================
class AIFitnessCoachAgent:

    def __init__(self, db: Client, bedrock_client):
        self.db = db
        self.bedrock = bedrock_client
        self.model_id = MODEL_ID

    def get_profiles(self):
        try:
            res = self.db.table("profiles").select("*").execute()
            return res.data or []
        except Exception as e:
            st.error(f"Database error fetching profiles: {e}")
            return []

    def create_or_update_profile(
        self,
        full_name: str,
        age: int,
        weight: float,
        height: float,
        primary_goal: str,
        experience_level: str,
    ):
        try:
            existing = (
                self.db.table("profiles")
                .select("*")
                .eq("full_name", full_name)
                .execute()
            )

            payload = {
                "full_name": full_name,
                "age": age,
                "weight": weight,
                "height": height,
                "primary_goal": primary_goal,
                "experience_level": experience_level,
            }

            if existing.data:
                res = (
                    self.db.table("profiles")
                    .update(payload)
                    .eq("id", existing.data[0]["id"])
                    .execute()
                )
            else:
                res = self.db.table("profiles").insert(payload).execute()

            return res.data[0] if res.data else None
        except Exception as e:
            st.error(f"Failed to save profile: {e}")
            return None

    def fetch_user_history_and_kpis(self, user_id: int):
        try:
            res = (
                self.db.table("generated_workouts")
                .select("*")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .execute()
            )
            data = res.data or []

            if not data:
                return {
                    "total_workouts": 0,
                    "avg_energy": 0.0,
                    "active_days": 0,
                    "consistency_rate": "0%",
                    "history_df": pd.DataFrame(),
                }

            df = pd.DataFrame(data)

            if "created_at" in df.columns:
                df["date"] = pd.to_datetime(df["created_at"]).dt.date
            else:
                df["date"] = pd.Timestamp.now().date()

            total_workouts = len(df)
            active_days = df["date"].nunique()

            ctx_res = (
                self.db.table("daily_user_context")
                .select("energy_level")
                .eq("user_id", user_id)
                .execute()
            )
            ctx_data = ctx_res.data or []

            avg_energy = 0.0
            if ctx_data:
                energies = [
                    c["energy_level"]
                    for c in ctx_data
                    if c.get("energy_level") is not None
                ]
                if energies:
                    avg_energy = round(sum(energies) / len(energies), 1)

            consistency_rate = f"{min(round((active_days / 30) * 100), 100)}%"

            return {
                "total_workouts": total_workouts,
                "avg_energy": avg_energy,
                "active_days": active_days,
                "consistency_rate": consistency_rate,
                "history_df": df,
            }
        except Exception as e:
            st.error(f"Error fetching analytics: {e}")
            return {
                "total_workouts": 0,
                "avg_energy": 0.0,
                "active_days": 0,
                "consistency_rate": "0%",
                "history_df": pd.DataFrame(),
            }

    def log_daily_context(
        self,
        user_id: int,
        energy: int,
        time_avail: int,
        location: str,
        is_injured: bool,
        notes: str,
    ):
        try:
            payload = {
                "user_id": user_id,
                "energy_level": energy,
                "time_available_mins": time_avail,
                "current_location": location,
                "is_injured_or_sick": is_injured,
                "user_notes": notes,
            }
            res = (
                self.db.table("daily_user_context").insert(payload).execute()
            )
            return res.data[0] if res.data else None
        except Exception as e:
            st.error(f"Failed to log context: {e}")
            return None

    def generate_adaptive_plan(
        self,
        profile: dict,
        energy: int,
        time_avail: int,
        location: str,
        is_injured: bool,
        notes: str = "",
    ):
        """Pure LLM Decision Engine: Claude evaluates all safety, energy, goals, and conditions."""
        goal = (
            profile.get("primary_goal")
            or profile.get("fitness_goal")
            or "General Fitness"
        )
        name = profile.get("full_name", "Athlete")
        experience = profile.get("experience_level", "Beginner")
        age = profile.get("age", 25)
        weight = profile.get("weight", 70)
        height = profile.get("height", 170)

        system_prompt = (
            "You are an expert AI Strength & Conditioning Coach and Sports Medicine Specialist. "
            "You have complete decision-making authority for prescribing workouts or recommending rest/medical safety protocols. "
            "Evaluate the user's physical profile, energy level, injury status, and open text notes. "
            "If the user notes indicate severe illness, chest pain, dizziness, or serious medical concern, recommend rest/medical consultation in the workout response. "
            "You MUST respond ONLY with a single valid JSON object (no markdown, no extra prose) matching this format:\n"
            "{\n"
            '  "workout_name": "Title of Workout or Safety Rest Routine",\n'
            '  "workout_type": "Goal Focus e.g. Mobility, Recovery, Hypertrophy, Safety Hold",\n'
            '  "intensity": "Low / Moderate / High / Rest & Recovery",\n'
            '  "exercise_list": ["Exercise 1 or instruction", "Exercise 2...", ...],\n'
            '  "explainability_reasoning": "Reasoning based on energy, symptoms, and goal",\n'
            '  "personalized_motivation": "Addressing the user directly by name",\n'
            '  "trade_off_note": "Explanation of intensity/volume trade-offs made"\n'
            "}"
        )

        user_content = f"""
        User Details:
        - Name: {name} | Age: {age} | Weight: {weight}kg | Height: {height}cm
        - Primary Goal: {goal}
        - Experience Level: {experience}

        Real-Time Input:
        - Energy Level: {energy}/10
        - Available Time: {time_avail} mins
        - Location: {location}
        - Injured / Sick: {is_injured}
        - Additional User Notes: "{notes}"

        Decide the most suitable routine or recovery protocol for today.
        """

        try:
            response = self.bedrock.converse(
                modelId=self.model_id,
                system=[{"text": system_prompt}],
                messages=[
                    {"role": "user", "content": [{"text": user_content}]}
                ],
                inferenceConfig={"maxTokens": 1000, "temperature": 0.3},
            )

            response_text = response["output"]["message"]["content"][0][
                "text"
            ].strip()

            if response_text.startswith("```"):
                response_text = (
                    response_text.strip("`").removeprefix("json").strip()
                )

            plan = json.loads(response_text)
            plan["user_id"] = profile["id"]
            return plan

        except Exception as e:
            st.error(f"LLM Generation Error: {e}")
            return {
                "user_id": profile["id"],
                "workout_name": "Adaptive Baseline Session",
                "workout_type": goal,
                "exercise_list": [
                    "5 mins Warmup",
                    "3 sets x 10 Bodyweight Squats",
                    "3 sets x 8 Push-ups",
                    "5 mins Stretch",
                ],
                "explainability_reasoning": "Fallback plan generated due to service timeout.",
                "personalized_motivation": f"Keep moving forward, {name}!",
                "trade_off_note": "Standard baseline template applied.",
                "intensity": "Moderate",
            }

    def generate_chat_response(
        self, user_prompt: str, user_profile: dict
    ) -> str:
        try:
            system_prompt = (
                f"You are an encouraging AI Fitness Coach conversing with {user_profile.get('full_name', 'your athlete')}. "
                f"Provide helpful, concise (2-3 sentences), and science-backed responses."
            )

            response = self.bedrock.converse(
                modelId=self.model_id,
                system=[{"text": system_prompt}],
                messages=[
                    {"role": "user", "content": [{"text": user_prompt}]}
                ],
                inferenceConfig={"maxTokens": 250, "temperature": 0.7},
            )
            return response["output"]["message"]["content"][0]["text"]
        except Exception as e:
            return f"I'm here to help you reach your goals! (Note: {e})"

    def save_workout(self, plan: dict):
        try:
            payload = {
                "user_id": plan["user_id"],
                "workout_name": plan["workout_name"],
                "workout_type": plan["workout_type"],
                "exercise_list": plan["exercise_list"],
                "explainability_reasoning": plan["explainability_reasoning"],
                "personalized_motivation": plan["personalized_motivation"],
            }
            res = (
                self.db.table("generated_workouts").insert(payload).execute()
            )
            return True
        except Exception as e:
            st.error(f"Failed to record workout: {e}")
            return False


agent = AIFitnessCoachAgent(supabase, bedrock_runtime)

# Session State Initializations
if "logged_in_user" not in st.session_state:
    st.session_state.logged_in_user = None
if "current_plan" not in st.session_state:
    st.session_state.current_plan = None

# ------------------------------------------
# SCREEN 1: LOGIN & PROFILE REGISTRATION
# ------------------------------------------
if not st.session_state.logged_in_user:
    st.title("🏋️‍♂️ AI Fitness Portal")

    tabs = st.tabs(["🔒 Existing Profile", "📝 New Registration"])

    with tabs[0]:
        profiles = agent.get_profiles()
        if profiles:
            user_options = {p["full_name"]: p for p in profiles}
            selected_name = st.selectbox("Select User", list(user_options.keys()))
            if st.button("Login"):
                st.session_state.logged_in_user = user_options[selected_name]
                st.rerun()
        else:
            st.info("No profiles registered yet.")

    with tabs[1]:
        with st.form("reg_form"):
            full_name = st.text_input("Full Name *", placeholder="e.g. Alex Rivera")
            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.number_input("Age", min_value=12, max_value=100, value=25)
            with c2:
                weight = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=70.0)
            with c3:
                height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)

            primary_goal = st.selectbox(
                "Primary Goal",
                [
                    "Habit Consistency & Energy",
                    "Muscle & Strength Gain",
                    "Fat Loss",
                    "Endurance Focus",
                ],
            )
            experience_level = st.selectbox(
                "Experience Level", ["Beginner", "Intermediate", "Advanced"]
            )

            if st.form_submit_button("Save Profile"):
                if full_name.strip():
                    u = agent.create_or_update_profile(
                        full_name, age, weight, height, primary_goal, experience_level
                    )
                    if u:
                        st.session_state.logged_in_user = u
                        st.rerun()

# ------------------------------------------
# SCREEN 2: MAIN STRUCTURED DASHBOARD
# ------------------------------------------
else:
    user = st.session_state.logged_in_user

    # HEADER BAR
    head_col, out_col = st.columns([5, 1])
    with head_col:
        st.title(f"🏃‍♂️ Athlete Dashboard: {user['full_name']}")
        st.caption(
            f"Age: {user.get('age', 'N/A')} | Weight: {user.get('weight', 'N/A')} kg | "
            f"Height: {user.get('height', 'N/A')} cm | Goal: {user.get('primary_goal', 'General Fitness')}"
        )
    with out_col:
        if st.button("🚪 Logout"):
            st.session_state.logged_in_user = None
            st.session_state.current_plan = None
            st.rerun()

    st.markdown("---")

    # SIDEBAR: CONTROLS & CONTEXT
    with st.sidebar:
        st.header("⚙️ Real-Time Context")
        energy_level = st.slider("Energy Level (1-10)", 1, 10, 7)
        time_available = st.select_slider(
            "Time Available (Mins)", options=[10, 15, 30, 45, 60], value=30
        )
        location = st.selectbox("Location", ["Gym", "Home", "Hotel / Travel"])
        is_injured = st.checkbox("Feeling Sick or Injured?")
        notes = st.text_input(
            "Notes (symptoms, pain, busyness)", placeholder="e.g. sore shoulder"
        )

        generate_btn = st.button("🚀 Generate New Plan")

    # SECTION 1: TOP KPI METRICS
    analytics = agent.fetch_user_history_and_kpis(user["id"])

    with st.container(border=True):
        st.subheader("📊 Key Performance Indicators")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Workouts", analytics["total_workouts"])
        k2.metric("30-Day Consistency", analytics["consistency_rate"])
        k3.metric("Active Days Logged", analytics["active_days"])
        k4.metric("Avg Readiness", f"{analytics['avg_energy']}/10")

    st.markdown("---")

    # SECTION 2: WORKOUT GENERATOR & DAILY PLAN
    if generate_btn:
        with st.spinner("Claude Sonnet is tailoring your plan..."):
            agent.log_daily_context(
                user["id"], energy_level, time_available, location, is_injured, notes
            )
            plan = agent.generate_adaptive_plan(
                user, energy_level, time_available, location, is_injured, notes
            )
            agent.save_workout(plan)
            st.session_state.current_plan = plan
            st.rerun()

    # SECTION 3: TWO-COLUMN MAIN VIEW (PLAN & HISTORY / CHAT)
    col_left, col_right = st.columns([1, 1])

    with col_left:
        with st.container(border=True):
            st.subheader("📋 Active Workout Plan")
            if st.session_state.current_plan:
                p = st.session_state.current_plan
                st.markdown(f"### {p.get('workout_name')}")
                st.write(
                    f"**Type:** {p.get('workout_type')} | **Intensity:** {p.get('intensity')}"
                )
                st.info(f"💡 {p.get('personalized_motivation')}")

                st.markdown("**Routine:**")
                for item in p.get("exercise_list", []):
                    st.markdown(f"- {item}")

                with st.expander("🔍 AI Decision Rationale & Safety"):
                    st.write(f"**Reasoning:** {p.get('explainability_reasoning')}")
                    st.write(f"**Trade-off:** {p.get('trade_off_note')}")
            else:
                st.info(
                    "👈 Adjust parameters in the sidebar and click **Generate New Plan** to start today's session."
                )

    with col_right:
        dash_tabs = st.tabs(["💬 AI Coach Chat", "📅 Activity History"])

        with dash_tabs[0]:
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = [
                    {
                        "role": "assistant",
                        "content": f"Hi {user['full_name']}, ready to train today?",
                    }
                ]

            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

            chat_in = st.chat_input("Ask a question about your routine...")
            if chat_in:
                st.session_state.chat_history.append(
                    {"role": "user", "content": chat_in}
                )
                with st.chat_message("user"):
                    st.write(chat_in)

                with st.chat_message("assistant"):
                    ans = agent.generate_chat_response(chat_in, user)
                    st.write(ans)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": ans}
                    )

        with dash_tabs[1]:
            df_hist = analytics["history_df"]
            if not df_hist.empty:
                # Group by date to keep history unique in view
                unique_days = (
                    df_hist.groupby("date")
                    .agg({"workout_name": "last", "workout_type": "last"})
                    .reset_index()
                )
                st.dataframe(
                    unique_days,
                    column_config={
                        "date": "Date",
                        "workout_name": "Workout Prescribed",
                        "workout_type": "Focus",
                    },
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("No history logged yet.")