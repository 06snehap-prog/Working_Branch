import asyncio
import urllib.parse
import traceback
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px

# ML IMPORTS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# MCP + LANGCHAIN IMPORTS
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_aws import ChatBedrockConverse
from langgraph.prebuilt import create_react_agent

# --- CONFIG ---
NPX_PATH = "/opt/homebrew/bin/npx"
#MODEL_ID = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
MODEL_ID = "us.amazon.nova-micro-v1:0"
SSO_PROFILE = "AdministratorAccess-715248855541"

DB_USER = "postgres"
DB_PASS = urllib.parse.quote_plus("Push@1245")
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "AGENTIC_DB"

DB_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- DB ENGINE ---
engine = create_engine(DB_URL)

# --- SCHEMA FETCH (CRITICAL FIX) ---
def get_schema():
    try:
        tables = ["iot_sensor_data", "sap_data"]
        schema_text = ""

        for table in tables:
            cols = pd.read_sql(f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = '{table}'
            """, engine)

            col_list = ", ".join(cols["column_name"].tolist())
            schema_text += f"\nTable {table}: {col_list}"

        return schema_text

    except Exception as e:
        return f"Schema fetch error: {e}"

SCHEMA = get_schema()

# --- STRICT SQL PROMPT ---
SYSTEM_PROMPT = f"""
You are a PostgreSQL SQL expert.

DATABASE SCHEMA:
{SCHEMA}

STRICT RULES:
- Only use tables: iot_sensor_data, sap_data
- Only use columns listed above
- NEVER assume columns (no status, health, etc.)
- Only generate SELECT queries
- Always LIMIT 50
- If invalid → return: Invalid query
"""

# --- UI ---
st.set_page_config(page_title="SAP Reliability AI", layout="wide")
st.title("🛡️ Enterprise Predictive Maintenance Agent")
st.markdown(f"Connected to `{DB_NAME}`")

# --- AGENT ---
async def run_mcp_agent(user_prompt):
    try:
        server_params = StdioServerParameters(
            command=NPX_PATH,
            args=["-y", "@modelcontextprotocol/server-postgres", DB_URL],
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)

                llm = ChatBedrockConverse(
                    model_id=MODEL_ID,
                    region_name="us-east-1",
                    credentials_profile_name=SSO_PROFILE,
                    temperature=0.0
                )

                agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)

                # ✅ NULL QUERY FIX
                if not user_prompt or user_prompt.strip() == "":
                    user_prompt = "Show first 10 rows from iot_sensor_data"

                try:
                    result = await agent.ainvoke(
                        {"messages": [("user", user_prompt)]}
                    )
                    return result["messages"][-1].content

                except Exception as agent_error:
                    return f"AGENT ERROR:\n{str(agent_error)}"

    except Exception as e:
        return f"MCP CONNECTION ERROR:\n{str(e)}"

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Equipment")
    try:
        ids_df = pd.read_sql("SELECT DISTINCT equipment_id FROM sap_data", engine)
        equipment_options = ["Fleet"] + ids_df["equipment_id"].tolist()
        selected_id = st.selectbox("Select Equipment", equipment_options)
    except Exception as e:
        st.error(e)
        selected_id = None

# --- DASHBOARD ---
st.subheader("📊 Equipment Health Dashboard")

try:
    df = pd.read_sql("SELECT * FROM iot_sensor_data", engine)

    # Auto column fix
    if "temperature" not in df.columns:
        for col in df.columns:
            if "temp" in col.lower():
                df["temperature"] = df[col]
                break

    if "vibration" not in df.columns:
        for col in df.columns:
            if "vibration" in col.lower():
                df["vibration"] = df[col]
                break

    if selected_id == "Fleet":
        eq_df = df.copy()
    else:
        eq_df = df[df["equipment_id"] == selected_id]

    if not eq_df.empty:
        avg_temp = eq_df["temperature"].mean()
        avg_vib = eq_df["vibration"].mean()

        if avg_temp > 80 or avg_vib > 80:
            status = "🔴 RED - Critical"
        elif avg_temp > 60 or avg_vib > 60:
            status = "🟡 YELLOW - Warning"
        else:
            status = "🟢 GREEN - Healthy"

        col1, col2, col3 = st.columns(3)
        col1.metric("Equipment", selected_id)
        col2.metric("Avg Temperature", round(avg_temp, 2))
        col3.metric("Avg Vibration", round(avg_vib, 2))

        st.markdown(f"### Status: {status}")
    else:
        st.warning("No data available")

except Exception as e:
    st.error(f"Dashboard Error: {e}")

# --- AI ---
st.subheader("💬 Ask AI")
user_input = st.text_input("Ask something about your equipment")

if st.button("Run AI"):
    prompt = user_input if user_input else f"Analyze equipment {selected_id}"
    with st.spinner("Running AI..."):
        response = asyncio.run(run_mcp_agent(prompt))
        st.subheader("🧠 Insights")
        st.code(response)

# --- HISTOGRAM ---
with st.expander("📈 Histogram"):
    if not df.empty:
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        col = st.selectbox("Select Column", numeric_cols)

        fig = px.histogram(
            df,
            x=col,
            color="equipment_id",
            hover_data=["equipment_id"],
            barmode="overlay"
        )
        st.plotly_chart(fig)

# --- ML ---
st.subheader("🤖 Algorithm Comparison")

try:
    if not eq_df.empty:
        threshold = eq_df['vibration'].median()
        eq_df['target'] = (eq_df['vibration'] > threshold).astype(int)

        if eq_df['target'].nunique() > 1:
            X = eq_df[["temperature", "vibration"]]
            y = eq_df["target"]

            X_train, X_test, y_train, y_test = train_test_split(X, y)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            models = {
                "XGBoost": XGBClassifier(eval_metric='logloss'),
                "RandomForest": RandomForestClassifier(),
                "LogisticRegression": LogisticRegression()
            }

            results = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                results[name] = accuracy_score(y_test, model.predict(X_test))

            st.table(pd.DataFrame.from_dict(results, orient='index', columns=["Accuracy"]))

except Exception as e:
    st.error(e)