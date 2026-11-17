import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Dissertation: Predictive Maintenance UI", layout="wide")

class ReportingAgent:
    def __init__(self):
        self.model_path = 'maintenance_xgboost_model.pkl'
        self.scaler_path = 'data_scaler.pkl'
        self.features_path = 'feature_names.pkl'
        self.data_path = 'sap_700_records.csv'

    def load_artifacts(self):
        try:
            with open(self.model_path, 'rb') as f: model = pickle.load(f)
            with open(self.scaler_path, 'rb') as f: scaler = pickle.load(f)
            with open(self.features_path, 'rb') as f: features = pickle.load(f)
            return model, scaler, features
        except FileNotFoundError:
            st.error("⚠️ ML Brain not found! Run dissertation.py first.")
            return None, None, None

    def run_dashboard(self):
        st.title("🛠️ Equipment Health Intelligence System")
        
        model, scaler, features = self.load_artifacts()
        if not model: return

        if os.path.exists(self.data_path):
            df = pd.read_csv(self.data_path)
            df.columns = df.columns.str.strip()
            
            # --- PREDICTION LOGIC ---
            # Get latest state for all equipment
            latest_status = df.sort_values('Timestamp').groupby('Equipment_ID').tail(1).copy()
            X = latest_status[features].apply(pd.to_numeric, errors='coerce').fillna(0)
            X_scaled = scaler.transform(X)
            probs = model.predict_proba(X_scaled)[:, 1]
            latest_status['Risk_Score'] = [round(float(p) * 100, 2) for p in probs]

            # --- SIDEBAR FILTERING ---
            st.sidebar.header("🔍 Filter Options")
            all_equipment = sorted(latest_status['Equipment_ID'].unique())
            
            # Option to see all or one
            selection = st.sidebar.selectbox("Select Equipment View", ["Entire Fleet"] + all_equipment)

            if selection == "Entire Fleet":
                self.show_fleet_view(latest_status)
            else:
                self.show_single_equipment_view(latest_status, selection)
        else:
            st.error(f"File {self.data_path} not found.")

    def show_fleet_view(self, data):
        st.subheader("🌐 Global Fleet Status")
        
        # Summary Metrics
        avg_risk = data['Risk_Score'].mean()
        critical_count = len(data[data['Risk_Score'] > 75])
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Equipment", len(data))
        m2.metric("Avg Fleet Risk", f"{avg_risk:.1f}%")
        m3.metric("Critical Alerts", critical_count, delta_color="inverse")

        # Fleet Chart
        fig = px.bar(data, x='Equipment_ID', y='Risk_Score', 
                     color='Risk_Score', color_continuous_scale='RdYlGn_r',
                     range_color=[0, 100], title="Risk Probability Across All Units")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Data Overview")
        st.dataframe(data[['Equipment_ID', 'Timestamp', 'Risk_Score', 'Avg_Vibration_Last_1_Hour']], use_container_width=True)

    def show_single_equipment_view(self, data, equipment_id):
        st.subheader(f"🔍 Detailed Analysis: {equipment_id}")
        
        # Filter for the selected item
        row = data[data['Equipment_ID'] == equipment_id].iloc[0]
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown(f"### Current Risk: **{row['Risk_Score']}%**")
            # Visual Progress Bar
            st.progress(int(row['Risk_Score']))
            
            if row['Risk_Score'] > 75:
                st.error("🚨 IMMEDIATE ACTION REQUIRED: High breakdown probability.")
            elif row['Risk_Score'] > 40:
                st.warning("⚠️ SCHEDULE INSPECTION: Signs of wear detected.")
            else:
                st.success("✅ HEALTHY: Operating within normal parameters.")

        with c2:
            st.markdown("### 📡 Sensor Data")
            st.write(f"**Vibration:** {row['Avg_Vibration_Last_1_Hour']}")
            st.write(f"**Temperature Delta:** {row['Temp_vs_7Day_Avg_Delta']}")
            st.write(f"**Days Since Last Breakdown:** {row['Days_Since_Last_Breakdown']}")

if __name__ == "__main__":
    agent = ReportingAgent()
    agent.run_dashboard()