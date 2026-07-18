import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 1. Page Configuration for Widescreen Layout
st.set_page_config(
    page_title="MindScreen AI: Adolescent Risk Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Modern UI Stylesheet Custom Injection
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    .header-banner {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: white;
        padding: 30px;
        border-radius: 16px;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.15);
    }
    .risk-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.04);
        text-align: center;
        border-top: 6px solid #e2e8f0;
        transition: transform 0.2s;
    }
    .risk-card:hover {
        transform: translateY(-2px);
    }
    .risk-score {
        font-size: 48px;
        font-weight: 800;
        margin: 10px 0;
    }
    .risk-label {
        font-weight: 700;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #475569;
    }
    .risk-status {
        font-size: 14px;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 20px;
        display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Securely Load the Multi-Output Model Pipeline
@st.cache_resource
def load_health_pipeline():
    model_path = 'model/social_media_mental_health_pipeline.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None

pipeline = load_health_pipeline()

# 4. App Title Section
st.markdown("""
    <div class="header-banner">
        <h1 style="margin:0; font-weight:800;">🧠 MindScreen AI</h1>
        <p style="margin:5px 0 0 0; opacity:0.9; font-size:16px;">
            Adolescent Risk Predictor Dashboard: Analyzing the intersection of digital lifestyle and mental well-being.
        </p>
    </div>
""", unsafe_allow_html=True)

# Terminate execution gracefully if the pickle file is missing
if pipeline is None:
    st.error("🚨 **Pipeline File Missing:** `teen_mental_health_pipeline.pkl` could not be located in your root directory.")
    st.info("Please execute your model training pipeline script to generate the saved pickle file before running this Streamlit dashboard.")
    st.stop()

# 5. Segmented Tabs for Form Input Elements
st.markdown("### 📝 Teen Behavioral Profiler")

tab_digital, tab_lifestyle, tab_academic = st.tabs([
    "🌐 Digital Habits", 
    "💤 Lifestyle & Sleep", 
    "🏫 Academics & Social"
])

with tab_digital:
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        daily_social_media_hours = st.slider("Daily Social Media Activity (Hours)", min_value=0.0, max_value=24.0, value=4.5, step=0.5)
    with col_d2:
        platform_usage = st.selectbox("Primary Active Platform", options=["Instagram", "TikTok", "Both"])
        
    screen_time_before_sleep = st.slider("Pre-Sleep Screen Exposure (Hours)", min_value=0.0, max_value=6.0, value=1.5, step=0.5, help="Time spent on digital screens within 1 hour before sleeping.")

with tab_lifestyle:
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        sleep_hours = st.slider("Nightly Sleep Duration (Hours)", min_value=3.0, max_value=12.0, value=7.0, step=0.5)
    with col_l2:
        physical_activity = st.slider("Daily Physical Exercise (Hours)", min_value=0.0, max_value=5.0, value=1.0, step=0.5)
        
    age = st.number_input("Adolescent Age", min_value=10, max_value=21, value=15, step=1)

with tab_academic:
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        academic_performance = st.number_input("Academic Performance / GPA (0.0 - 4.0)", min_value=0.0, max_value=4.0, value=3.0, step=0.1)
    with col_a2:
        social_interaction_level = st.select_slider("Real-World Social Interaction Quality", options=["low", "medium", "high"], value="medium")
        
    gender = st.radio("Gender Profile", options=["male", "female"], horizontal=True)

# 6. Make Predictions using the Preprocessing Pipeline
input_features = pd.DataFrame([{
    'age': age,
    'gender': gender,
    'daily_social_media_hours': daily_social_media_hours,
    'platform_usage': platform_usage,
    'sleep_hours': sleep_hours,
    'screen_time_before_sleep': screen_time_before_sleep,
    'academic_performance': academic_performance,
    'physical_activity': physical_activity,
    'social_interaction_level': social_interaction_level
}])

# Model returns a 2D array: [[stress, anxiety, addiction]]
predictions = pipeline.predict(input_features)[0]

# Clip values between 1 and 10 to protect against extreme edge cases
stress_score = np.clip(predictions[0], 1.0, 10.0)
anxiety_score = np.clip(predictions[1], 1.0, 10.0)
addiction_score = np.clip(predictions[2], 1.0, 10.0)

# Helper function to dynamically output specific colors and warnings based on scores
def determine_risk_tier(score):
    if score >= 7.0:
        return "#EF4444", "#FEF2F2", "HIGH RISK"      # Red theme
    elif score >= 4.0:
        return "#F59E0B", "#FFFBEB", "MODERATE"       # Yellow theme
    else:
        return "#10B981", "#ECFDF5", "MINIMAL"        # Green theme

st.markdown("---")
st.markdown("### 🔮 Multi-Dimensional Risk Metrics Output")

# 7. Rendering Three Horizontal Alert Cards
card_cols = st.columns(3)
metrics_list = [
    {"label": "Stress Index", "score": stress_score, "border": "#6366F1"},
    {"label": "Anxiety Index", "score": anxiety_score, "border": "#EC4899"},
    {"label": "Digital Addiction", "score": addiction_score, "border": "#F59E0B"}
]

for idx, metric in enumerate(metrics_list):
    with card_cols[idx]:
        txt_color, bg_color, tier_label = determine_risk_tier(metric["score"])
        
        st.markdown(f"""
            <div class="risk-card" style="border-top-color: {metric['border']};">
                <div class="risk-label">{metric['label']}</div>
                <div class="risk-score" style="color: {metric['border']};">{metric['score']:.1f}<span style="font-size:18px; color:#94A3B8;">/10</span></div>
                <div class="risk-status" style="color: {txt_color}; background-color: {bg_color};">
                    {tier_label}
                </div>
            </div>
        """, unsafe_allow_html=True)

# 8. Dynamic Warning Flags section
if stress_score >= 7.0 or anxiety_score >= 7.0 or addiction_score >= 7.0:
    st.markdown("### ⚠️ Critical Alerts Detected")
    if addiction_score >= 7.0:
        st.warning("**Screen Exposure Mitigation Recommended:** The model indicates highly elevated risk scores for digital dependency. Introduce intentional device-free slots and limit screen time before sleep.")
    if stress_score >= 7.0 or anxiety_score >= 7.0:
        st.error("**Support System Intervention Advised:** Predicted stress or anxiety metrics fall into the high-risk bracket. Consider consulting educational advisors or counseling support teams to evaluate workload and lifestyle habits.")