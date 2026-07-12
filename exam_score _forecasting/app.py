import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page configuration for a modern widescreen look
st.set_page_config(
    page_title="AI Student Performance Advisor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern styling, clean rounded cards, and vibrant gradients
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border-left: 5px solid #4F46E5;
        margin-bottom: 20px;
    }
    .score-display {
        font-size: 54px;
        font-weight: 800;
        color: #4F46E5;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .title-banner {
        background: linear-gradient(135deg, #4F46E5 0%, #06B6D4 100%);
        color: white;
        padding: 30px;
        border-radius: 16px;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.2);
    }
    .tip-box {
        background-color: #ECFDF5;
        border-left: 5px solid #10B981;
        padding: 15px;
        border-radius: 8px;
        margin-top: 15px;
    }
    .warning-box {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 15px;
        border-radius: 8px;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)  # <-- Fixed parameter


# Define Helper Function to load the ML Pipeline
@st.cache_resource
def load_pipeline():
    model_path = 'student_score_pipeline.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None


pipeline = load_pipeline()

# Header Banner
st.markdown("""
    <div class="title-banner">
        <h1 style="margin:0; font-weight:800;">🎓 AI Student Performance Advisor</h1>
        <p style="margin:5px 0 0 0; opacity:0.9; font-size:16px;">Predict exam performance instantly and discover actionable behavioral improvements.</p>
    </div>
""", unsafe_allow_html=True)  # <-- Fixed parameter

# Handle cases where model file hasn't been generated yet
if pipeline is None:
    st.error(
        "🚨 **Error:** Machine learning pipeline file (`student_score_pipeline.pkl`) was not found in your directory.")
    st.info("Please run your pipeline training script first to save your model before launching this app.")
    st.stop()

# Main Grid Layout (Inputs on left, Predictive card on the right)
col_input, col_pred = st.columns([1.1, 1])

with col_input:
    st.markdown("### 📝 Enter Student Metrics")

    # Sub-card 1: Study Habits
    with st.container():
        st.write("**📚 Academic & Study Engagement**")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            study_hours = st.slider("Daily Study Hours", min_value=1, max_value=11, value=6, step=1,
                                    help="Average number of hours studied per day.")
        with col_s2:
            assignments = st.slider("Assignments Completed (Out of 20)", min_value=0, max_value=20, value=10, step=1)

    # Sub-card 2: Class Attendance & History
    st.markdown("---")
    with st.container():
        st.write("**🏫 Class Attendance & History**")
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            attendance = st.number_input("School Attendance (%)", min_value=40, max_value=100, value=85, step=1)
        with col_a2:
            prev_score = st.number_input("Previous Term Score (Out of 100)", min_value=35, max_value=95, value=70,
                                         step=1)

    # Sub-card 3: Lifestyle Metrics
    st.markdown("---")
    with st.container():
        st.write("**💤 Lifestyle & Internet Balance**")
        col_l1, col_l2 = st.columns(2)
        with col_l1:
            sleep_hours = st.slider("Daily Sleep Hours", min_value=4, max_value=9, value=7, step=1)
        with col_l2:
            internet_usage = st.slider("Daily Internet Usage Hours", min_value=1, max_value=11, value=4, step=1)

with col_pred:
    st.markdown("### 🔮 Predicted Academic Output")

    # Prepare the raw inputs as a DataFrame for the prediction pipeline
    input_data = pd.DataFrame([{
        'study_hours': study_hours,
        'attendance': attendance,
        'sleep_hours': sleep_hours,
        'internet_usage': internet_usage,
        'assignments_completed': assignments,
        'previous_score': prev_score
    }])

    # Run the raw data directly through our loaded pipeline
    raw_prediction = pipeline.predict(input_data)[0]
    final_score = np.clip(raw_prediction, 0.0, 100.0)  # Ensure score is bounded between 0-100

    # Dynamically color the visual output based on prediction
    if final_score >= 80:
        status_color = "#10B981"  # Emerald Green
        status_text = "Excellent Standing (Grade A)"
    elif final_score >= 60:
        status_color = "#F59E0B"  # Warm Amber
        status_text = "Good Progress (Grade B/C)"
    else:
        status_color = "#EF4444"  # Crimson Red
        status_text = "Risk of Academic Underperformance"

    # Beautiful Custom HTML Output Card
    st.markdown(f"""
        <div class="metric-card" style="border-left-color: {status_color};">
            <span style="text-transform: uppercase; letter-spacing: 1.5px; font-size: 12px; color: #6B7280; font-weight: bold;">Forecasted Final Grade</span>
            <div class="score-display">{final_score:.1f}%</div>
            <div style="font-weight: 700; color: {status_color}; font-size: 18px;">
                ● {status_text}
            </div>
        </div>
    """, unsafe_allow_html=True)  # <-- Fixed parameter

    # Dynamic Behavioral Feedback Recommendations
    st.markdown("### 💡 AI Strategic Feedback")

    if study_hours <= 4:
        st.markdown("""
            <div class="warning-box">
                <strong>⚠️ Low Study Engagement:</strong> Daily study hours are below critical levels. 
                Increasing study time by just 1-2 hours per day has the highest mathematical leverage to boost this student's score.
            </div>
        """, unsafe_with_html=False, unsafe_allow_html=True)  # <-- Fixed parameter

    if attendance < 75:
        st.markdown("""
            <div class="warning-box">
                <strong>⚠️ Attendance Warning:</strong> Attendance levels are low. Class presence is a core structural feature; 
                missing lessons significantly dampens retention and assignment success.
            </div>
        """, unsafe_with_html=False, unsafe_allow_html=True)  # <-- Fixed parameter

    if sleep_hours < 6:
        st.markdown("""
            <div class="tip-box">
                <strong>💤 Prioritize Sleep Recovery:</strong> Sleep hours are below the recommended amount. 
                Sufficient sleep (7+ hours) improves next-day memory consolidation and cognitive exam performance.
            </div>
        """, unsafe_with_html=False, unsafe_allow_html=True)  # <-- Fixed parameter

    if assignments >= 15 and attendance >= 85 and final_score >= 75:
        st.markdown("""
            <div class="tip-box" style="background-color: #EEF2FF; border-left-color: #4F46E5;">
                <strong>🚀 Keep Up the Excellent Habits:</strong> High assignment submissions and solid class attendance 
                are working in harmony! Maintain this operational routine to guarantee top outcomes.
            </div>
        """, unsafe_with_html=False, unsafe_allow_html=True)  # <-- Fixed parameter