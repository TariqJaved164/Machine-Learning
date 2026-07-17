import streamlit as st
import pandas as pd
import numpy as np
import datetime
import joblib
import os

# Set page layout to widescreen
st.set_page_config(
    page_title="Retail Profit Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS styling for a professional enterprise look
st.markdown("""
    <style>
    .main {
        background-color: #f3f4f6;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border-left: 5px solid #0EA5E9;
        margin-bottom: 20px;
    }
    .profit-display {
        font-size: 50px;
        font-weight: 800;
        color: #0EA5E9;
        margin-top: 5px;
        margin-bottom: 5px;
    }
    .banner {
        background: linear-gradient(135deg, #0F172A 0%, #2563EB 100%);
        color: white;
        padding: 30px;
        border-radius: 16px;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.2);
    }
    .kpi-title {
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-size: 11px;
        color: #6B7280;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


# Helper Function to safely load the pre-trained ML Pipeline
@st.cache_resource
def load_pipeline():
    model_path = 'model/retail_profit_pipeline.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None


pipeline = load_pipeline()

# Header Banner
st.markdown("""
    <div class="banner">
        <h1 style="margin:0; font-weight:800;">💰 Retail Transaction Profit Predictor</h1>
        <p style="margin:5px 0 0 0; opacity:0.9; font-size:16px;">
            Estimate transaction margins and analyze operational targets instantly using machine learning.
        </p>
    </div>
""", unsafe_allow_html=True)

# Safeguard check if the pipeline hasn't been generated
if pipeline is None:
    st.error("🚨 **Error:** Machine learning pipeline file (`retail_profit_pipeline.pkl`) was not found.")
    st.info(
        "Please make sure you have trained and saved your model first using your pipeline script before launching this web app.")
    st.stop()

# Set up main columns
col_inputs, col_predictions = st.columns([1.1, 1])

with col_inputs:
    st.markdown("### 📝 Transaction Parameters")

    # Card 1: Core Financials
    with st.container():
        st.write("**💵 Financials & Volume**")
        col_sales, col_qty = st.columns(2)
        with col_sales:
            sales = st.number_input("Gross Sales Amount ($)", min_value=1.0, max_value=50000.0, value=1200.0, step=50.0)
        with col_qty:
            quantity = st.number_input("Quantity Sold", min_value=1.0, max_value=500.0, value=5.0, step=1.0)

    st.markdown("---")

    # Card 2: Catalog Details
    with st.container():
        st.write("**📦 Catalog & Distribution**")
        col_cat, col_reg = st.columns(2)
        with col_cat:
            # Dropdown menu containing exact categories from your dataset
            category = st.selectbox(
                "Product Category",
                options=["Electronics", "Clothing", "Home Goods", "Sports", "Books"]
            )
        with col_reg:
            # Dropdown menu containing regional markets from your dataset
            region = st.selectbox(
                "Retail Region",
                options=["North", "East", "South", "West"]
            )

    st.markdown("---")

    # Card 3: Date & Seasonality
    with st.container():
        st.write("**📅 Time & Seasonality**")
        transaction_date = st.date_input(
            "Transaction Date",
            value=datetime.date.today()
        )
        # Extract features dynamically from chosen date picker
        month = transaction_date.month
        day_of_week = transaction_date.weekday()  # Monday=0, Sunday=6

with col_predictions:
    st.markdown("### 🔮 Margin Analysis & Forecast")

    # Format current user selections as a Pandas DataFrame matching your model's pipeline feature requirements
    new_data = pd.DataFrame([{
        'Category': category,
        'Sales': sales,
        'Quantity': quantity,
        'Region': region,
        'Month': month,
        'DayOfWeek': day_of_week
    }])

    # Run user selections through the ML Pipeline
    predicted_profit = pipeline.predict(new_data)[0]

    # Calculate operational metrics based on prediction outputs
    profit_margin_pct = (predicted_profit / sales) * 100 if sales > 0 else 0.0
    cost_of_goods_sold = sales - predicted_profit

    # Format color themes dynamically based on margins
    if profit_margin_pct >= 25:
        margin_theme = "#10B981"  # Green (Healthy margin)
        health_status = "Highly Profitable Sale"
    elif profit_margin_pct >= 10:
        margin_theme = "#F59E0B"  # Orange (Normal margin)
        health_status = "Standard Margin Profile"
    else:
        margin_theme = "#EF4444"  # Red (Low margin / loss)
        health_status = "Sub-optimal Profit Margin"

    # Display KPI Output Card with dynamic styling
    st.markdown(f"""
        <div class="metric-card" style="border-left-color: {margin_theme};">
            <span class="kpi-title">Forecasted Profit</span>
            <div class="profit-display">${predicted_profit:,.2f}</div>
            <div style="font-weight: 700; color: {margin_theme}; font-size: 16px;">
                ● {health_status} ({profit_margin_pct:.1f}% Margin)
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Side-by-side secondary metrics cards
    col_sub_margin, col_sub_cost = st.columns(2)
    with col_sub_margin:
        st.metric(
            label="Profit Margin %",
            value=f"{profit_margin_pct:.1f}%",
            delta=f"{profit_margin_pct - 20:.1f}% vs Target (20%)"
        )
    with col_sub_cost:
        st.metric(
            label="Estimated Internal Cost",
            value=f"${max(0.0, cost_of_goods_sold):,.2f}",
            delta=None
        )

    # Contextual Strategic Recommendations
    st.markdown("### 📈 Operational Advice")
    if profit_margin_pct < 15:
        st.warning(
            f"**⚠️ Margin Warning:** This transaction falls below safe retail targets. "
            f"Consider bundling with high-margin items in *{category}* or offering a lower wholesale bulk discount "
            f"to offset costs."
        )
    else:
        st.success(
            f"**✅ Healthy Margin Verified:** High profitability zone. "
            f"Your current pricing structure for *{category}* in the *{region}* region "
            f"is optimized effectively."
        )