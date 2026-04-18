import os
import csv
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="ChurnX AI Retention Studio",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_advanced_model.pkl")
DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
HISTORY_PATH = "models/prediction_history.csv"
COMPARISON_PATH = "models/model_comparison.csv"
RESULTS_PATH = "models/results.txt"

# ----------------------------
# CUSTOM CSS
# ----------------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #1e293b 100%);
    color: white;
}
.main .block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 15px;
    border-radius: 18px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.18);
}
div.stButton > button {
    width: 100%;
    border-radius: 14px;
    height: 3rem;
    font-weight: 700;
    border: none;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
}
.stTabs [data-baseweb="tab"] {
    background-color: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding-left: 16px;
    padding-right: 16px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# HELPERS
# ----------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df

def safe_load_history():
    if not os.path.exists(HISTORY_PATH):
        return pd.DataFrame()
    try:
        return pd.read_csv(HISTORY_PATH, on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()

def save_history(row_df):
    os.makedirs("models", exist_ok=True)
    row_df.to_csv(
        HISTORY_PATH,
        mode="a" if os.path.exists(HISTORY_PATH) else "w",
        header=not os.path.exists(HISTORY_PATH),
        index=False,
        quoting=csv.QUOTE_MINIMAL
    )

def risk_band(prob):
    if prob < 0.20:
        return "Very Low"
    elif prob < 0.40:
        return "Low"
    elif prob < 0.60:
        return "Medium"
    elif prob < 0.80:
        return "High"
    return "Critical"

def segment_customer(probability, tenure, monthly_charges, total_charges):
    if probability >= 0.80 and monthly_charges > 80:
        return "Critical High-Value"
    elif probability >= 0.70 and tenure < 12:
        return "New Customer At Risk"
    elif probability < 0.30 and tenure > 24:
        return "Loyal Stable"
    elif total_charges > 3000 and probability >= 0.60:
        return "Premium Save Priority"
    return "Watchlist"

def persona_name(row, prob):
    if prob >= 0.80 and row["Contract"] == "Month-to-month":
        return "The Flight Risk"
    if row["tenure"] < 12 and prob >= 0.60:
        return "The Unsettled Newcomer"
    if row["MonthlyCharges"] > 85 and prob >= 0.65:
        return "The Expensive Doubter"
    if row["tenure"] > 24 and prob < 0.30:
        return "The Loyal Veteran"
    return "The Uncertain Subscriber"

def recommendations(row, prob):
    recs = []
    if row["Contract"] == "Month-to-month":
        recs.append("Offer discounted annual contract")
    if row["TechSupport"] == "No":
        recs.append("Provide free premium tech support")
    if row["OnlineSecurity"] == "No":
        recs.append("Bundle online security")
    if row["PaymentMethod"] == "Electronic check":
        recs.append("Push auto-pay migration offer")
    if row["MonthlyCharges"] > 80:
        recs.append("Recommend lower-cost plan")
    if row["tenure"] < 12:
        recs.append("Trigger onboarding retention campaign")
    if not recs and prob < 0.4:
        recs.append("Maintain loyalty rewards")
    return recs[:4]

def retention_value(prob, monthly_charges, tenure):
    return round((prob * monthly_charges * max(tenure, 1)) / 3, 2)

def save_urgency(prob, monthly_charges):
    score = prob * 100 + (monthly_charges / 2)
    if score >= 120:
        return "Immediate"
    elif score >= 90:
        return "High"
    elif score >= 60:
        return "Moderate"
    return "Low"

def make_input_df(
    gender, senior_citizen, partner, dependents, tenure, phone_service,
    multiple_lines, internet_service, online_security, online_backup,
    device_protection, tech_support, streaming_tv, streaming_movies,
    contract, paperless_billing, payment_method, monthly_charges, total_charges
):
    return pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

def customer_twin(df):
    better = df.copy()
    worse = df.copy()

    better["Contract"] = "Two year"
    better["TechSupport"] = "Yes"
    better["OnlineSecurity"] = "Yes"
    better["PaymentMethod"] = "Credit card (automatic)"
    better["MonthlyCharges"] = max(float(df["MonthlyCharges"].iloc[0]) - 15, 20)

    worse["Contract"] = "Month-to-month"
    worse["TechSupport"] = "No"
    worse["OnlineSecurity"] = "No"
    worse["PaymentMethod"] = "Electronic check"
    worse["MonthlyCharges"] = min(float(df["MonthlyCharges"].iloc[0]) + 15, 150)

    return better, worse

# ----------------------------
# LOAD FILES
# ----------------------------
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Run `python churn_advanced.py` first.")
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error("Dataset file not found.")
    st.stop()

model = load_model()
df = load_data()
history_df = safe_load_history()

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.title("⚡ Control Center")
threshold = st.sidebar.slider("Decision Threshold", 0.10, 0.90, 0.50, 0.05)
bulk_threshold = st.sidebar.slider("Bulk Threshold", 0.10, 0.90, 0.50, 0.05)
st.sidebar.markdown("---")
st.sidebar.info("ChurnX AI Retention Studio\n\nPortfolio-grade churn intelligence dashboard.")

# ----------------------------
# HEADER
# ----------------------------
st.title("🚀 ChurnX AI Retention Studio")
st.caption("Ultra-pro retention intelligence platform with simulation, ranking, action planning, and executive insights.")

# ----------------------------
# EXECUTIVE TOP KPIS
# ----------------------------
total_customers = len(df)
observed_churn = (df["Churn"].eq("Yes").mean() * 100) if "Churn" in df.columns else 0
avg_monthly = df["MonthlyCharges"].mean()
avg_tenure = df["tenure"].mean()
premium_share = (df["MonthlyCharges"] > 80).mean() * 100

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Customer Base", f"{total_customers:,}")
k2.metric("Observed Churn", f"{observed_churn:.2f}%")
k3.metric("Avg Monthly Charges", f"${avg_monthly:.2f}")
k4.metric("Avg Tenure", f"{avg_tenure:.1f} mo")
k5.metric("Premium Customer Share", f"{premium_share:.1f}%")

tabs = st.tabs([
    "🎯 Customer AI Mission",
    "🧪 Action Simulator",
    "🧬 Customer Twin Lab",
    "🏆 Smart Portfolio Ranking",
    "🌍 Executive Portfolio View",
    "📈 Strategy Intelligence",
    "🗂 Audit Trail"
])

# ----------------------------
# TAB 1
# ----------------------------
with tabs[0]:
    st.subheader("Customer AI Mission")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure", 0, 72, 12)

    with c2:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

    with c3:
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    with c4:
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        monthly_charges = st.slider("Monthly Charges", 0.0, 150.0, 70.0, 0.5)
        total_charges = st.slider("Total Charges", 0.0, 10000.0, 1500.0, 1.0)

    current_input = make_input_df(
        gender, senior_citizen, partner, dependents, tenure, phone_service,
        multiple_lines, internet_service, online_security, online_backup,
        device_protection, tech_support, streaming_tv, streaming_movies,
        contract, paperless_billing, payment_method, monthly_charges, total_charges
    )

    if st.button("Launch AI Customer Analysis", use_container_width=True):
        prob = float(model.predict_proba(current_input)[0][1])
        pred = "Churn" if prob >= threshold else "Stay"
        risk = risk_band(prob)
        segment = segment_customer(prob, tenure, monthly_charges, total_charges)
        persona = persona_name(current_input.iloc[0], prob)
        recs = recommendations(current_input.iloc[0], prob)
        value_score = retention_value(prob, monthly_charges, tenure)
        urgency = save_urgency(prob, monthly_charges)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Churn Probability", f"{prob*100:.2f}%")
        m2.metric("Decision", pred)
        m3.metric("Risk Band", risk)
        m4.metric("Save Urgency", urgency)
        m5.metric("Retention Value", f"${value_score}")

        s1, s2 = st.columns([1, 1])

        with s1:
            st.markdown("### AI Persona")
            st.success(f"**{persona}**")
            st.write(f"Segment: **{segment}**")

            st.markdown("### Recommended Actions")
            for rec in recs:
                st.write(f"- {rec}")

        with s2:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Churn Heat Index"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "steps": [
                        {"range": [0, 20], "color": "#14532d"},
                        {"range": [20, 40], "color": "#3f6212"},
                        {"range": [40, 60], "color": "#a16207"},
                        {"range": [60, 80], "color": "#c2410c"},
                        {"range": [80, 100], "color": "#991b1b"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 4}, "value": threshold * 100}
                }
            ))
            st.plotly_chart(gauge, use_container_width=True)

        save_row = current_input.copy()
        save_row["Prediction"] = pred
        save_row["Probability"] = round(prob, 4)
        save_row["RiskLevel"] = risk
        save_row["Segment"] = segment
        save_row["Persona"] = persona
        save_row["SaveUrgency"] = urgency
        save_row["RetentionValue"] = value_score
        save_row["Actions"] = " | ".join(recs)
        save_history(save_row)

# ----------------------------
# TAB 2
# ----------------------------
with tabs[1]:
    st.subheader("Action Simulator")
    st.caption("Test strategic changes and measure risk reduction.")

    base_input = make_input_df(
        "Female", 0, "Yes", "No", 12, "Yes", "No",
        "Fiber optic", "No", "Yes", "No", "No", "Yes", "Yes",
        "Month-to-month", "Yes", "Electronic check", 89.5, 1050.0
    )

    current_prob = float(model.predict_proba(base_input)[0][1])

    a1, a2, a3, a4 = st.columns(4)
    with a1:
        sim_contract = st.selectbox("Switch Contract", ["Month-to-month", "One year", "Two year"])
    with a2:
        sim_tech = st.selectbox("Add Tech Support", ["Yes", "No", "No internet service"])
    with a3:
        sim_security = st.selectbox("Add Online Security", ["Yes", "No", "No internet service"])
    with a4:
        sim_payment = st.selectbox("Change Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

    sim_charge = st.slider("Adjusted Monthly Charges", 0.0, 150.0, 89.5, 0.5)

    scenario = base_input.copy()
    scenario["Contract"] = sim_contract
    scenario["TechSupport"] = sim_tech
    scenario["OnlineSecurity"] = sim_security
    scenario["PaymentMethod"] = sim_payment
    scenario["MonthlyCharges"] = sim_charge

    scenario_prob = float(model.predict_proba(scenario)[0][1])
    reduction = (current_prob - scenario_prob) * 100

    r1, r2, r3 = st.columns(3)
    r1.metric("Current Risk", f"{current_prob*100:.2f}%")
    r2.metric("Scenario Risk", f"{scenario_prob*100:.2f}%")
    r3.metric("Risk Reduction", f"{reduction:.2f} pts")

    compare_df = pd.DataFrame({
        "State": ["Current", "Scenario"],
        "Probability": [current_prob * 100, scenario_prob * 100]
    })
    fig_compare = px.bar(compare_df, x="State", y="Probability", text_auto=".2f", title="Retention Strategy Impact")
    st.plotly_chart(fig_compare, use_container_width=True)

# ----------------------------
# TAB 3
# ----------------------------
with tabs[2]:
    st.subheader("Customer Twin Lab")
    st.caption("Compare this customer against a better and worse version.")

    base = current_input
    better, worse = customer_twin(base)

    base_prob = float(model.predict_proba(base)[0][1])
    better_prob = float(model.predict_proba(better)[0][1])
    worse_prob = float(model.predict_proba(worse)[0][1])

    t1, t2, t3 = st.columns(3)
    t1.metric("Current Customer", f"{base_prob*100:.2f}%")
    t2.metric("Retained Twin", f"{better_prob*100:.2f}%")
    t3.metric("Risk Twin", f"{worse_prob*100:.2f}%")

    twin_df = pd.DataFrame({
        "Profile": ["Current", "Retained Twin", "Risk Twin"],
        "Probability": [base_prob * 100, better_prob * 100, worse_prob * 100]
    })
    fig_twin = px.bar(twin_df, x="Profile", y="Probability", color="Profile", text_auto=".2f", title="Customer Twin Comparison")
    st.plotly_chart(fig_twin, use_container_width=True)

# ----------------------------
# TAB 4
# ----------------------------
with tabs[3]:
    st.subheader("Smart Portfolio Ranking")

    portfolio = df.copy()
    scoring_df = portfolio.drop(columns=["customerID", "Churn"], errors="ignore").copy()
    probs = model.predict_proba(scoring_df)[:, 1]

    portfolio["PredictedProbability"] = probs
    portfolio["RiskLevel"] = [risk_band(p) for p in probs]
    portfolio["RetentionValue"] = [
        retention_value(p, mc, tn)
        for p, mc, tn in zip(probs, portfolio["MonthlyCharges"], portfolio["tenure"])
    ]
    portfolio["SaveUrgency"] = [
        save_urgency(p, mc)
        for p, mc in zip(probs, portfolio["MonthlyCharges"])
    ]

    portfolio["PriorityScore"] = (
        portfolio["PredictedProbability"] * 100 * 0.6 +
        portfolio["MonthlyCharges"] * 0.2 +
        portfolio["tenure"] * 0.2
    )

    leaderboard = portfolio.sort_values("PriorityScore", ascending=False).head(25)

    st.dataframe(
        leaderboard[[
            "customerID", "Contract", "MonthlyCharges", "tenure",
            "PredictedProbability", "RiskLevel", "SaveUrgency",
            "RetentionValue", "PriorityScore"
        ]],
        use_container_width=True
    )

    fig_priority = px.bar(
        leaderboard,
        x="customerID",
        y="PriorityScore",
        color="RiskLevel",
        title="Top Priority Customers"
    )
    st.plotly_chart(fig_priority, use_container_width=True)

# ----------------------------
# TAB 5
# ----------------------------
with tabs[4]:
    st.subheader("Executive Portfolio View")

    portfolio = df.copy()
    scoring_df = portfolio.drop(columns=["customerID", "Churn"], errors="ignore").copy()
    probs = model.predict_proba(scoring_df)[:, 1]
    portfolio["PredictedProbability"] = probs
    portfolio["RiskLevel"] = [risk_band(p) for p in probs]

    c1, c2 = st.columns(2)

    with c1:
        fig1 = px.histogram(portfolio, x="Contract", color="RiskLevel", barmode="group", title="Risk by Contract")
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        fig2 = px.histogram(portfolio, x="InternetService", color="RiskLevel", barmode="group", title="Risk by Internet Service")
        st.plotly_chart(fig2, use_container_width=True)

    portfolio["TenureBand"] = pd.cut(
        portfolio["tenure"],
        bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6", "7-12", "13-24", "25-48", "49-72"]
    )

    heat = portfolio.groupby(["TenureBand", "Contract"], observed=False)["PredictedProbability"].mean().reset_index()
    heat_pivot = heat.pivot(index="TenureBand", columns="Contract", values="PredictedProbability")
    heat_fig = px.imshow(heat_pivot, text_auto=".2f", aspect="auto", title="Average Predicted Churn Heatmap")
    st.plotly_chart(heat_fig, use_container_width=True)

# ----------------------------
# TAB 6
# ----------------------------
with tabs[5]:
    st.subheader("Strategy Intelligence")

    if os.path.exists(COMPARISON_PATH):
        comp = pd.read_csv(COMPARISON_PATH)
        st.markdown("### Model Benchmarking")
        st.dataframe(comp, use_container_width=True)

        fig_comp = px.bar(
            comp.melt(id_vars="Model", value_vars=["Accuracy", "ROC_AUC"]),
            x="Model",
            y="value",
            color="variable",
            barmode="group",
            title="Model Comparison"
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    if os.path.exists(RESULTS_PATH):
        st.markdown("### Saved Evaluation Metrics")
        with open(RESULTS_PATH, "r") as f:
            st.code(f.read())

# ----------------------------
# TAB 7
# ----------------------------
with tabs[6]:
    st.subheader("Audit Trail")

    hist = safe_load_history()
    if hist.empty:
        st.info("No predictions saved yet.")
    else:
        st.dataframe(hist.tail(50), use_container_width=True)

        h1, h2 = st.columns(2)
        with h1:
            fig_hist = px.histogram(hist, x="RiskLevel", color="Prediction", barmode="group", title="Historical Risk Mix")
            st.plotly_chart(fig_hist, use_container_width=True)
        with h2:
            if "Persona" in hist.columns:
                fig_persona = px.histogram(hist, x="Persona", color="Prediction", barmode="group", title="Historical Persona Mix")
                st.plotly_chart(fig_persona, use_container_width=True)
