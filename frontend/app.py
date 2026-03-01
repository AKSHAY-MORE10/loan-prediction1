import os
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Loan Eligibility", layout="wide")

# Works for both local runs and Docker Compose:
# - Local default: http://localhost:8000
# - Compose should set: BACKEND_URL=http://backend:8000
BACKEND_URL = os.getenv("BACKEND_URL") or os.getenv("BACKEND_HOST") or "http://localhost:8000"

# =========================
# DARK PROFESSIONAL CSS
# =========================
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #0E1117;
    color: #E6EDF3;
    font-family: Inter, system-ui, sans-serif;
}
.block-container { padding: 2rem 3rem; }
section[data-testid="stSidebar"] {
    background-color: #0B0F14;
    border-right: 1px solid #1F2937;
}
.card {
    background-color: #161B22;
    border: 1px solid #2A2F36;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}
input, select {
    background-color: #0E1117 !important;
    color: #E6EDF3 !important;
    border: 1px solid #2A2F36 !important;
}
button[kind="primary"] {
    background-color: #2563EB !important;
    color: white !important;
    font-weight: 600;
    height: 48px;
}
[data-testid="stMetric"] {
    background-color: #161B22;
    border: 1px solid #2A2F36;
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HELPERS
# =========================
def backend_alive():
    try:
        return requests.get(f"{BACKEND_URL}/health", timeout=2).status_code == 200
    except:
        return False

def predict(data):
    r = requests.post(f"{BACKEND_URL}/predict", json=data)
    r.raise_for_status()
    return r.json()

def gauge(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#2563EB"},
            "steps": [
                {"range": [0, 50], "color": "#3F1D1D"},
                {"range": [50, 75], "color": "#3F3A1D"},
                {"range": [75, 100], "color": "#1D3F2A"},
            ],
        }
    ))
    fig.update_layout(height=300, paper_bgcolor="#161B22", font={"color": "#E6EDF3"})
    return fig

# =========================
# HEADER
# =========================
st.title("Loan Eligibility")
st.caption("AI-based credit assessment system")

if not backend_alive():
    st.error("Backend not running on port 8000")
    st.stop()

# =========================
# SIDEBAR MODE
# =========================
with st.sidebar:
    mode = st.radio("Select Mode", ["Single Prediction", "Batch Prediction"], key="mode")

# =========================
# SINGLE PREDICTION
# =========================
if mode == "Single Prediction":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Applicant Details")

    c1, c2, c3 = st.columns(3)

    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
        married = st.selectbox("Married", ["Yes", "No"], key="married")
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"], key="deps")

    with c2:
        applicant_income = st.number_input("Applicant Income (₹ / month)", 0, 200000, 5000, key="ai")
        coapplicant_income = st.number_input("Co-applicant Income (₹ / month)", 0, 200000, 2000, key="ci")

        loan_amount_lakh = st.number_input(
            "Loan Amount (₹ in Lakhs)",
            min_value=0.5,
            max_value=100.0,
            value=5.0,
            step=0.5,
            key="loan_lakh"
        )
        st.caption("Example: 5 = ₹5,00,000 | 10 = ₹10,00,000")

        # 🔁 Convert Lakhs → Thousands (model unit)
        loan_amount = loan_amount_lakh * 100

    with c3:
        credit_history = st.selectbox("Credit History", [1.0, 0.0], key="ch")
        education = st.selectbox("Education", ["Graduate", "Not Graduate"], key="edu")
        loan_term = st.selectbox("Loan Term (months)", [120, 180, 240, 360], key="lt")

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Check Eligibility", use_container_width=True):
        payload = {
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": "No",
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,   # ✅ converted value
            "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history,
            "Property_Area": "Urban"
        }

        result = predict(payload)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📊 Prediction Results")

        m1, m2, m3 = st.columns(3)
        m1.metric("Decision", result["decision"])
        m2.metric("Confidence", result["confidence"])
        m3.metric("Approval Probability", f"{result['probability']*100:.1f}%")

        st.plotly_chart(gauge(result["probability"]), use_container_width=True)

        exp = result["explanation"]
        st.subheader("📈 Key Factors")

        k1, k2, k3 = st.columns(3)
        k1.metric("Credit History", exp["Credit_History"])
        k2.metric("Total Income", f"₹{exp['TotalIncome']:,.0f}")
        k3.metric("Loan Amount", f"₹{exp['LoanAmount']/100:.1f} Lakhs")

        st.subheader("💡 Recommendations")
        if result["decision"] == "Eligible":
            st.success("Strong eligibility. Maintain good credit and income stability.")
        else:
            st.warning("Improve income, reduce loan amount, or build credit history.")

        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# BATCH MODE
# =========================
 # else:
#     st.markdown("<div class='card'>", unsafe_allow_html=True)
#     st.subheader("Batch Prediction")

#     file = st.file_uploader("Upload CSV", type="csv", key="csv")

#     if file:
#         df = pd.read_csv(file)
#         st.dataframe(df.head(), use_container_width=True)

#         if st.button("Run Batch Prediction"):
#             results = []
#             for i, row in df.iterrows():
#                 r = predict(row.to_dict())
#                 allowed_keys = [
#     "Gender", "Married", "Dependents", "Education", "Self_Employed",
#     "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
#     "Loan_Amount_Term", "Credit_History", "Property_Area"
# ]
#                 results.append({
#                     "ID": i + 1,
#                     "Decision": r["decision"],
#                     "Probability": r["probability"],
#                     "Confidence": r["confidence"]
#                 })

#             out = pd.DataFrame(results)
#             st.dataframe(out, use_container_width=True)
#             st.download_button("Download Results", out.to_csv(index=False), "loan_predictions.csv", "text/csv")

#     st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Batch Prediction")

    file = st.file_uploader("Upload CSV", type="csv", key="csv")

    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head(), use_container_width=True)

        if st.button("Run Batch Prediction"):
            results = []

            # ✅ Allowed columns as per API schema
            allowed_keys = [
                "Gender", "Married", "Dependents", "Education", "Self_Employed",
                "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
                "Loan_Amount_Term", "Credit_History", "Property_Area"
            ]

            for i, row in df.iterrows():
                # ✅ Filter only required keys
                payload = {k: row[k] for k in allowed_keys if k in row}

                # ✅ Call API with safe payload
                r = predict(payload)

                results.append({
                    "ID": i + 1,
                    "Decision": r["decision"],
                    "Probability": r["probability"],
                    "Confidence": r["confidence"]
                })

            out = pd.DataFrame(results)
            st.dataframe(out, use_container_width=True)
            st.download_button(
                "Download Results",
                out.to_csv(index=False),
                "loan_predictions.csv",
                "text/csv"
            )

    st.markdown("</div>", unsafe_allow_html=True)
