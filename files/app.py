import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import base64
from sklearn.ensemble import RandomForestClassifier

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnRadar | Telecom Analytics",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg: #0a0e1a;
    --surface: #111827;
    --surface2: #1a2236;
    --accent: #00e5ff;
    --accent2: #ff4d6d;
    --accent3: #7c3aed;
    --text: #e2e8f0;
    --muted: #64748b;
    --success: #10b981;
    --warning: #f59e0b;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

.stApp { background: var(--bg) !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid #1e293b !important;
}

/* Header */
.app-header {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a0e1a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -20%;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(0,229,255,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.app-title {
    font-family: 'Space Mono', monospace !important;
    font-size: 2.8rem;
    font-weight: 700;
    color: var(--accent) !important;
    letter-spacing: -1px;
    margin: 0;
    line-height: 1;
}
.app-subtitle {
    font-size: 1rem;
    color: var(--muted);
    margin-top: 0.5rem;
    font-weight: 300;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* Metric cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}
.metric-card {
    background: var(--surface);
    border-radius: 12px;
    padding: 1.4rem 1.2rem;
    border: 1px solid #1e293b;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: var(--accent); }
.metric-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: var(--accent-bar, var(--accent));
    border-radius: 3px 0 0 3px;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
}
.metric-label {
    font-size: 0.75rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 0.4rem;
}

/* Upload zone */
.upload-zone {
    background: var(--surface);
    border: 2px dashed #1e3a5f;
    border-radius: 16px;
    padding: 3rem 2rem;
    text-align: center;
    margin: 1rem 0;
    transition: border-color 0.3s;
}
.upload-zone:hover { border-color: var(--accent); }

/* Table styling */
.stDataFrame { border-radius: 12px !important; overflow: hidden !important; }

/* Risk badges */
.badge-high {
    background: rgba(255,77,109,0.15);
    color: #ff4d6d;
    border: 1px solid rgba(255,77,109,0.3);
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
}
.badge-low {
    background: rgba(16,185,129,0.12);
    color: #10b981;
    border: 1px solid rgba(16,185,129,0.3);
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
}
.badge-medium {
    background: rgba(245,158,11,0.12);
    color: #f59e0b;
    border: 1px solid rgba(245,158,11,0.3);
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
}

/* Section heading */
.section-head {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent);
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 0.5rem;
    margin-bottom: 1.2rem;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0284c7, #0ea5e9) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.8rem !important;
    letter-spacing: 0.5px !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Progress bars */
.risk-bar-wrap {
    background: #1e293b;
    border-radius: 4px;
    height: 6px;
    width: 100%;
    margin-top: 4px;
}
.risk-bar-fill {
    height: 6px;
    border-radius: 4px;
    background: linear-gradient(90deg, #10b981, #f59e0b, #ff4d6d);
}

/* Info box */
.info-box {
    background: rgba(0,229,255,0.05);
    border: 1px solid rgba(0,229,255,0.2);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.88rem;
    color: #94a3b8;
}

/* Chart container */
.chart-card {
    background: var(--surface);
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 1.5rem;
    margin: 0.5rem 0;
}

div[data-testid="stFileUploader"] {
    background: var(--surface2) !important;
    border: 1px solid #1e293b !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

div[data-testid="stSelectbox"] > div,
div[data-testid="stTextInput"] > div > div {
    background: var(--surface2) !important;
    border-color: #2d3f55 !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

.stAlert {
    background: var(--surface2) !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('churn_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('feature_cols.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    return model, encoders, feature_cols

model, encoders, feature_cols = load_model()

CAT_COLS = ['gender','Partner','Dependents','PhoneService','MultipleLines',
            'InternetService','OnlineSecurity','TechSupport','Contract',
            'PaperlessBilling','PaymentMethod']

EXPECTED_COLS = ['tenure','MonthlyCharges','TotalCharges','SeniorCitizen',
                 'gender','Partner','Dependents','PhoneService','MultipleLines',
                 'InternetService','OnlineSecurity','TechSupport','Contract',
                 'PaperlessBilling','PaymentMethod']

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-head">// ChurnRadar v1.0</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.85rem; color:#64748b; line-height:1.7;">
    Upload a customer CSV to get instant churn predictions and risk scores for your entire base.
    <br><br>
    <b style="color:#94a3b8;">Required columns:</b>
    </div>
    """, unsafe_allow_html=True)
    
    for col in EXPECTED_COLS:
        st.markdown(f"<span style='font-family:Space Mono,monospace;font-size:0.72rem;color:#0ea5e9;'>→ {col}</span>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.78rem;color:#475569;">
    <b style="color:#94a3b8;">Model:</b> Random Forest<br>
    <b style="color:#94a3b8;">Accuracy:</b> 91.5%<br>
    <b style="color:#94a3b8;">Built by:</b> Neha Chaudhari
    </div>
    """, unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-title">📡 ChurnRadar</div>
    <div class="app-subtitle">Telecom Customer Churn Prediction Engine</div>
</div>
""", unsafe_allow_html=True)

# ── Sample CSV generator ───────────────────────────────────────────────────────
def generate_sample_csv():
    sample = pd.DataFrame({
        'tenure': [2, 34, 2, 45, 60, 8, 22, 10],
        'MonthlyCharges': [89.1, 45.0, 92.3, 55.2, 35.8, 78.9, 60.5, 95.1],
        'TotalCharges': [178.2, 1530.0, 184.6, 2484.0, 2148.0, 631.2, 1331.0, 951.0],
        'SeniorCitizen': [0, 0, 0, 0, 1, 0, 0, 0],
        'gender': ['Female','Male','Male','Female','Female','Male','Male','Female'],
        'Partner': ['Yes','No','No','No','No','Yes','No','Yes'],
        'Dependents': ['No','No','No','No','No','No','No','No'],
        'PhoneService': ['Yes','Yes','Yes','Yes','Yes','Yes','Yes','Yes'],
        'MultipleLines': ['No','No','No','No','No','Yes','No','Yes'],
        'InternetService': ['Fiber optic','DSL','Fiber optic','DSL','No','Fiber optic','DSL','Fiber optic'],
        'OnlineSecurity': ['No','Yes','No','Yes','No internet service','No','Yes','No'],
        'TechSupport': ['No','No','No','Yes','No internet service','No','Yes','No'],
        'Contract': ['Month-to-month','One year','Month-to-month','One year','Two year','Month-to-month','One year','Month-to-month'],
        'PaperlessBilling': ['Yes','No','Yes','No','No','Yes','Yes','Yes'],
        'PaymentMethod': ['Electronic check','Mailed check','Mailed check','Bank transfer (automatic)','Bank transfer (automatic)','Electronic check','Credit card (automatic)','Electronic check'],
    })
    return sample.to_csv(index=False)

col1, col2 = st.columns([3, 1])
with col2:
    sample_csv = generate_sample_csv()
    b64 = base64.b64encode(sample_csv.encode()).decode()
    st.markdown(f"""
    <a href="data:file/csv;base64,{b64}" download="sample_telecom_customers.csv">
        <button style="background:transparent;border:1px solid #1e3a5f;color:#0ea5e9;
        padding:0.5rem 1rem;border-radius:8px;cursor:pointer;font-size:0.82rem;
        font-family:DM Sans,sans-serif;width:100%;transition:border-color 0.2s;"
        onmouseover="this.style.borderColor='#0ea5e9'" 
        onmouseout="this.style.borderColor='#1e3a5f'">
        ⬇ Download Sample CSV
        </button>
    </a>
    """, unsafe_allow_html=True)

# ── Upload ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-head">// Upload Customer Data</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["csv"], label_visibility="collapsed")

if uploaded_file is None:
    st.markdown("""
    <div class="info-box">
    💡 <b>First time?</b> Download the sample CSV above, take a look at the format, then upload your own customer file.
    Predictions run instantly — no training required.
    </div>
    """, unsafe_allow_html=True)

# ── Predict ────────────────────────────────────────────────────────────────────
def preprocess(df):
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    for col in CAT_COLS:
        if col in df.columns:
            le = encoders[col]
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
            df[col + '_enc'] = le.transform(df[col])
    return df[feature_cols]

def risk_label(prob):
    if prob >= 0.70:
        return "HIGH"
    elif prob >= 0.40:
        return "MEDIUM"
    else:
        return "LOW"

def risk_badge(label):
    cls = {"HIGH": "badge-high", "MEDIUM": "badge-medium", "LOW": "badge-low"}[label]
    return f'<span class="{cls}">{label}</span>'

if uploaded_file:
    try:
        df_raw = pd.read_csv(uploaded_file)

        missing = [c for c in EXPECTED_COLS if c not in df_raw.columns]
        if missing:
            st.error(f"❌ Missing columns: {', '.join(missing)}")
            st.stop()

        X = preprocess(df_raw)
        probs = model.predict_proba(X)[:, 1]
        preds = model.predict(X)

        df_raw['Churn_Probability'] = np.round(probs * 100, 1)
        df_raw['Churn_Prediction'] = ['Will Churn' if p == 1 else 'Will Stay' for p in preds]
        df_raw['Risk_Level'] = [risk_label(p) for p in probs]

        # ── Summary metrics ────────────────────────────────────────────────────
        total = len(df_raw)
        churners = int(preds.sum())
        churn_rate = churners / total * 100
        avg_prob = probs.mean() * 100
        high_risk = (df_raw['Risk_Level'] == 'HIGH').sum()

        st.markdown('<p class="section-head">// Prediction Summary</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card" style="--accent-bar: #0ea5e9;">
                <div class="metric-value" style="color:#0ea5e9;">{total}</div>
                <div class="metric-label">Total Customers</div>
            </div>
            <div class="metric-card" style="--accent-bar: #ff4d6d;">
                <div class="metric-value" style="color:#ff4d6d;">{churners}</div>
                <div class="metric-label">Predicted to Churn</div>
            </div>
            <div class="metric-card" style="--accent-bar: #f59e0b;">
                <div class="metric-value" style="color:#f59e0b;">{churn_rate:.1f}%</div>
                <div class="metric-label">Churn Rate</div>
            </div>
            <div class="metric-card" style="--accent-bar: #7c3aed;">
                <div class="metric-value" style="color:#7c3aed;">{high_risk}</div>
                <div class="metric-label">High-Risk Customers</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Charts ─────────────────────────────────────────────────────────────
        st.markdown('<p class="section-head">// Visual Insights</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)

        with c1:
            risk_counts = df_raw['Risk_Level'].value_counts()
            import json
            labels = risk_counts.index.tolist()
            values = risk_counts.values.tolist()
            colors = {'HIGH': '#ff4d6d', 'MEDIUM': '#f59e0b', 'LOW': '#10b981'}
            bar_html = '<div style="background:#111827;border:1px solid #1e293b;border-radius:14px;padding:1.2rem;">'
            bar_html += '<div style="font-family:Space Mono,monospace;font-size:0.65rem;letter-spacing:2px;color:#00e5ff;margin-bottom:1rem;">RISK DISTRIBUTION</div>'
            for lbl, val in zip(labels, values):
                pct = val / total * 100
                c = colors.get(lbl, '#64748b')
                bar_html += f'''
                <div style="margin-bottom:0.8rem;">
                    <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                        <span style="font-size:0.8rem;color:#94a3b8;">{lbl}</span>
                        <span style="font-family:Space Mono,monospace;font-size:0.8rem;color:{c};">{val}</span>
                    </div>
                    <div style="background:#1e293b;border-radius:4px;height:8px;">
                        <div style="width:{pct:.0f}%;background:{c};height:8px;border-radius:4px;"></div>
                    </div>
                </div>'''
            bar_html += '</div>'
            st.markdown(bar_html, unsafe_allow_html=True)

        with c2:
            if 'Contract' in df_raw.columns:
                contract_churn = df_raw.groupby('Contract')['Churn_Prediction'].apply(
                    lambda x: (x == 'Will Churn').sum() / len(x) * 100
                ).reset_index()
                bar_html2 = '<div style="background:#111827;border:1px solid #1e293b;border-radius:14px;padding:1.2rem;">'
                bar_html2 += '<div style="font-family:Space Mono,monospace;font-size:0.65rem;letter-spacing:2px;color:#00e5ff;margin-bottom:1rem;">CHURN BY CONTRACT TYPE</div>'
                for _, row in contract_churn.iterrows():
                    pct = row['Churn_Prediction']
                    bar_html2 += f'''
                    <div style="margin-bottom:0.8rem;">
                        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                            <span style="font-size:0.75rem;color:#94a3b8;">{row["Contract"]}</span>
                            <span style="font-family:Space Mono,monospace;font-size:0.8rem;color:#0ea5e9;">{pct:.0f}%</span>
                        </div>
                        <div style="background:#1e293b;border-radius:4px;height:8px;">
                            <div style="width:{pct:.0f}%;background:linear-gradient(90deg,#0284c7,#0ea5e9);height:8px;border-radius:4px;"></div>
                        </div>
                    </div>'''
                bar_html2 += '</div>'
                st.markdown(bar_html2, unsafe_allow_html=True)

        with c3:
            buckets = pd.cut(probs * 100, bins=[0, 20, 40, 60, 80, 100],
                             labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'])
            bucket_counts = buckets.value_counts().sort_index()
            bar_html3 = '<div style="background:#111827;border:1px solid #1e293b;border-radius:14px;padding:1.2rem;">'
            bar_html3 += '<div style="font-family:Space Mono,monospace;font-size:0.65rem;letter-spacing:2px;color:#00e5ff;margin-bottom:1rem;">PROBABILITY DISTRIBUTION</div>'
            grad_colors = ['#10b981', '#84cc16', '#f59e0b', '#f97316', '#ff4d6d']
            for (lbl, val), gc in zip(bucket_counts.items(), grad_colors):
                pct = val / total * 100
                bar_html3 += f'''
                <div style="margin-bottom:0.8rem;">
                    <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                        <span style="font-size:0.8rem;color:#94a3b8;">{lbl}</span>
                        <span style="font-family:Space Mono,monospace;font-size:0.8rem;color:{gc};">{val}</span>
                    </div>
                    <div style="background:#1e293b;border-radius:4px;height:8px;">
                        <div style="width:{pct:.0f}%;background:{gc};height:8px;border-radius:4px;"></div>
                    </div>
                </div>'''
            bar_html3 += '</div>'
            st.markdown(bar_html3, unsafe_allow_html=True)

        # ── Results table ──────────────────────────────────────────────────────
        st.markdown('<p class="section-head" style="margin-top:1.5rem;">// Customer Predictions</p>', unsafe_allow_html=True)

        filter_col1, filter_col2 = st.columns([2, 3])
        with filter_col1:
            risk_filter = st.selectbox("Filter by Risk Level", ["All", "HIGH", "MEDIUM", "LOW"])
        with filter_col2:
            sort_by = st.selectbox("Sort by", ["Churn_Probability (High → Low)", "Churn_Probability (Low → High)", "tenure"])

        df_display = df_raw.copy()
        if risk_filter != "All":
            df_display = df_display[df_display['Risk_Level'] == risk_filter]

        if "High → Low" in sort_by:
            df_display = df_display.sort_values('Churn_Probability', ascending=False)
        elif "Low → High" in sort_by:
            df_display = df_display.sort_values('Churn_Probability', ascending=True)
        else:
            df_display = df_display.sort_values('tenure')

        show_cols = ['tenure', 'MonthlyCharges', 'Contract', 'InternetService',
                     'Churn_Probability', 'Risk_Level', 'Churn_Prediction']
        show_cols = [c for c in show_cols if c in df_display.columns]

        st.dataframe(
            df_display[show_cols].reset_index(drop=True),
            use_container_width=True,
            height=400,
            column_config={
                "Churn_Probability": st.column_config.ProgressColumn(
                    "Churn Probability (%)",
                    min_value=0,
                    max_value=100,
                    format="%.1f%%"
                ),
                "Risk_Level": st.column_config.TextColumn("Risk Level"),
                "Churn_Prediction": st.column_config.TextColumn("Prediction"),
            }
        )

        # ── Download ───────────────────────────────────────────────────────────
        st.markdown('<p class="section-head" style="margin-top:1.5rem;">// Export Results</p>', unsafe_allow_html=True)
        csv_out = df_raw.to_csv(index=False)
        b64_out = base64.b64encode(csv_out.encode()).decode()
        st.markdown(f"""
        <a href="data:file/csv;base64,{b64_out}" download="churn_predictions.csv">
            <button style="background:linear-gradient(135deg,#0f766e,#0d9488);color:white;
            border:none;padding:0.65rem 2rem;border-radius:8px;cursor:pointer;
            font-family:DM Sans,sans-serif;font-weight:600;font-size:0.9rem;
            letter-spacing:0.5px;">
            ⬇ Download Full Predictions CSV
            </button>
        </a>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-box" style="margin-top:1.5rem;">
        ✅ <b>Analysis complete.</b> {churners} out of {total} customers are predicted to churn 
        ({churn_rate:.1f}% churn rate). Focus retention efforts on the <b>{high_risk} HIGH-risk customers</b> first.
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
        st.exception(e)
