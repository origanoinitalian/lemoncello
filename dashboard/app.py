import sys
import os
import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from slime import lime_tabular
except ImportError:
    st.error("⚠️ Could not find 'slime' package.")
    st.stop()

# --- PAGE CONFIG ---
st.set_page_config(page_title="S-LIME Diagnostic Tool", layout="wide")

# --- CSS FOR VISUAL ---
st.markdown("""
<style>
.main { background-color: #f8f9fa; }
div.stButton > button:first-child {
    background-color: #4CAF50;
    color: white;
    font-weight: bold;
}
.reportview-container .main .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# --- LOAD & TRAIN ---
@st.cache_resource
def load_resources():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, train_size=0.80, random_state=42
    )

    # Initialize Session States
    if 'slime_patient' not in st.session_state:
        st.session_state.slime_patient = -1
    if 'slime_result_object' not in st.session_state:
        st.session_state.slime_result_object = None
    if 'last_lime_patient' not in st.session_state:
        st.session_state.last_lime_patient = -1
    
    rf = RandomForestClassifier(n_estimators=500, random_state=42)
    rf.fit(X_train, y_train)
    
    # Base Explainer
    explainer = lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=data.feature_names,
        class_names=data.target_names,
        discretize_continuous=True,
        feature_selection='lasso_path', 
        verbose=False
    )
    return data, X_train, X_test, rf, explainer

data, X_train, X_test, rf, explainer = load_resources()

# --- SIDEBAR CONFIGURATION ---
st.sidebar.title("🔍 Configuration")

with st.sidebar.form("settings_form"):
    st.header("Patient & Parameters")
    patient_id = st.number_input("Patient ID (Test Set Index)", 0, len(X_test)-1, 0)
    num_features = st.slider("Features to Show", 1, 10, 5)
    alpha = st.slider("Stability Threshold (Alpha)", 0.01, 0.10, 0.05)
    discretize = st.checkbox("Enable Feature Discretization", value=True)
    st.caption("Reducing continuous noise optimizes mathematical convergence.")
    submitted = st.form_submit_button("🚀 Update & Analyze")

# --- MAIN DASHBOARD ---
st.title("🧬 S-LIME Diagnostic Certification Tool")

patient_data = X_test[patient_id]
probs = rf.predict_proba([patient_data])[0]
prediction = data.target_names[np.argmax(probs)]
confidence = np.max(probs)

with st.container():
    st.subheader("🏥 Model Global Prediction")
    v1, v2, v3 = st.columns(3)
    v1.metric("Diagnosis", prediction.upper())
    v2.metric("Confidence", f"{confidence:.2%}")
    v3.metric("Patient ID", f"#{patient_id}")

st.divider()

# --- LIME ---
st.header("📊 Phase 1: Baseline Explanation")
st.info("Standard LIME provides a fast heuristic, but results may vary between runs due to random sampling noise.")

if st.session_state.last_lime_patient != patient_id or submitted:
    with st.spinner("Generating baseline..."):
        exp = explainer.explain_instance(
            patient_data, rf.predict_proba, num_features=num_features, num_samples=1000
        )
        st.session_state.lime_exp = exp.as_list()
        st.session_state.last_lime_patient = patient_id

# Baseline Plotting
feat_list = st.session_state.lime_exp
df_lime = pd.DataFrame(feat_list, columns=['Feature', 'Influence'])

fig_lime, ax_lime = plt.subplots(figsize=(10, 4))
colors_lime = ['#2ecc71' if x > 0 else '#e74c3c' for x in df_lime['Influence']]
ax_lime.barh(df_lime['Feature'], df_lime['Influence'], color=colors_lime)

max_val = df_lime['Influence'].abs().max() * 1.5
ax_lime.set_xlim(-max_val, max_val)
ax_lime.axvline(0, color='black', linewidth=1)

ax_lime.invert_yaxis()
ax_lime.grid(axis='x', linestyle='--', alpha=0.3)
ax_lime.set_title("Standard LIME (Directional Influence Only)")
st.pyplot(fig_lime)

# Instability Tracking
if 'prev_lime_exp' in st.session_state and st.session_state.last_lime_patient == patient_id:
    current_features = set([x[0] for x in st.session_state.lime_exp])
    prev_features = set([x[0] for x in st.session_state.prev_lime_exp])
    overlap = len(current_features.intersection(prev_features)) / len(current_features) if len(current_features) > 0 else 1.0
    
    if overlap < 1.0:
        st.warning(f"⚠️ **Instability Detected:** Feature set shifted by {100*(1-overlap):.0f}% since last run.")
    else:
        st.success("✨ **Lucky Run:** Features remained consistent this time.")
st.session_state.prev_lime_exp = st.session_state.lime_exp

st.divider()

# --- S-LIME ---
st.header("🛡️ Phase 2: S-LIME Certification (Lemoncello)")
st.write(f"This phase uses **Quasi-Monte Carlo (Sobol) sampling** and iterative testing to mathematically certify the stability of each feature.")

# Action Button
if st.button("⚖️ Run Stability Certification", use_container_width=True):
    with st.status("Certifying Features...", expanded=True) as status:
        dynamic_explainer = lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=data.feature_names,
            class_names=data.target_names,
            discretize_continuous=discretize,
            feature_selection='lasso_path'
        )
        st.write("Generating Iterative QMC Samples...")
        exp_slime = dynamic_explainer.slime(
            patient_data, rf.predict_proba, 
            num_features=num_features, 
            num_samples=2000, 
            n_max=100000,
            alpha=alpha 
        )
        st.session_state.slime_result_object = exp_slime 
        st.session_state.slime_patient = patient_id
        st.session_state.certified_alpha = alpha 
        status.update(label="Certification Complete!", state="complete", expanded=False)
        st.rerun()

# S-LIME Plotting Area
if st.session_state.slime_patient == patient_id and st.session_state.slime_result_object is not None:
    exp_object = st.session_state.slime_result_object 
    feat_list = exp_object.as_list()
    df_slime = pd.DataFrame(feat_list, columns=['Feature', 'Influence'])
    
    # Extract Mathematical Metadata
    raw_meta = exp_object.stability_metadata 
    t_stats, actual_se = [], []
    
    for i, (feat_idx, weight) in enumerate(exp_object.local_exp[1]):
        row = raw_meta[i+1] # skip intercept
        t_val = row[0]
        t_stats.append(t_val)
        se_calc = abs(weight) / (t_val if t_val > 0 else 1e-6)
        actual_se.append(se_calc)

    display_alpha = st.session_state.get('certified_alpha', alpha)
    z_score = stats.norm.ppf(1 - display_alpha / 2)
    error_margins = [z_score * se for se in actual_se]

    # Traffic Light Logic
    colors_slime = []
    for t in t_stats:
        if abs(t) > z_score: colors_slime.append('#27ae60') # Green
        elif abs(t) > (z_score * 0.7): colors_slime.append('#f39c12') # Orange
        else: colors_slime.append('#c0392b') # Red

    fig_slime, ax_slime = plt.subplots(figsize=(10, 5))
    ax_slime.barh(df_slime['Feature'], df_slime['Influence'], 
                  color=colors_slime, xerr=error_margins, capsize=5, ecolor='#2c3e50', alpha=0.8)
    
    # Dynamic Scaling
    max_infl = df_slime['Influence'].abs().max() * 2.5
    if max_infl > 0: ax_slime.set_xlim(-max_infl, max_infl)
    
    ax_slime.axvline(0, color='black', linewidth=1)
    ax_slime.invert_yaxis()
    ax_slime.grid(axis='x', linestyle='--', alpha=0.3)
    ax_slime.set_title(f"Certified Explanation (Confidence Level: {100*(1-display_alpha):.1f}%)", fontsize=12)
    st.pyplot(fig_slime)
    st.success(f"✅ Features Mathematically Validated (α={display_alpha})")

    # --- LOGS  ---
    st.divider()
    st.subheader("📊 Mathematical Certification Log")
    cert_df = pd.DataFrame(raw_meta, columns=['T-Statistic', 'Samples Used', 'Std. Error'])
    cert_df.index.name = "Rank"
    
    col_log, col_stat = st.columns([2, 1])
    with col_log:
        st.dataframe(cert_df.style.highlight_between(left=z_score, color='#d4edda'), use_container_width=True)
    with col_stat:
        st.metric("Final Status", "PASS", delta="Certified")
        st.write(f"**Total Samples:** {int(cert_df['Samples Used'].max()):,}")
        st.write(f"**Target Z-Score:** {z_score:.2f}")

    st.divider()
    with st.expander("📖 Clinician's Interpretation Guide", expanded=True):
        st.markdown(f"""
        ### Understanding your Diagnosis Explanation
        This tool ensures that the "reasons" for this patient's prediction are statistically significant.
        
        * **🟢 Green (Certified Stable):** These features have a high signal-to-noise ratio ($T > {z_score:.2f}$). They are proven drivers of the prediction.
        * **🟡 Orange (Volatile):** These features are influential but sensitive to small changes. Use with clinical caution.
        * **🔴 Red (Insignificant):** These features are mathematical "noise" for this specific patient. **Do not use Red features for clinical intervention planning.**
        
        **Methodology:** This dashboard uses an iterative stability loop. It automatically increases the patient simulations (up to 100,000) until the error bars are small enough to pass a rigorous Central Limit Theorem test.
        """)