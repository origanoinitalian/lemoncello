import sys
import os
import streamlit as st
import numpy as np
import pandas as pd
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
st.set_page_config(page_title="S-LIME Dashboard", layout="wide")

# --- CSS FOR VISUAL FEEDBACK ---
st.markdown("""
<style>
.main { background-color: #f8f9fa; }
div.stButton > button:first-child {
    background-color: #4CAF50;
    color: white;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# --- 1. LOAD & TRAIN (Cached) ---
@st.cache_resource
def load_resources():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, train_size=0.80, random_state=42
    )
    
    rf = RandomForestClassifier(n_estimators=500, random_state=42)
    rf.fit(X_train, y_train)
    
    # Initialize Explainer Once
    explainer = lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=data.feature_names,
        class_names=data.target_names,
        discretize_continuous=True, 
        verbose=False
    )
    return data, X_test, rf, explainer

data, X_test, rf, explainer = load_resources()

# --- 2. SIDEBAR CONFIGURATION ---
st.sidebar.title("🔍 Configuration")

with st.sidebar.form("settings_form"):
    st.header("Patient & Parameters")
    
    # Inputs
    patient_id = st.number_input("Patient ID (Test Set Index)", 0, len(X_test)-1, 0)
    num_features = st.slider("Features to Show", 1, 10, 5)
    alpha = st.slider("Stability Threshold (Alpha)", 0.01, 0.10, 0.05)
    
    # This button now clearly states it triggers a reload
    submitted = st.form_submit_button("🚀 Update & Analyze")

# --- 3. MAIN DASHBOARD ---
st.title("🧬 S-LIME Diagnostic Tool")

# Get Patient Data
patient_data = X_test[patient_id]
probs = rf.predict_proba([patient_data])[0]
prediction = data.target_names[np.argmax(probs)]
confidence = np.max(probs)

# --- SECTION: VITALS ---
col1, col2 = st.columns([1, 3])
with col1:
    # Diagnosis Card
    if prediction == 'malignant':
        st.error(f"### {prediction.upper()}\nConfidence: **{confidence:.2%}**")
    else:
        st.success(f"### {prediction.upper()}\nConfidence: **{confidence:.2%}**")
    
    st.info(f"**Patient ID:** {patient_id}")

with col2:
    # Raw Data Peek
    with st.expander("📋 View Patient Medical Records", expanded=False):
        df = pd.DataFrame([patient_data], columns=data.feature_names)
        st.dataframe(df)

st.divider()

# --- SECTION: EXPLANATIONS ---
left_col, right_col = st.columns(2)

# --- LEFT: STANDARD LIME (AUTO-RUNS) ---
# We auto-run this because it is fast. This gives instant feedback.
with left_col:
    st.subheader("📊 Standard LIME (Unstable)")
    
    # Run Standard LIME
    exp = explainer.explain_instance(
        patient_data, rf.predict_proba, num_features=num_features, num_samples=1000
    )
    
    # Plotting
    feat_list = exp.as_list()
    features = [x[0] for x in feat_list]
    scores = [x[1] for x in feat_list]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['green' if x > 0 else 'red' for x in scores]
    ax.barh(features, scores, color=colors)
    ax.set_title("Standard Explanation")
    st.pyplot(fig)
    
    st.warning("⚠️ Note: If you reload, these bars might shift due to sampling noise.")

# --- RIGHT: S-LIME (BUTTON TRIGGERED) ---
# We keep this manual because it is computationally heavy
with right_col:
    st.subheader("🛡️ S-LIME (Stabilized)")
    
    # Use session state to remember if we ran S-LIME for this specific patient
    if 'slime_patient' not in st.session_state:
        st.session_state.slime_patient = -1
        
    # Button to trigger S-LIME
    if st.button("Generate S-LIME Certification"):
        with st.spinner(f"Running Stability Tests (Alpha={alpha})..."):
            exp_slime = explainer.slime(
                patient_data, rf.predict_proba, 
                num_features=num_features, 
                num_samples=1000, 
                alpha=alpha # Fixed: Now using the slider value!
            )
            # Save result to session state
            st.session_state.slime_result = exp_slime.as_list()
            st.session_state.slime_patient = patient_id

    # If we have a result for THIS patient, show it
    if st.session_state.slime_patient == patient_id:
        feat_list = st.session_state.slime_result
        features = [x[0] for x in feat_list]
        scores = [x[1] for x in feat_list]
        
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['green' if x > 0 else 'red' for x in scores]
        ax.barh(features, scores, color=colors)
        ax.set_title(f"Stabilized Explanation (alpha={alpha})")
        st.pyplot(fig)
        st.success("✅ Certified Stable via Hypothesis Testing")
    else:
        st.info("👈 Click to verify stability.")