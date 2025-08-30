import streamlit as st
import requests
import os
from config import API_URL

st.set_page_config(page_title="💊 AI Medical Prescription Verifier", layout="wide")
st.title("💊 AI Medical Prescription Verifier (Granite Edition)")
st.caption("Educational demo. Not medical advice.")

# Token inputs (optional; you can also set env vars before running)
with st.expander("🔐 Hugging Face Token"):
    hf_token = st.text_input("Hugging Face Token", type="password")

col1, col2 = st.columns([2,1])
with col1:
    prescription_text = st.text_area(
        "Paste prescription text",
        placeholder="e.g., Paracetamol 500 mg twice daily + Ibuprofen 400 mg every 8 hours for 3 days"
    )
with col2:
    age = st.number_input("Patient age", min_value=0, max_value=120, value=30)

st.markdown("**OR** add structured drugs below (overrides text parsing).")
with st.expander("➕ Add drugs manually"):
    if "manual_drugs" not in st.session_state:
        st.session_state.manual_drugs = []
    drug_name = st.text_input("Drug name (e.g., Paracetamol)")
    dose_mg = st.number_input("Dose (mg)", min_value=0, max_value=10000, value=0)
    freq = st.number_input("Frequency per day", min_value=0, max_value=12, value=0)
    if st.button("Add Drug"):
        st.session_state.manual_drugs.append({
            "drug": drug_name, "dose_mg": int(dose_mg) if dose_mg else None,
            "frequency_per_day": int(freq) if freq else None
        })
    if st.session_state.manual_drugs:
        st.table(st.session_state.manual_drugs)
        if st.button("Clear Drugs"):
            st.session_state.manual_drugs = []

if st.button("Analyze"):
    payload = {
        "age": int(age),
        "prescription_text": prescription_text if prescription_text.strip() else None,
        "drugs": st.session_state.manual_drugs if st.session_state.manual_drugs else None,
        "hf_token": hf_token or None
    }
    try:
        r = requests.post(f"{API_URL}/analyze", json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()

        st.subheader("✅ Extracted / Input Drugs")
        st.json(data.get("drugs_parsed", []))

        st.subheader("⚠️ Interactions")
        inter = data.get("interactions", [])
        if inter:
            for item in inter:
                st.warning(f"{item['pair'][0]} + {item['pair'][1]} — {item['severity']}: {item['note']}")
        else:
            st.info("No interactions found in demo dataset.")

        st.subheader("📏 Dosage Guidance")
        st.json(data.get("dosage_guidance", {}))

        st.subheader("🔔 Warnings")
        warns = data.get("warnings", [])
        if warns:
            for w in warns:
                st.error(f"{w['drug']}: {w['issue']} (computed {w['computed_mg_per_day']} mg/day, max {w['max_daily_mg']} mg/day)")
        else:
            st.info("No dosage warnings computed.")

        st.subheader("🔁 Alternatives")
        st.json(data.get("alternatives", {}))

    except Exception as e:
        st.error(f"Request failed: {e}")

st.sidebar.markdown("**Backend**")
st.sidebar.write(f"API: {API_URL}")
st.sidebar.info("Start FastAPI first, then click Analyze.")
