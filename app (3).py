import json
import pickle

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prediksi Risiko Stroke", page_icon="🧠")


@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model_meta.json") as f:
        meta = json.load(f)
    return model, meta


model, meta = load_artifacts()

st.title("🧠 Prediksi Risiko Stroke")
st.caption(f"Model: {meta['model_name']} | Threshold keputusan: {meta['best_threshold']:.3f}")

inputs = {}
col1, col2 = st.columns(2)
for i, c in enumerate(meta["feature_order"]):
    box = col1 if i % 2 == 0 else col2
    with box:
        if c in meta["numerical"]:
            info = meta["numerical"][c]
            inputs[c] = st.number_input(c, min_value=float(info["min"]), max_value=float(info["max"]),
                                        value=float(info["median"]))
        else:
            options = [o for o in meta["categorical"][c] if o != "Other"]
            inputs[c] = st.selectbox(c, options)

if st.button("Prediksi", type="primary"):
    row = pd.DataFrame([inputs], columns=meta["feature_order"])
    for c in meta["categorical"]:
        row[c] = row[c].astype(str)          # harus string, sama seperti saat training
    proba = float(model.predict_proba(row)[0, 1])
    thr = meta["best_threshold"]
    st.metric("Probabilitas stroke", f"{proba*100:.1f}%")
    if proba >= thr:
        st.error("Berisiko tinggi stroke (di atas threshold). Disarankan konsultasi ke tenaga medis.")
    else:
        st.success("Risiko rendah (di bawah threshold).")
    st.caption("Alat bantu edukasi tugas kuliah, bukan pengganti diagnosis dokter.")
