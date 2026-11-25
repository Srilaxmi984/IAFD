"""Streamlit demo: score one transaction with the two final models and explain the score with SHAP.

    streamlit run app/app.py
Needs models/final_single.joblib and models/final_ensemble.joblib (python scripts/final_evaluate.py).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from iafd.config import load_config, resolve
from iafd.data import TYPES
from iafd.serving import (OPS, OP_LABEL, default_background, decide, explain_row, load_bundles, make_row, score_row)

st.set_page_config(page_title="Explainable fraud detection", layout="wide")
st.title("Explainable AI-Based Fraud Detection Using Ensemble Learning and SHAP")
st.caption("Two final models trained on PaySim with a chronological split. PaySim is synthetic: the scores show how the "
           "method behaves on this data, not how real bank fraud would look.")

cfg = load_config()


@st.cache_resource
def get_bundles():
    return load_bundles(resolve(cfg["paths"]["models_dir"]))


@st.cache_resource
def get_background(key: str):
    return default_background(get_bundles()[key], 100, resolve(cfg["paths"]["dev_parquet"]))


try:
    bundles = get_bundles()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

PRESETS = {
    "Full-balance TRANSFER (fraud-like)": ("TRANSFER", 250000.0, 250000.0, 0.0, 0.0, 0.0),
    "Ordinary CASH_OUT": ("CASH_OUT", 5000.0, 80000.0, 75000.0, 20000.0, 25000.0),
    "Small PAYMENT": ("PAYMENT", 64.0, 900.0, 836.0, 0.0, 0.0),
}
FIELDS = ("type", "amount", "old_o", "new_o", "old_d", "new_d")
for k, v in zip(FIELDS, PRESETS["Full-balance TRANSFER (fraud-like)"]):
    st.session_state.setdefault(k, v)


def load_preset(name):
    for k, v in zip(FIELDS, PRESETS[name]):
        st.session_state[k] = v


with st.sidebar:
    st.header("Model")
    which = st.radio("Final model", ["single", "ensemble"],
                     format_func=lambda k: {"single": "Single model (best under high imbalance)", "ensemble": "Ensemble"}[k])
    op = st.selectbox("Operating point", OPS, format_func=lambda o: OP_LABEL[o])
    st.header("Examples")
    for name in PRESETS:
        st.button(name, on_click=load_preset, args=(name,))
    b = bundles[which]
    desc = (f"{b['model']} ({b['strategy']} class weight)" if which == "single"
            else " + ".join(f"{m['model']}({m['strategy']})" for m in b["members"]) + f", {b['aggregator']}")
    st.caption(f"Using: {desc}. Features: {b['feature_set']}. Type gating: "
               f"{'on, scored types ' + ', '.join(b['allowed_types']) if b.get('gate') else 'off'}.")

c1, c2, c3 = st.columns(3)
tx_type = c1.selectbox("Transaction type", list(TYPES), key="type")
amount = c1.number_input("Amount", min_value=0.0, step=100.0, key="amount")
old_o = c2.number_input("Sender balance before", min_value=0.0, step=100.0, key="old_o")
new_o = c2.number_input("Sender balance after", min_value=0.0, step=100.0, key="new_o")
old_d = c3.number_input("Receiver balance before", min_value=0.0, step=100.0, key="old_d")
new_d = c3.number_input("Receiver balance after", min_value=0.0, step=100.0, key="new_d")

if st.button("Score this transaction", type="primary"):
    row = make_row(tx_type, amount, old_o, new_o, old_d, new_d)
    res = score_row(b, row)
    flagged, thr = decide(b, res["score"], op)
    m1, m2, m3 = st.columns(3)
    m1.metric("Fraud score", f"{res['score']:.4f}")
    m2.metric("Threshold", "never flags" if not np.isfinite(thr) else f"{thr:.4f}")
    m3.metric("Decision", "FLAG for review" if flagged else "No alert")
    if res["gated"]:
        st.info(f"Type {tx_type} had no fraud in the training data, so it is scored 0 and never flagged. "
                f"This is a property of PaySim, not a guarantee for other data.")
    else:
        with st.spinner("Computing SHAP explanation..."):
            sv, base, scale = explain_row(b, row, get_background(which))
        top = sv.reindex(sv.abs().sort_values(ascending=False).index).head(10)[::-1]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(top.index, top.values, color=["#d62728" if v > 0 else "#1f77b4" for v in top.values])
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_xlabel(f"SHAP value ({scale}); red raises the fraud score, blue lowers it")
        ax.set_title(f"Why this score? (starting point {base:.3f} for a typical transaction)")
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("SHAP shows how each feature moved this score away from a typical transaction. "
                   "It explains the model, not whether the transaction is truly fraud.")
