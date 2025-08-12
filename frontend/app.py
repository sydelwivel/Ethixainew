# frontend/app.py
"""
EthixAI - Streamlit frontend (final)
- No experimental_* usage
- Sidebar colored green (#007B55) with white text
- Preserves all original features
- Synthetic generation verification included
- Uses local community backend for crowdsourcing
"""

import sys
import os
from pathlib import Path
import io
import pickle
import tempfile
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------
# Ensure backend folder is on sys.path
# ---------------------------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent  # assumes frontend/ and backend/ are siblings
BACKEND_PATH = PROJECT_ROOT / "backend"
if str(BACKEND_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PATH))

# ---------------------------
# Import backend modules (graceful)
# ---------------------------
_missing = []
def _try_import(name):
    try:
        module = __import__(name)
        return module
    except Exception:
        _missing.append(name)
        return None

audit = _try_import("audit")
privacy = _try_import("privacy")
synthetic = _try_import("synthetic")
retrain = _try_import("retrain")
scorecard = _try_import("scorecard")
drift = _try_import("drift")
explain = _try_import("explain")
simulator = _try_import("simulator")
community = _try_import("community")

# ---------------------------
# Embedded Corporate (green) CSS
# ---------------------------
st.set_page_config(page_title="EthixAI - Auditor", layout="wide")
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    html, body, [class*="css"]  {{
        font-family: 'Poppins', sans-serif;
        background-color: #f4f7f9;
    }}
    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: #007B55;  /* green per request */
        color: #fff;
    }}
    section[data-testid="stSidebar"] label, section[data-testid='stSidebar'] div, section[data-testid='stSidebar'] p {{
        color: #fff !important;
    }}
    /* Card style */
    .card {{
        background: #ffffff;
        border-radius: 8px;
        padding: 18px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.08);
        margin-bottom: 16px;
    }}
    /* Buttons */
    div.stButton > button, button[kind="primary"] {{
        background-color: #007B55;
        color: #ffffff;
        border-radius: 6px;
        padding: 8px 14px;
    }}
    div.stButton > button:hover {{
        background-color: #0a8a63;
    }}
    /* Metrics */
    .metric-card {{
        background: #ffffff;
        border-radius: 8px;
        padding: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.06);
    }}
    .muted {{ color: #6b7280; font-size: 13px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Helpers
# ---------------------------
def detect_sensitive_attributes(df: pd.DataFrame):
    sensitive_keywords = [
        "gender", "sex", "race", "ethnicity", "age",
        "religion", "disability", "nationality",
        "marital_status", "income", "sexual_orientation"
    ]
    detected = []
    for col in df.columns:
        low = col.lower().replace(" ", "_")
        for kw in sensitive_keywords:
            if kw in low:
                detected.append(col)
                break
    return detected

def binarize_labels(series: pd.Series):
    s = pd.Series(series).copy()
    uniq = pd.Series(s.dropna().unique())
    set_uniq = set(uniq)
    if set_uniq <= {0,1} or set_uniq <= {-1,1}:
        return s.astype(int).replace({-1:1})
    # choose positive label heuristically
    if any(isinstance(x, str) for x in uniq):
        for val in uniq:
            lv = str(val).lower()
            if ">" in lv or "yes" in lv or lv in ("1","true","t"):
                pos = val
                break
        else:
            pos = uniq.iloc[-1]
    else:
        pos = max(uniq)
    return s.apply(lambda x: 1 if x == pos else 0).astype(int)

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

# Helper: show small card
def start_card():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

def end_card():
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Sidebar / Navigation
# ---------------------------
logo_path = os.environ.get("ETHIXAI_LOGO_PATH", "")  # optional environment variable
if logo_path and os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=140)

st.sidebar.title("EthixAI")
st.sidebar.caption("Ethical AI Auditor")

# Initialize page state in session_state to avoid deprecated query param mutation
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

menu = st.sidebar.radio("Go to", [
    "Dashboard",
    "Fairness Audit",
    "Proxy Bias",
    "Privacy Risk",
    "Synthetic Data",
    "Train & Evaluate",
    "Scorecard",
    "Simulator",
    "Drift",
    "Community"
], index=["Dashboard","Fairness Audit","Proxy Bias","Privacy Risk","Synthetic Data","Train & Evaluate","Scorecard","Simulator","Drift","Community"].index(st.session_state.get("page", "Dashboard")))

# store menu selection so quick buttons can change it safely
st.session_state.page = menu

if _missing:
    st.sidebar.error(f"Missing backend modules: {_missing}. Some features disabled.")

# ---------------------------
# Dataset upload / load
# ---------------------------
st.header("EthixAI — Complete Ethical AI Auditor")

uploaded = st.file_uploader("Upload dataset (CSV / Excel / JSON)", type=["csv","xlsx","json"])
if uploaded:
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        elif uploaded.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded)
        else:
            df = pd.read_json(uploaded)
        st.session_state["df"] = df
        st.success("Dataset loaded.")
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")

if "df" not in st.session_state:
    st.session_state["df"] = None

df = st.session_state["df"]

# ---------------------------
# Dashboard page
# ---------------------------
if st.session_state.page == "Dashboard":
    start_card()
    st.subheader("Dataset Preview & Quick Actions")
    if df is None:
        st.info("Upload a dataset to begin auditing.")
    else:
        st.write(df.head())
        auto_sensitive = detect_sensitive_attributes(df)
        if auto_sensitive:
            st.info(f"Auto-detected sensitive columns: {auto_sensitive}")
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            if st.button("Run Fairness Audit"):
                st.session_state.page = "Fairness Audit"
                st.rerun()
        with col2:
            if st.button("Generate Synthetic (quick)"):
                st.session_state.page = "Synthetic Data"
                st.rerun()
        with col3:
            if st.button("Train Model (quick)"):
                st.session_state.page = "Train & Evaluate"
                st.rerun()
    end_card()

# ---------------------------
# Fairness Audit page
# ---------------------------
if st.session_state.page == "Fairness Audit":
    start_card()
    st.subheader("Fairness Audit")
    if df is None:
        st.warning("Upload dataset first.")
    else:
        cols = list(df.columns)
        # choose sensible default for target (last column)
        default_target_idx = min(len(cols)-1, 0) if cols else 0
        target = st.selectbox("Target column", cols, index=default_target_idx)
        auto_sensitive = detect_sensitive_attributes(df)
        sensitive = st.selectbox("Sensitive attribute", auto_sensitive + cols if auto_sensitive else cols)
        if st.button("Run Fairness Audit"):
            if audit is None:
                st.error("Audit backend missing.")
            else:
                # determine y_true and y_pred
                y_true = binarize_labels(df[target])

                pred_cols = [c for c in df.columns if "pred" in c.lower() or c.lower().endswith("_pred")]
                prob_cols = [c for c in df.columns if ("prob" in c.lower()) or ("score" in c.lower())]

                if pred_cols:
                    y_pred = binarize_labels(df[pred_cols[0]])
                    st.info(f"Using predictions from column: {pred_cols[0]}")
                elif prob_cols:
                    y_pred = (pd.to_numeric(df[prob_cols[0]], errors="coerce") >= 0.5).astype(int)
                    st.info(f"Using probabilities from column: {prob_cols[0]} (threshold 0.5)")
                else:
                    # fallback: train internal model excluding sensitive column
                    try:
                        from sklearn.ensemble import RandomForestClassifier
                        from sklearn.model_selection import train_test_split
                        feat_cols = [c for c in df.columns if c not in [target, sensitive]]
                        if len(feat_cols) == 0:
                            st.error("No features available for training fallback model.")
                            y_pred = y_true
                        else:
                            X = pd.get_dummies(df[feat_cols], drop_first=True)
                            y = y_true
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                            clf = RandomForestClassifier(n_estimators=100, random_state=42)
                            clf.fit(X_train, y_train)
                            y_pred = pd.Series(clf.predict(X), index=df.index)
                            st.info("Trained internal model (sensitive excluded) for audit.")
                    except Exception as e:
                        st.error(f"Fallback model failed: {e}")
                        y_pred = y_true

                metrics = audit.run_fairness_audit(y_true, y_pred, df[sensitive])
                # interpret and show
                def interpret(results):
                    out = {}
                    for metric, val in results.items():
                        a = abs(val)
                        if a <= 0.05:
                            level = "✅ Fair"; col = "green"; expl = "Little to no measurable bias."
                        elif a <= 0.15:
                            level = "⚠️ Mild Bias"; col = "orange"; expl = "Small differences — review."
                        else:
                            level = "❌ Significant Bias"; col = "red"; expl = "Large differences — action required."
                        out[metric] = {"value": round(val,4), "level": level, "color": col, "explanation": expl}
                    return out

                interp = interpret(metrics)
                for m, info in interp.items():
                    st.markdown(f"**{m}:** {info['value']}   {info['level']}")
                    st.markdown(f"<small style='color:{info['color']}'>{info['explanation']}</small>", unsafe_allow_html=True)
                    st.progress(min(abs(info['value']), 1.0))

                # group-wise stats
                st.subheader("Group statistics")
                grp = df[sensitive].astype(str)
                pos_rate = pd.Series(y_pred).groupby(grp).mean().rename("Positive Rate")
                counts = grp.value_counts().rename("Count")
                gdf = pd.concat([counts, pos_rate], axis=1).fillna(0).reset_index().rename(columns={"index":"Group"})
                st.table(gdf)
    end_card()

# ---------------------------
# Proxy Bias page
# ---------------------------
if st.session_state.page == "Proxy Bias":
    start_card()
    st.subheader("Proxy Bias Detection")
    if df is None:
        st.warning("Upload dataset first.")
    else:
        sensitive_candidates = detect_sensitive_attributes(df)
        sensitive = st.selectbox("Sensitive attribute", sensitive_candidates + list(df.columns) if sensitive_candidates else list(df.columns))
        top_n = st.number_input("Top N proxies", min_value=1, max_value=20, value=5)
        if st.button("Detect Proxy Bias"):
            if audit is None:
                st.error("Audit backend missing.")
            else:
                proxies = audit.detect_proxy_bias(df, sensitive, top_n=int(top_n))
                rows = []
                for i, p in enumerate(proxies):
                    rows.append({
                        "Rank": i+1,
                        "Column": p["column"],
                        "Strength": round(p["strength"],4),
                        "Risk Level": p["risk_level"],
                        "Why": p["reason"],
                        "Suggested Action": p["suggestion"]
                    })
                st.dataframe(pd.DataFrame(rows))
    end_card()

# ---------------------------
# Privacy Risk page
# ---------------------------
if st.session_state.page == "Privacy Risk":
    start_card()
    st.subheader("Privacy Risk Audit (Top 5)")
    if df is None:
        st.warning("Upload dataset first.")
    else:
        if st.button("Run Privacy Audit"):
            if privacy is None:
                st.error("Privacy backend missing.")
            else:
                risky = privacy.reidentifiable_features(df, top_n=5)
                if not risky:
                    st.success("No high-risk combos found (thresholds applied).")
                else:
                    rows = []
                    for i, r in enumerate(risky):
                        rows.append({
                            "Rank": i+1,
                            "Combination": ", ".join(r["combination"]),
                            "Uniqueness (%)": f"{r['unique_ratio']*100:.1f}",
                            "Why": r["reason"],
                            "Suggested Action": r["suggestion"]
                        })
                    st.table(pd.DataFrame(rows))
    end_card()

# ---------------------------
# Synthetic Data page (generation + verification)
# ---------------------------
if st.session_state.page == "Synthetic Data":
    start_card()
    st.subheader("Synthetic Data Generation & Verification")
    if df is None:
        st.warning("Upload dataset first.")
    else:
        cols = list(df.columns)
        target_col = st.selectbox("Select target column to balance", cols)
        sensitive_candidates = detect_sensitive_attributes(df)
        sensitive_to_check = st.selectbox("Sensitive column to verify retention (optional)", ["None"] + sensitive_candidates) if sensitive_candidates else "None"
        top_n_preview = st.number_input("Preview rows after generation", min_value=3, max_value=50, value=5)

        if st.button("Generate synthetic dataset"):
            if synthetic is None:
                st.error("Synthetic backend missing.")
            else:
                try:
                    # show pre-balance class distribution
                    before_counts = df[target_col].value_counts(dropna=False)
                    st.write("#### Class distribution (original)")
                    st.write(before_counts)

                    syn_df = synthetic.generate_synthetic_data(df, target_col)
                    if syn_df is None:
                        st.error("Synthetic generator returned nothing.")
                    else:
                        st.success("Synthetic dataset generated.")
                        # basic checks
                        st.write("#### Preview (synthetic)")
                        st.dataframe(syn_df.head(top_n_preview))

                        # check class balance after
                        after_counts = syn_df[target_col].value_counts(dropna=False)
                        st.write("#### Class distribution (synthetic)")
                        st.write(after_counts)

                        # retention check for sensitive column
                        if sensitive_to_check and sensitive_to_check != "None":
                            retained = sensitive_to_check in syn_df.columns
                            st.write(f"Sensitive column '{sensitive_to_check}' retained in synthetic dataset: {retained}")
                            if retained:
                                st.write(syn_df[sensitive_to_check].value_counts(dropna=False))
                            else:
                                st.error(f"Sensitive column '{sensitive_to_check}' missing in synthetic output!")

                        # offer download
                        csv_bytes = df_to_csv_bytes(syn_df)
                        st.download_button("Download synthetic CSV", data=csv_bytes, file_name="synthetic_dataset.csv", mime="text/csv")
                        st.session_state["df"] = syn_df

                except Exception as e:
                    st.error(f"Synthetic generation failed: {e}")
    end_card()

# ---------------------------
# Train & Evaluate page (classification & regression)
# ---------------------------
if st.session_state.page == "Train & Evaluate":
    start_card()
    st.subheader("Train & Evaluate Model")
    if df is None:
        st.warning("Upload dataset first.")
    else:
        cols = list(df.columns)
        target = st.selectbox("Target column", cols)
        problem_type = st.selectbox("Problem type", ["Auto-detect", "Classification", "Regression"])
        exclude_sensitive = st.checkbox("Exclude detected sensitive column from features (recommended)", value=True)
        sensitive_candidates = detect_sensitive_attributes(df)
        sensitive = None
        if sensitive_candidates:
            sensitive = st.selectbox("Detected sensitive attribute (for exclusion/fairness)", ["None"] + sensitive_candidates)
        test_size = st.slider("Test set fraction", min_value=0.1, max_value=0.5, value=0.3, step=0.05)

        if st.button("Train & Evaluate"):
            if retrain is None:
                st.error("Retrain backend missing.")
            else:
                try:
                    X_all = df.drop(columns=[target])
                    if exclude_sensitive and sensitive and sensitive != "None":
                        if sensitive in X_all.columns:
                            X_all = X_all.drop(columns=[sensitive])
                    y_all = df[target]

                    # detect type
                    if problem_type == "Auto-detect":
                        if pd.api.types.is_numeric_dtype(y_all):
                            detected = "Regression"
                        else:
                            detected = "Classification"
                    else:
                        detected = problem_type

                    st.info(f"Detected problem type: {detected}")

                    # prepare features
                    X = pd.get_dummies(X_all, drop_first=True)
                    y = y_all
                    if detected == "Classification":
                        y = binarize_labels(y_all)

                    # train/test split
                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                    # call retrain.train_and_evaluate
                    model, metrics, preds = retrain.train_and_evaluate(X_train, y_train, X_test, y_test, problem_type=detected)
                    st.success("Training complete — summary metrics:")
                    st.write(metrics)

                    # allow download of model
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tf:
                        pickle.dump(model, tf)
                        tf_path = tf.name
                    with open(tf_path, "rb") as f:
                        st.download_button("Download trained model (.pkl)", data=f, file_name="trained_model.pkl", mime="application/octet-stream")

                    # if classification and audit available & sensitive selected -> run fairness on test set
                    if detected == "Classification" and audit is not None and sensitive and sensitive != "None":
                        try:
                            # try to obtain sensitive values aligned with test set indices
                            sens_series = None
                            if sensitive in df.columns:
                                # df index preserved earlier — map using X_test index where possible
                                # use intersection by position: X_test rows correspond to original df rows only if no reindexing
                                # We'll try to infer using boolean mask via index intersection
                                try:
                                    sens_series = df.loc[X_test.index, sensitive]
                                except Exception:
                                    # fallback: use full column (not ideal)
                                    sens_series = df[sensitive]
                            if sens_series is not None:
                                fairness_metrics = audit.run_fairness_audit(y_test, preds, sens_series)
                                st.subheader("Fairness on test set")
                                st.write(fairness_metrics)
                        except Exception as e:
                            st.warning(f"Fairness on test set failed: {e}")

                    # store preds
                    st.session_state["last_model_preds"] = pd.Series(preds, index=y_test.index)

                except Exception as e:
                    st.error(f"Training failed: {e}")
    end_card()

# ---------------------------
# Scorecard page
# ---------------------------
if st.session_state.page == "Scorecard":
    start_card()
    st.subheader("Generate Ethical Scorecard")
    if df is None:
        st.warning("Upload dataset first.")
    else:
        if st.button("Generate Scorecard PDF"):
            if scorecard is None:
                st.error("Scorecard backend missing.")
            else:
                try:
                    demo_metrics = {"Accuracy": 0.90, "Demographic Parity": 0.10, "Equalized Odds": 0.05}
                    out = "reports/scorecard.pdf"
                    os.makedirs("reports", exist_ok=True)
                    scorecard.generate_scorecard(demo_metrics, out)
                    st.success(f"Scorecard saved to {out}")
                    with open(out, "rb") as f:
                        st.download_button("Download Scorecard PDF", data=f, file_name="scorecard.pdf", mime="application/pdf")
                except Exception as e:
                    st.error(f"Scorecard failed: {e}")
    end_card()

# ---------------------------
# Simulator page
# ---------------------------
if st.session_state.page == "Simulator":
    start_card()
    st.subheader("Bias Simulator")
    if df is None:
        st.warning("Upload dataset first.")
    else:
        st.write("Adjust bias strength and simulate predictions.")
        bias_strength = st.slider("Bias strength", -1.0, 1.0, 0.0, 0.05)
        if simulator is None:
            st.info("Simulator backend missing.")
        else:
            try:
                preds_sim, metrics_sim = simulator.simulate_bias_effect(df, target if 'target' in locals() else df.columns[0],
                                                                       sensitive if 'sensitive' in locals() else (detect_sensitive_attributes(df)[0] if detect_sensitive_attributes(df) else df.columns[0]),
                                                                       bias_strength)
                st.write(metrics_sim)
            except Exception as e:
                st.error(f"Simulator failed: {e}")
    end_card()

# ---------------------------
# Drift page
# ---------------------------
if st.session_state.page == "Drift":
    start_card()
    st.subheader("Data Drift")
    if df is None:
        st.warning("Upload dataset first.")
    else:
        prev_score = st.number_input("Previous model score", value=0.95, step=0.01)
        curr_score = st.number_input("Current model score", value=0.90, step=0.01)
        if st.button("Check Drift"):
            if drift is None:
                st.error("Drift backend missing.")
            else:
                try:
                    drifted = drift.detect_drift(prev_score, curr_score)
                    st.write("Drift detected:", drifted)
                except Exception as e:
                    st.error(f"Drift check failed: {e}")
    end_card()

# ---------------------------
# Community page (local CSV storage)
# ---------------------------
if st.session_state.page == "Community":
    start_card()
    st.subheader("Community / Crowdsourcing")
    if community is None:
        st.info("Community backend missing.")
    else:
        # Show existing submissions
        subs = community.list_submissions()
        st.write("### Recent submissions")
        if subs.empty:
            st.info("No submissions yet.")
        else:
            st.dataframe(subs.sort_values("timestamp", ascending=False).head(20))

        st.write("---")
        st.write("### Submit to the community")
        name = st.text_input("Your name")
        email = st.text_input("Your email (optional)")
        role = st.text_input("Your role / organization (optional)")
        sub_type = st.selectbox("Submission type", ["report", "dataset", "comment"])
        content = st.text_area("Describe your submission (what, why, notes)")
        attached = st.file_uploader("Attach a file (optional)", type=["csv","xlsx","json","txt","pdf"], key="community_attach")
        if st.button("Submit"):
            # save attached file if present
            attached_filename = None
            if attached is not None:
                attach_dir = Path(BACKEND_PATH) / "data"
                attach_dir.mkdir(parents=True, exist_ok=True)
                attached_filename = f"{int(datetime.utcnow().timestamp())}_{attached.name}"
                with open(attach_dir / attached_filename, "wb") as f:
                    f.write(attached.getbuffer())
            # call backend
            try:
                community.add_submission(
                    name=name or "Anonymous",
                    email=email or "",
                    role=role or "",
                    submission_type=sub_type,
                    content=content or "",
                    attached_filename=attached_filename
                )
                st.success("Submission saved. Thank you!")
            except Exception as e:
                st.error(f"Failed to save submission: {e}")
    end_card()

# ---------------------------
# End of app
# ---------------------------
