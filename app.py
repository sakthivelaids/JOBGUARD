import streamlit as st
import pandas as pd
from model_utils import build_demo_dataset, preprocess_df, train_model, predict_text

st.set_page_config(page_title="Fake Job Predictor", page_icon="🛡️", layout="wide")

st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(135deg, #0b1020 0%, #111827 100%);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            color: #f8fafc;
        }
        .stButton > button {
            background: linear-gradient(90deg, #22c55e, #06b6d4);
            color: white;
            border: none;
            border-radius: 10px;
            font-weight: 600;
        }
        .stFileUploader label, .stTextArea label {
            color: #e2e8f0;
        }
        .metric-card {
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid rgba(148, 163, 184, 0.3);
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        }
        .small-note {
            color: #cbd5e1;
            font-size: 0.92rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛡️ Fake Job Posting Detector")
st.caption("Detect scammy job posts using a trained machine learning model.")

if 'model' not in st.session_state:
    st.session_state['model'] = None
    st.session_state['vectorizer'] = None
    st.session_state['training_status'] = "Not trained"

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="metric-card"><h3>Dataset</h3><p>Upload a CSV with job text and a fraudulent label</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><h3>Model</h3><p>TF-IDF + Naive Bayes or Decision Tree</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><h3>Status</h3><p>{st.session_state.training_status}</p></div>', unsafe_allow_html=True)

st.markdown("---")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset preview")
    st.dataframe(df.head(), use_container_width=True)
    st.caption(f"Rows: {len(df)} | Columns: {len(df.columns)}")

    algo = st.selectbox("Choose model", ["MultinomialNB", "DecisionTree"], index=0)
    if st.button("Train model"):
        try:
            with st.spinner("Training model, please wait..."):
                X_train, X_test, y_train, y_test, vect = preprocess_df(df)
                model = train_model(X_train, y_train, algorithm=algo)
                acc = model.score(X_test, y_test)
                st.session_state['model'] = model
                st.session_state['vectorizer'] = vect
                st.session_state['training_status'] = f"Trained ({acc:.2f} accuracy)"
            st.success(f"Model ready. Test accuracy: {acc:.3f}")
        except Exception as exc:
            st.error(f"Training failed: {exc}")

else:
    st.info("No CSV uploaded yet. You can either upload your dataset or use the built-in demo dataset below.")
    if st.button("Use demo dataset"):
        demo_df = build_demo_dataset()
        try:
            X_train, X_test, y_train, y_test, vect = preprocess_df(demo_df)
            model = train_model(X_train, y_train, algorithm="MultinomialNB")
            st.session_state['model'] = model
            st.session_state['vectorizer'] = vect
            st.session_state['training_status'] = "Demo model trained"
            st.success("Demo model trained successfully.")
        except Exception as exc:
            st.error(f"Demo training failed: {exc}")

st.markdown("---")
st.subheader("Check a job posting")
input_text = st.text_area(
    "Paste a job description here",
    value="We are hiring a data analyst to manage reports, SQL, Excel, and Python, with a real company profile and stable salary.",
    height=180,
)

if st.button("Predict"):
    if input_text.strip() == "":
        st.error("Please enter a job posting text before predicting.")
    elif st.session_state.get('model') is None or st.session_state.get('vectorizer') is None:
        st.warning("Train a model first, then predict a job description.")
    else:
        pred = predict_text(st.session_state['model'], st.session_state['vectorizer'], input_text)
        result = "Fake / Scam" if pred == 1 else "Real / Legitimate"
        color = "#ef4444" if pred == 1 else "#22c55e"
        st.markdown(f"<div class='metric-card'><h3 style='color:{color};'>{result}</h3><p class='small-note'>Prediction result from the trained model.</p></div>", unsafe_allow_html=True)
