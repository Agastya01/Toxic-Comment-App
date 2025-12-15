# streamlit_app.py
"""
Single-file Streamlit app with 5 pages:
- Home
- Predict (single comment)
- Batch Predict (CSV upload)
- Dataset Stats (EDA)
- About

Place this file in the same folder as:
- model.h5
- tokenizer.pkl

Run using:
streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Toxic Comment Classifier", layout="centered")
# ---------------- CONFIG / PATHS ----------------
MODEL_PATH = "model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
DATASET_PATH = r"C:\Users\AGASTYA\Downloads\ToxicCommentApp\train.csv"

MAX_LEN = 150
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# ----------------- Load model & tokenizer -----------------
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}.")
    model = load_model(MODEL_PATH)

    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer file not found at {TOKENIZER_PATH}.")
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer

model, tokenizer = load_artifacts()

# ----------------- Text Cleaning -----------------
def clean_text_inline(s):
    s = str(s)
    s = s.lower()

    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"@\w+", " @user ", s)
    s = re.sub(r"#\w+", " #tag ", s)

    s = s.replace("can't", "cannot")
    s = s.replace("won't", "will not")
    s = s.replace("n't", " not")
    s = s.replace("'re", " are")
    s = s.replace("'s", " is")
    s = s.replace("'d", " would")
    s = s.replace("'ll", " will")
    s = s.replace("'ve", " have")

    s = re.sub(r"[^a-z0-9\s\!\?\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    return s

# ----------------- Prediction helper -----------------
def predict_comment(text, threshold=0.5):
    cleaned = clean_text_inline(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    probs = model.predict(padded)[0]
    preds = (probs >= threshold).astype(int)
    return cleaned, probs, preds

# ----------------- Streamlit UI -----------------

st.title("🛡️ Toxic Comment Classification Demo")

page = st.sidebar.radio("Navigate", ["Home", "Predict", "Batch Predict", "Dataset Stats", "About"])

# ---------------- Home ----------------
if page == "Home":
    st.header("Home")
    st.markdown("""
    This app demonstrates a simple **TextCNN-based Toxic Comment Classifier**.

    Pages:
    - **Home**
    - **Predict**
    - **Batch Predict**
    - **Dataset Stats**
    - **About**
    """)

# ---------------- Predict (Single) ----------------
elif page == "Predict":
    st.header("Single Comment Prediction")
    user_text = st.text_area("Enter comment", height=140)
    threshold = st.slider("Binary threshold", 0.1, 0.9, 0.5, step=0.05)

    if st.button("Predict"):
        if not user_text.strip():
            st.warning("Please enter a comment.")
        else:
            cleaned, probs, preds = predict_comment(user_text, threshold)
            st.subheader("Cleaned text")
            st.write(cleaned)

            st.subheader("Probabilities")
            for label, p in zip(LABELS, probs):
                st.write(f"- **{label}**: {p:.4f}")

            st.subheader("Binary Predictions")
            st.json({label: int(v) for label, v in zip(LABELS, preds)})

            if preds.sum() > 0:
                st.error("⚠️ Toxic content detected.")
            else:
                st.success("✅ No toxic content detected.")

# ---------------- Batch Predict ----------------
elif page == "Batch Predict":
    st.header("Batch Prediction (CSV Upload)")
    uploaded = st.file_uploader("Upload CSV (requires column 'comment_text')", type=["csv"])
    threshold = st.slider("Binary threshold", 0.1, 0.9, 0.5, step=0.05)

    if uploaded:
        df = pd.read_csv(uploaded)

        if "comment_text" not in df.columns:
            st.error("CSV must contain a 'comment_text' column.")
        else:
            st.info("Running predictions...")
            comments = df["comment_text"].fillna("").astype(str).tolist()

            cleaned_list = []
            probs_list = []
            preds_list = []

            for txt in comments:
                cleaned, probs, preds = predict_comment(txt, threshold)
                cleaned_list.append(cleaned)
                probs_list.append(probs)
                preds_list.append(preds)

            probs_df = pd.DataFrame(probs_list, columns=[f"{c}_prob" for c in LABELS])
            preds_df = pd.DataFrame(preds_list, columns=[f"{c}_pred" for c in LABELS])

            final_df = pd.concat([df, probs_df, preds_df], axis=1)

            st.subheader("Sample Output")
            st.dataframe(final_df.head(30))

            csv_bytes = final_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Predictions CSV",
                data=csv_bytes,
                file_name="predictions.csv",
                mime="text/csv"
            )

# ---------------- Dataset Stats ----------------
elif page == "Dataset Stats":
    st.header("Dataset Statistics / EDA")
    data_path = st.text_input("Path to dataset", value=DATASET_PATH)

    if st.button("Load dataset"):
        if not os.path.exists(data_path):
            st.error("File not found.")
        else:
            df = pd.read_csv(data_path)

            st.subheader("Preview")
            st.dataframe(df.head())

            st.subheader("Missing Values")
            st.table(df.isnull().sum())

            st.subheader("Sample Comments")
            if "comment_text" in df.columns:
                st.write(df["comment_text"].dropna().sample(10))

            st.subheader("Label Distribution")
            present = [c for c in LABELS if c in df.columns]
            if present:
                counts = df[present].sum()
                st.bar_chart(counts)

# ---------------- About ----------------
elif page == "About":
    st.header("About this App")
    st.markdown("""
    **Model:** TextCNN  
    **Labels:** toxic, severe_toxic, obscene, threat, insult, identity_hate  
    ```bash
    streamlit run streamlit_app.py
    ```
    Place **model.h5** and **tokenizer.pkl** in the same folder.
    """)