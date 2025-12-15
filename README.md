# Toxic-Comment-App
# 🚫 Toxic Comment Classification App

A web application that classifies text comments as **toxic or non-toxic** using a deep learning / NLP model, built with **Streamlit** and Python.

This project focuses on building and deploying a **real-time text classification app**. The dataset used for training is **not included** in the repository due to its size — you can download it separately (see below). The trained model and tokenizer files are included or loaded as part of the app.

---

## 📌 Project Overview

**Toxic comments** are those containing rude, disrespectful, or abusive language that might discourage healthy interactions. This app predicts whether a comment is toxic using a machine learning model.

The classification model was trained on a **Toxic Comment Classification dataset**, such as the one from the Jigsaw Kaggle challenge. :contentReference[oaicite:1]{index=1}

---

## 🚀 Features

✔️ Classifies input text into toxic or non-toxic  
✔️ Simple, user-friendly Streamlit interface  
✔️ Fast, real-time inference  
✔️ Ideal for moderation tools or AI demos

---

## 🧠 Model

The app uses a pre-trained classification model (e.g., a neural network) along with a tokenizer for text preprocessing. These files should be placed in the project directory. The model predicts toxicity based on learned patterns from labeled comment data.

---

## 📥 Dataset (Not Included)

⚠️ The dataset is **NOT included** in this repo due to large file size.  

You can download the dataset from Kaggle (example: Jigsaw Toxic Comment Classification Challenge) or any similar toxic comment dataset:

👉 https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview :contentReference[oaicite:2]{index=2}

After downloading:
1. Extract the data locally.
2. Use it to train your model.
3. Save the trained model (`.h5`, `.pkl`, or similar) and add it to this repository (if small enough).

---

## 📦 Installation

Make sure you have Python 3.7 or later.

1. **Clone the repository**

   ```bash
   git clone https://github.com/Agastya01/Toxic-Comment-App.git
   cd Toxic-Comment-App
