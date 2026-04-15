import streamlit as st
st.set_page_config(page_title="Fraud Detection System", layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


# =========================
# LOGIN SYSTEM
# =========================

users = {
    "admin": {"password": "admin123", "role": "admin"},
    "user": {"password": "user123", "role": "user"}
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.username = None
    st.session_state.history = []


def login():
    st.title("🔐 Login to Fraud Detection System")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users and users[username]["password"] == password:
            st.session_state.logged_in = True
            st.session_state.role = users[username]["role"]
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Invalid username or password")


def logout():
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.username = None
    st.rerun()


# =========================
# AUTH CHECK
# =========================

if not st.session_state.logged_in:
    login()
    st.stop()


# =========================
# MAIN APP
# =========================

st.title("💳 Fraud Detection Dashboard")

st.sidebar.write(f"👤 Logged in as: {st.session_state.username}")

if st.sidebar.button("Logout"):
    logout()

threshold = st.sidebar.slider("Fraud Threshold", 0.1, 0.9, 0.3)


# =========================
# ADMIN CONTROL
# =========================

if st.session_state.role == "admin":
    st.sidebar.success("Admin Access")

    if st.sidebar.button("Clear History"):
        st.session_state.history = []
        st.success("History cleared")

    uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])

else:
    st.sidebar.info("User Access")
    st.warning("Only admin can upload dataset")
    uploaded_file = None


# =========================
# LOAD DATA SAFELY
# =========================

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    df_original = df.copy()

    # =========================
    # CLEANING
    # =========================

    df = df.drop(columns=[
        'Unnamed: 0', 'trans_date_trans_time', 'cc_num',
        'first', 'last', 'street', 'trans_num'
    ], errors='ignore')

    if 'merch_zipcode' in df.columns:
        imputer = SimpleImputer(strategy='most_frequent')
        df['merch_zipcode'] = imputer.fit_transform(df[['merch_zipcode']])

    # =========================
    # ENCODING
    # =========================

    encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # =========================
    # CHECK TARGET COLUMN
    # =========================

    if 'is_fraud' not in df.columns:
        st.error("Dataset must contain 'is_fraud' column")
        st.stop()

    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']


    # =========================
    # TRAIN TEST SPLIT
    # =========================

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =========================
    # SMOTE
    # =========================

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)


    # =========================
    # MODELS
    # =========================

    rf_model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42
    )

    rf_model.fit(X_resampled, y_resampled)
    rf_pred = rf_model.predict(X_test)

    rf_acc = accuracy_score(y_test, rf_pred)
    rf_report = classification_report(y_test, rf_pred, output_dict=True)


    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=10,
        eval_metric='logloss',
        use_label_encoder=False,
        tree_method="hist"
    )

    xgb_model.fit(X_resampled, y_resampled)
    xgb_pred = xgb_model.predict(X_test)

    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_report = classification_report(y_test, xgb_pred, output_dict=True)


    # =========================
    # COMPARISON
    # =========================

    st.subheader("⚙️ Model Comparison")

    comparison_df = pd.DataFrame({
        "Model": ["Random Forest", "XGBoost"],
        "Accuracy": [rf_acc, xgb_acc],
        "Recall (Fraud)": [
            rf_report["1"]["recall"],
            xgb_report["1"]["recall"]
        ],
        "Precision (Fraud)": [
            rf_report["1"]["precision"],
            xgb_report["1"]["precision"]
        ]
    })

    st.dataframe(comparison_df)

    st.success("Final Model Selected: XGBoost")

    model = xgb_model


    # =========================
    # CONFUSION MATRIX
    # =========================

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, xgb_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    st.pyplot(fig)


    # =========================
    # PREDICTION SYSTEM
    # =========================

    st.subheader("Predict Transaction")

    sample_input = {}

    categorical_cols = [
        col for col in ['merchant', 'category', 'gender', 'city', 'state', 'Payment_Method']
        if col in df_original.columns
    ]

    for col in categorical_cols:
        options = df_original[col].dropna().unique()
        selected = st.selectbox(col, options)

        try:
            sample_input[col] = encoders[col].transform([str(selected)])[0]
        except:
            sample_input[col] = 0


    numeric_cols = ['amt', 'city_pop', 'Customer_Age']

    for col in numeric_cols:
        if col in df.columns:
            sample_input[col] = st.number_input(col, value=0.0)


    if st.button("Predict Fraud"):

        input_df = pd.DataFrame([sample_input])

        for col in X.columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df.reindex(columns=X.columns, fill_value=0)

        prob = model.predict_proba(input_df)[0][1]
        fraud_percent = prob * 100
        prediction = 1 if prob > threshold else 0

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fraud_percent,
            title={'text': "Fraud Risk (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"},
                ],
            }
        ))

        st.plotly_chart(fig)

        st.write(f"Fraud Probability: {fraud_percent:.2f}%")

        if prediction == 1:
            st.error("⚠️ FRAUD DETECTED")
        else:
            st.success("✅ LEGITIMATE")

        st.session_state.history.append({
            "Probability": round(fraud_percent, 2),
            "Result": "Fraud" if prediction == 1 else "Legit"
        })


# =========================
# HISTORY
# =========================

st.subheader("Transaction History")

if len(st.session_state.history) > 0:
    st.dataframe(pd.DataFrame(st.session_state.history))
else:
    st.info("No transactions yet.")
