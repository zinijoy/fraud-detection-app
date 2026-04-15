
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


# LOGIN SYSTEM

users = {
    "admin": {"password": "admin123", "role": "admin"},
    "user": {"password": "user123", "role": "user"}
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.username = None

def login():
    st.title("🔐 Login to Fraud Detection System")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users and users[username]["password"] == password:
            st.session_state.logged_in = True
            st.session_state.role = users[username]["role"]
            st.session_state.username = username
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid username or password")

def logout():
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.username = None
    st.rerun()

# AUTH CHECK

if not st.session_state.logged_in:
    login()

else:
    st.set_page_config(page_title="Fraud Detection System", layout="wide")
    st.title("💳 Fraud Detection Dashboard")

    st.sidebar.write(f"👤 Logged in as: {st.session_state.username}")
    if st.sidebar.button("Logout"):
        logout()

    threshold = st.sidebar.slider("Fraud Threshold", 0.1, 0.9, 0.3)

    if "history" not in st.session_state:
        st.session_state.history = []


    # ADMIN CONTROL

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


    # MAIN SYSTEM

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        # SAVE ORIGINAL
        df_original = df.copy()

        # CLEANING
        df = df.drop(columns=[
            'Unnamed: 0', 'trans_date_trans_time', 'cc_num',
            'first', 'last', 'street', 'trans_num'
        ], errors='ignore')

        if 'merch_zipcode' in df.columns:
            imputer = SimpleImputer(strategy='most_frequent')
            df['merch_zipcode'] = imputer.fit_transform(df[['merch_zipcode']])

        # ENCODING + STORE ENCODERS
        encoders = {}
        for col in df.select_dtypes(include='object').columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

        if 'is_fraud' not in df.columns:
            st.error("Dataset must contain 'is_fraud'")
        else:
            X = df.drop('is_fraud', axis=1)
            y = df['is_fraud']


            # SMALL CHARTS

            col1, col2 = st.columns(2)
            with col1:
                fig1, ax1 = plt.subplots(figsize=(3, 2))
                y.value_counts().plot(kind='bar', ax=ax1)
                ax1.set_title("Before SMOTE")
                st.pyplot(fig1)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

            with col2:
                fig2, ax2 = plt.subplots(figsize=(3, 2))
                pd.Series(y_resampled).value_counts().plot(kind='bar', ax=ax2)
                ax2.set_title("After SMOTE")
                st.pyplot(fig2)

            # MODEL

            # =========================
            # MODEL IMPLEMENTATION & COMPARISON
            # =========================

            st.subheader("⚙️ Model Implementation & Comparison")

            # -------------------------
            # MODEL 1: RANDOM FOREST (BASELINE)
            # -------------------------
            rf_model = RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=42
            )

            rf_model.fit(X_resampled, y_resampled)
            rf_pred = rf_model.predict(X_test)

            rf_acc = accuracy_score(y_test, rf_pred)
            rf_report = classification_report(y_test, rf_pred, output_dict=True)

            # -------------------------
            # MODEL 2: XGBOOST (IMPROVED)
            # -------------------------
            xgb_model = XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=10,
                eval_metric='logloss',
                use_label_encoder=False
            )

            xgb_model.fit(X_resampled, y_resampled)
            xgb_pred = xgb_model.predict(X_test)

            xgb_acc = accuracy_score(y_test, xgb_pred)
            xgb_report = classification_report(y_test, xgb_pred, output_dict=True)

            # -------------------------
            # DISPLAY COMPARISON
            # -------------------------
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

            st.write("📊 Model Comparison")
            st.dataframe(comparison_df)

            # -------------------------
            # FINAL MODEL SELECTION
            # -------------------------
            st.success("✅ Final Model Selected: XGBoost (Higher Fraud Recall)")

            model = xgb_model  # used for prediction

            # -------------------------
            # CONFUSION MATRIX (XGBOOST)
            # -------------------------
            st.subheader("📉 Confusion Matrix (Final Model)")

            cm = confusion_matrix(y_test, xgb_pred)

            fig_cm, ax = plt.subplots(figsize=(3, 2))
            sns.heatmap(cm, annot=True, fmt="d", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig_cm)

            # -------------------------
            # PERFORMANCE REPORT
            # -------------------------
            st.subheader("📈 Model Performance")

            st.write("Accuracy:", round(xgb_acc, 4))
            st.text(classification_report(y_test, xgb_pred))


            # DROPDOWN INPUT UI

            st.subheader("Predict Transaction")

            sample_input = {}

            categorical_cols = ['merchant', 'category', 'gender', 'city', 'state', 'Payment_Method']

            for col in categorical_cols:
                if col in df_original.columns:
                    options = df_original[col].unique()
                    selected = st.selectbox(col, options)
                    sample_input[col] = encoders[col].transform([selected])[0]

            numeric_cols = ['amt', 'city_pop', 'Customer_Age']

            for col in numeric_cols:
                if col in df.columns:
                    sample_input[col] = st.number_input(col, value=0.0)

            if st.button("Predict Fraud"):
                input_df = pd.DataFrame([sample_input])

                for col in X.columns:
                    if col not in input_df.columns:
                        input_df[col] = 0

                input_df = input_df[X.columns]

                prob = model.predict_proba(input_df)[0][1]
                fraud_percent = prob * 100
                prediction = 1 if prob > threshold else 0

                # GAUGE
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
                fig.update_layout(width=350, height=250)
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

            # HISTORY
            st.subheader("Transaction History")

            if len(st.session_state.history) > 0:
                st.dataframe(pd.DataFrame(st.session_state.history))
            else:
                st.info("No transactions yet.")