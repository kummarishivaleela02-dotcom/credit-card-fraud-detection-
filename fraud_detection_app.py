"""
Credit Card Fraud Detection App
================================
A beginner-friendly Streamlit app that uses a Random Forest classifier
to detect fraudulent credit card transactions.

Structure:
    1. Data Loading & Generation
    2. Preprocessing
    3. Model Training
    4. Prediction Function
    5. Streamlit UI
"""

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

@st.cache_data
def load_data():
    """
    Generate a synthetic credit card transaction dataset.
    In a real project, you'd load a CSV here:
        df = pd.read_csv("creditcard.csv")

    Features:
        - Time   : seconds elapsed between this and the first transaction
        - Amount : transaction amount in USD
        - Class  : 0 = Not Fraud, 1 = Fraud
    """
    np.random.seed(42)
    n_samples = 5000

    # Simulate legitimate transactions (majority class)
    n_legit = int(n_samples * 0.97)
    legit_time   = np.random.uniform(0, 172800, n_legit)      # up to 48 hrs
    legit_amount = np.abs(np.random.normal(80, 120, n_legit)) # avg ~$80
    legit_labels = np.zeros(n_legit)

    # Simulate fraudulent transactions (minority class)
    n_fraud = n_samples - n_legit
    fraud_time   = np.random.uniform(0, 172800, n_fraud)
    fraud_amount = np.abs(np.random.normal(400, 300, n_fraud)) # avg ~$400
    fraud_labels = np.ones(n_fraud)

    # Combine and shuffle
    time_col   = np.concatenate([legit_time,   fraud_time])
    amount_col = np.concatenate([legit_amount, fraud_amount])
    labels     = np.concatenate([legit_labels, fraud_labels])

    df = pd.DataFrame({"Time": time_col, "Amount": amount_col, "Class": labels})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    return df


# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────

@st.cache_resource
def preprocess_and_train(model_choice: str):
    """
    Load data → handle missing values → scale features → train model.

    Returns:
        scaler  : fitted StandardScaler
        model   : trained classifier
        accuracy: test-set accuracy (float)
        report  : classification report (str)
    """
    df = load_data()

    # --- Handle missing values (none in synthetic data, but good practice) ---
    df.dropna(inplace=True)

    X = df[["Time", "Amount"]]
    y = df["Class"]

    # --- Feature Scaling ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Train / Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Model Selection ---
    if model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = LogisticRegression(max_iter=1000, random_state=42)

    model.fit(X_train, y_train)

    # --- Evaluation ---
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred, target_names=["Not Fraud", "Fraud"])

    return scaler, model, accuracy, report


# ─────────────────────────────────────────────
# 3. PREDICTION FUNCTION
# ─────────────────────────────────────────────

def predict_transaction(scaler, model, time_val: float, amount_val: float):
    """
    Predict whether a transaction is fraudulent.

    Args:
        scaler     : fitted StandardScaler
        model      : trained classifier
        time_val   : transaction time (seconds)
        amount_val : transaction amount (USD)

    Returns:
        prediction   : 0 (Not Fraud) or 1 (Fraud)
        probability  : confidence score for the predicted class [0, 1]
    """
    input_data = np.array([[time_val, amount_val]])
    input_scaled = scaler.transform(input_data)

    prediction  = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][int(prediction)]

    return int(prediction), float(probability)


# ─────────────────────────────────────────────
# 4. STREAMLIT UI
# ─────────────────────────────────────────────

def main():
    # ── Page config ──────────────────────────
    st.set_page_config(
        page_title="Credit Card Fraud Detection",
        page_icon="🛡️",
        layout="centered",
    )

    # ── Custom CSS for a cleaner look ────────
    st.markdown("""
        <style>
            .result-fraud {
                background-color: #ffe4e4;
                border-left: 5px solid #e53e3e;
                padding: 1rem 1.25rem;
                border-radius: 6px;
                font-size: 1.15rem;
                font-weight: 600;
                color: #c53030;
            }
            .result-safe {
                background-color: #e6ffed;
                border-left: 5px solid #38a169;
                padding: 1rem 1.25rem;
                border-radius: 6px;
                font-size: 1.15rem;
                font-weight: 600;
                color: #276749;
            }
            .confidence-label {
                font-size: 0.9rem;
                color: #555;
                margin-top: 0.4rem;
            }
        </style>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────
    with st.sidebar:
        st.title("ℹ️ About This App")
        st.markdown("""
        **Credit Card Fraud Detection** uses machine learning to classify
        credit card transactions as **fraudulent** or **safe**.

        ---
        **How it works:**
        1. A synthetic dataset of 5,000 transactions is generated.
        2. Features (`Time`, `Amount`) are scaled with `StandardScaler`.
        3. A classifier is trained on 80% of the data.
        4. You enter a transaction → the model predicts in real time.

        ---
        **Models available:**
        - 🌲 Random Forest *(default)*
        - 📈 Logistic Regression

        ---
        **Note:** This app uses *synthetic* data for demo purposes.
        For a real use-case, load the
        [Kaggle Credit Card Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
        """)

        st.markdown("---")
        model_choice = st.selectbox(
            "🤖 Select Model",
            ["Random Forest", "Logistic Regression"],
        )

        show_report = st.checkbox("Show full classification report", value=False)

    # ── Main title ────────────────────────────
    st.title("🛡️ Credit Card Fraud Detection App")
    st.markdown("Enter transaction details below to check if it's **fraudulent or safe**.")
    st.markdown("---")

    # ── Load & train ──────────────────────────
    with st.spinner("🔄 Loading data and training model…"):
        scaler, model, accuracy, report = preprocess_and_train(model_choice)

    # ── Model accuracy badge ──────────────────
    st.success(f"✅ **{model_choice}** trained successfully  |  Test Accuracy: **{accuracy * 100:.2f}%**")

    if show_report:
        with st.expander("📊 Full Classification Report"):
            st.code(report, language="text")

    st.markdown("---")

    # ── Input fields ──────────────────────────
    st.subheader("🔢 Enter Transaction Details")

    col1, col2 = st.columns(2)

    with col1:
        time_val = st.number_input(
            "⏱️ Transaction Time (seconds)",
            min_value=0.0,
            max_value=172800.0,
            value=50000.0,
            step=100.0,
            help="Seconds elapsed since the first transaction in the dataset (0 – 172,800).",
        )

    with col2:
        amount_val = st.slider(
            "💵 Transaction Amount (USD)",
            min_value=0.0,
            max_value=5000.0,
            value=150.0,
            step=1.0,
            help="Dollar value of the transaction.",
        )

    # Show the chosen amount as a formatted number
    st.caption(f"Selected amount: **${amount_val:,.2f}**")

    st.markdown("")

    # ── Predict button ────────────────────────
    if st.button("🔍 Predict Transaction", use_container_width=True, type="primary"):
        prediction, probability = predict_transaction(scaler, model, time_val, amount_val)

        st.markdown("---")
        st.subheader("🎯 Prediction Result")

        if prediction == 1:
            st.markdown(
                f'<div class="result-fraud">⚠️ Fraudulent Transaction Detected!</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="result-safe">✅ Safe Transaction — No Fraud Detected</div>',
                unsafe_allow_html=True,
            )

        # Confidence score
        st.markdown(
            f'<p class="confidence-label">Model confidence: <strong>{probability * 100:.1f}%</strong></p>',
            unsafe_allow_html=True,
        )

        # Visual confidence bar
        st.progress(probability, text=f"Confidence: {probability * 100:.1f}%")

        # Summary table
        st.markdown("#### 📋 Transaction Summary")
        summary_df = pd.DataFrame({
            "Feature": ["Time (s)", "Amount (USD)", "Prediction", "Confidence"],
            "Value":   [
                f"{time_val:,.0f}",
                f"${amount_val:,.2f}",
                "🚨 Fraud" if prediction == 1 else "✅ Safe",
                f"{probability * 100:.1f}%",
            ],
        })
        st.table(summary_df)

    # ── Footer ────────────────────────────────
    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit · scikit-learn · NumPy · Pandas")


if __name__ == "__main__":
    main()
