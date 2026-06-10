import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import shap
from sklearn.metrics import confusion_matrix


# CONFIG


st.set_page_config(
    page_title="Telco Churn Dashboard",
    page_icon="📞",
    layout="wide"
)

ARTIFACTS = Path("../models")

st.title("📞 Telco Customer Churn Dashboard")

# CHARGEMENT


@st.cache_resource
def load_models():
    return {
        "mlp": joblib.load(ARTIFACTS / "mlp.pkl"),
        "rf": joblib.load(ARTIFACTS / "rf.pkl"),
        "encoder": joblib.load(ARTIFACTS / "encoder.pkl"),
        "scaler": joblib.load(ARTIFACTS / "scaler.pkl"),
        "baseline": joblib.load(ARTIFACTS / "baseline.pkl"),
        "elasticnet": joblib.load(ARTIFACTS / "elasticnet.pkl")
    }

models = load_models()

df = pd.read_csv("../data/raw/dataset.csv")


df = df.drop(columns=["customerID"])


# SIDEBAR


st.sidebar.header("⚙️ Paramètres")

model_choice = st.sidebar.radio(
    "Modèle",
    ["mlp", "rf", "baseline", "elasticnet"],
    format_func=lambda x: "MLP" if x == "mlp" else "Random Forest" if x == "rf" else "Baseline" if x == "baseline" else "Elastic Net" if x == "elasticnet" else x
)

seuil = st.sidebar.slider(
    "Seuil churn",
    0.1, 0.9, 0.5
)

compare = st.sidebar.checkbox("Comparer les modèles")

# TABS


tab1, tab2, tab3 = st.tabs([
    "🔮 Prédiction",
    "📊 Analyse",
    "📈 Performance"
])


# TAB 1 - PREDICTION


with tab1:
    st.subheader("🧠 Simulation client")

    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Clients", len(df))

    col2.metric(
        "Taux de churn",
        f"{(df['Churn']=='Yes').mean():.1%}"
    )

    col3.metric(
        "Charges moyennes",
        f"{df['MonthlyCharges'].mean():.2f}$"
    )

    col4.metric(
        "Ancienneté moyenne",
        f"{df['tenure'].mean():.1f} mois"
    )

    with col1:
        gender = st.selectbox("Genre", df["gender"].unique())
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", df["Partner"].unique())
        dependents = st.selectbox("Dependents", df["Dependents"].unique())

    with col2:
        tenure = st.slider("Tenure (mois)", 0, 72, 12)
        phone = st.selectbox("PhoneService", df["PhoneService"].unique())
        internet = st.selectbox("InternetService", df["InternetService"].unique())
        contract = st.selectbox("Contract", df["Contract"].unique())

    with col3:
        monthly = st.number_input("MonthlyCharges", 0.0, 200.0, 70.0)
        total = st.number_input("TotalCharges", 0.0, 10000.0, 1000.0)
        payment = st.selectbox("PaymentMethod", df["PaymentMethod"].unique())

    with col4:
        # SHAP values
        shap_values = st.checkbox("Afficher les valeurs SHAP")

    # construire input
    user_input = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": "No",
        "InternetService": internet,
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": contract,
        "PaperlessBilling": "Yes",
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    # encoding
    X_enc = models["encoder"].transform(user_input)

    # scaling (si MLP)
    X_scaled = models["scaler"].transform(X_enc)

    def predict_model(name):
        model = models[name]

        X = X_scaled if name == "mlp" else X_enc

        return float(model.predict_proba(X)[0,1])

 
    # MODE COMPARAISON


    if compare:
        st.subheader("📊 Comparaison des modèles")

        probs = {
            "MLP": predict_model("mlp"),
            "Random Forest": predict_model("rf"),
            "Baseline": predict_model("baseline"),
            "Elastic Net": predict_model("elasticnet")
           
        }

        fig = px.bar(
            x=list(probs.keys()),
            y=list(probs.values()),
            text=[f"{p:.1%}" for p in probs.values()],
            range_y=[0, 1]
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
       
        proba = predict_model(model_choice)
        st.subheader("🎯 Résultat")

        if proba >= seuil:
            st.error(f"🔴 RISQUE DE CHURN : {proba:.1%}")
        else:
            st.success(f"🟢 RISQUE DE CHURN FAIBLE : {proba:.1%}")

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                gauge={
                    "axis": {"range": [0, 100]},
                    "steps": [
                        {"range": [0, 30], "color": "#d4edda"},
                        {"range": [30, 70], "color": "#fff3cd"},
                        {"range": [70, 100], "color": "#f8d7da"}
                    ]
                }
            )
        )

        st.plotly_chart(fig, use_container_width=True)


# TAB 2 - ANALYSE


with tab2:
    st.subheader("📊 Analyse exploratoire")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(df, names="Churn", title="Répartition churn")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(df, x="Contract", color="Churn", barmode="group")
        st.plotly_chart(fig, use_container_width=True)

    fig = px.box(df, x="Churn", y="tenure", title="Ancienneté vs churn")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(df, x="MonthlyCharges", color="Churn")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(df, x="TotalCharges", color="Churn")
    st.plotly_chart(fig, use_container_width=True)
    

# TAB 3 - PERFORMANCE


with tab3:
    st.subheader("📈 Performance des modèles")

    metrics = pd.DataFrame({
        "Modèle": ["MLP", "Random Forest", "Baseline", "Elastic Net"],
        "Accuracy": [0.84, 0.82, 0.75, 0.79],
        "F1-score": [0.78, 0.76, 0.68, 0.72],
        "AUC": [0.88, 0.85, 0.73, 0.81]
    })

    st.dataframe(metrics)

    fig = px.bar(
        metrics.melt(id_vars="Modèle"),
        x="Modèle",
        y="value",
        color="variable",
        barmode="group"
    )


    st.plotly_chart(fig, use_container_width=True)
