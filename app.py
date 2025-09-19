# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings

# -----------------------------
# 1. Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/robiul911/Sn/refs/heads/main/Sn_Ag_Cu_Large_Dataset_with_Conductivity.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

# Features & Targets
X = df[['Sn', 'Ag', 'Cu']]
y = df[['Yield_Strength_MPa', 'Melting_Temp_C', 'Electrical_Conductivity_MS_per_m']]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42))
model.fit(X_train, y_train)

# -----------------------------
# 2. Vectorized Optimization Function
# -----------------------------
def optimize_alloy_vectorized(model, requirements, prices, n_iter=50000):
    ag = np.random.uniform(-5, 5, n_iter)
    cu = np.random.uniform(-1, 1, n_iter)
    sn = 100 - ag - cu
    valid_idx = sn > 0
    sn, ag, cu = sn[valid_idx], ag[valid_idx], cu[valid_idx]

    if len(sn) == 0:
        return None, None, None

    input_df = pd.DataFrame({'Sn': sn, 'Ag': ag, 'Cu': cu})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preds = model.predict(input_df)

    strength = preds[:, 0]
    temp = preds[:, 1]
    cond = preds[:, 2]

    mask = (strength >= requirements['Strength']) & \
           (temp >= requirements['Melting_Temp']) & \
           (cond >= requirements['Conductivity'])

    if mask.any():
        costs = sn[mask]*prices['Sn'] + ag[mask]*prices['Ag'] + cu[mask]*prices['Cu']
        best_idx = np.argmin(costs)
        best_comp = (sn[mask][best_idx], ag[mask][best_idx], cu[mask][best_idx])
        best_pred = preds[mask][best_idx]
        best_cost = costs[best_idx]
        return best_comp, best_cost, best_pred
    else:
        return None, None, None

# -----------------------------
# 3. Streamlit UI
# -----------------------------
st.set_page_config(page_title="SAC Alloy Optimizer", layout="wide")
st.title("🔬 SAC Alloy Optimizer (Sn–Ag–Cu)")
st.write("Find the cheapest Sn–Ag–Cu alloy composition that meets your property requirements.")

# Inputs in two columns
col1, col2 = st.columns(2)

with col1:
    st.header("⚙️ Input Requirements")
    req_strength = st.number_input("Yield Strength (MPa)", value=85.0, step=1.0, format="%.2f")
    req_temp = st.number_input("Melting Temperature (°C)", value=150.0, step=1.0, format="%.2f")
    req_cond = st.number_input("Electrical Conductivity (MS/m)", value=160.0, step=1.0, format="%.2f")

with col2:
    st.header("💰 Market Prices per % Composition")
    price_Ag = st.number_input("Silver (Ag)", value=10.0, step=1.0, format="%.2f")
    price_Cu = st.number_input("Copper (Cu)", value=120.0, step=1.0, format="%.2f")
    price_Sn = st.number_input("Tin (Sn)", value=10.0, step=1.0, format="%.2f")

requirements = {'Strength': req_strength, 'Melting_Temp': req_temp, 'Conductivity': req_cond}
prices = {'Ag': price_Ag, 'Cu': price_Cu, 'Sn': price_Sn}

# -----------------------------
# 4. Optimize Alloy
# -----------------------------
if st.button("🔎 Optimize Alloy"):
    with st.spinner("⏳ Optimization running, please wait..."):
        best_comp, best_cost, best_pred = optimize_alloy_vectorized(model, requirements, prices, n_iter=50000)

    if best_comp:
        sn, ag, cu = best_comp
        st.success("✅ Optimal SAC Composition Found")
        st.metric("Sn (%)", f"{sn:.2f}")
        st.metric("Ag (%)", f"{ag:.2f}")
        st.metric("Cu (%)", f"{cu:.2f}")

        st.subheader("📐 Predicted Properties")
        st.write(f"- Yield Strength: **{best_pred[0]:.2f} MPa**")
        st.write(f"- Melting Temperature: **{best_pred[1]:.2f} °C**")
        st.write(f"- Electrical Conductivity: **{best_pred[2]:.2f} MS/m**")

        st.subheader("💰 Estimated Cost")
        st.write(f"**{best_cost:.2f} TAKA**")
    else:
        st.error("❌ No feasible composition found with the given requirements.")

# -----------------------------
# 5. Optional Evaluation Script
# -----------------------------
if st.checkbox("Show Model Evaluation Script"):
    st.subheader("📊 Model Evaluation Metrics")
    y_pred = model.predict(X_test)

    for i, col in enumerate(y.columns):
        mse = mean_squared_error(y_test[col], y_pred[:, i])
        r2 = r2_score(y_test[col], y_pred[:, i])
        st.write(f"**{col}**: MSE = {mse:.2f}, R² = {r2:.2f}")

    st.subheader("🔹 Predicted vs Actual Plots")
    for i, col in enumerate(y.columns):
        fig, ax = plt.subplots()
        ax.scatter(y_test[col], y_pred[:, i], alpha=0.5)
        ax.plot([y_test[col].min(), y_test[col].max()],
                [y_test[col].min(), y_test[col].max()], 'r--')
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{col}: Predicted vs Actual")
        st.pyplot(fig)

    st.subheader("🔹 Feature Importance per Target")
    for i, target in enumerate(y.columns):
        importances = model.estimators_[i].feature_importances_
        fig, ax = plt.subplots()
        ax.bar(X.columns, importances)
        ax.set_title(f"Feature Importance for {target}")
        st.pyplot(fig)
