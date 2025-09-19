# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
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
# 2. Optimization Function
# -----------------------------
def evaluate_composition(sn, ag, cu, model, requirements, prices):
    """Evaluate a composition for feasibility and cost"""
    input_df = pd.DataFrame([[sn, ag, cu]], columns=['Sn', 'Ag', 'Cu'])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = model.predict(input_df)[0]
    strength, temp, cond = pred
    if (strength >= requirements['Strength'] and
        temp >= requirements['Melting_Temp'] and
        cond >= requirements['Conductivity']):
        cost = sn*prices['Sn'] + ag*prices['Ag'] + cu*prices['Cu']
        return cost, pred, (sn, ag, cu)
    else:
        return np.inf, pred, (sn, ag, cu)

def optimize_alloy(model, requirements, prices, n_iter=5000):
    best_cost = np.inf
    best_pred = None
    best_comp = None

    for _ in range(n_iter):
        ag = np.random.uniform(0, 4.0)
        cu = np.random.uniform(0, 0.7)
        sn = 100 - ag - cu
        if sn <= 0:
            continue
        cost, pred, comp = evaluate_composition(sn, ag, cu, model, requirements, prices)
        if cost < best_cost:
            best_cost = cost
            best_pred = pred
            best_comp = comp

    return best_comp, best_cost, best_pred

# -----------------------------
# 3. Streamlit UI
# -----------------------------
st.title("🔬 SAC Alloy Optimizer (Sn–Ag–Cu)")
st.write("Find the cheapest Sn–Ag–Cu alloy composition that meets your property requirements.")

# User inputs
st.sidebar.header("⚙️ Input Parameters")

req_strength = st.sidebar.number_input("Required Yield Strength (MPa)", 10.0, 200.0, 85.0)
req_temp = st.sidebar.number_input("Required Melting Temperature (°C)", 100.0, 300.0, 150.0)
req_cond = st.sidebar.number_input("Required Electrical Conductivity (MS/m)", 1.0, 200.0, 160.0)

st.sidebar.header("💰 Market Prices (per % unit in TAKA)")
price_Ag = st.sidebar.number_input("Silver (Ag)", 1.0, 1000.0, 10.0)
price_Cu = st.sidebar.number_input("Copper (Cu)", 1.0, 1000.0, 120.0)
price_Sn = st.sidebar.number_input("Tin (Sn)", 1.0, 1000.0, 10.0)

requirements = {'Strength': req_strength, 'Melting_Temp': req_temp, 'Conductivity': req_cond}
prices = {'Ag': price_Ag, 'Cu': price_Cu, 'Sn': price_Sn}

if st.button("🔎 Optimize Alloy"):
    best_comp, best_cost, best_pred = optimize_alloy(model, requirements, prices, n_iter=10000)

    if best_comp:
        sn, ag, cu = best_comp
        st.success("✅ Optimal SAC Composition Found")
        st.write(f"**Sn:** {sn:.2f} %")
        st.write(f"**Ag:** {ag:.2f} %")
        st.write(f"**Cu:** {cu:.2f} %")

        st.subheader("📐 Predicted Properties")
        st.write(f"- Yield Strength: **{best_pred[0]:.2f} MPa**")
        st.write(f"- Melting Temp: **{best_pred[1]:.2f} °C**")
        st.write(f"- Conductivity: **{best_pred[2]:.2f} MS/m**")

        st.subheader("💰 Estimated Cost")
        st.write(f"**{best_cost:.2f} TAKA**")
    else:
        st.error("❌ No feasible composition found with the given requirements.")
