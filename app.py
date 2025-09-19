# @title
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import warnings

# -----------------------------
# 1. Load Dataset
# -----------------------------
import os

# Prefer the local CSV in the workspace if present; otherwise fall back to the remote URL.
local_csv = os.path.join(os.path.dirname(__file__), 'synthetic_sac_dataset_v2.csv')
remote_url = "https://raw.githubusercontent.com/robiul911/Sn/refs/heads/main/Sn_Ag_Cu_Large_Dataset_with_Conductivity.csv"
if os.path.exists(local_csv):
    df = pd.read_csv(local_csv)
else:
    # try remote (may fail if offline)
    df = pd.read_csv(remote_url)

# Features: composition
X = df[['Sn', 'Ag', 'Cu']]

# Targets: properties to satisfy
y = df[['Yield_Strength_MPa', 'Melting_Temp_C', 'Electrical_Conductivity_MS_per_m']]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 2. Train Multi-output Model
# -----------------------------
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42))
model.fit(X_train, y_train)

print("✅ Model trained successfully.")

# -----------------------------
# 3. Optimization Function
# -----------------------------
def evaluate_composition(sn, ag, cu, model, requirements, prices):
    """Evaluate a composition for feasibility and cost"""
    # Use DataFrame with correct column names
    input_df = pd.DataFrame([[sn, ag, cu]], columns=['Sn', 'Ag', 'Cu'])
    # Suppress warnings for feature mismatch
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = model.predict(input_df)[0]
    strength, temp, cond = pred
    # Check if composition meets requirements
    if (strength >= requirements['Strength'] and
        temp >= requirements['Melting_Temp'] and
        cond >= requirements['Conductivity']):
        # Compute cost
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
# 4. Interactive User Input
# -----------------------------
try:
    print("Enter required properties:")
    req_strength = float(input("Required Yield Strength (MPa): "))
    req_temp = float(input("Required Melting Temperature (°C): "))
    req_cond = float(input("Required Electrical Conductivity (MS/m): "))

    print("\nEnter current market prices per unit composition in TAKA:")
    price_Ag = float(input("Silver (Ag): "))
    price_Cu = float(input("Copper (Cu): "))
    price_Sn = float(input("Tin (Sn): "))

    requirements = {'Strength': req_strength, 'Melting_Temp': req_temp, 'Conductivity': req_cond}
    prices = {'Ag': price_Ag, 'Cu': price_Cu, 'Sn': price_Sn}

    # Run optimization
    best_comp, best_cost, best_pred = optimize_alloy(model, requirements, prices)

    if best_comp:
        sn, ag, cu = best_comp
        print("\n✅ Optimal SAC Composition Found:")
        print(f"Sn: {sn:.2f}%, Ag: {ag:.2f}%, Cu: {cu:.2f}%")
        print(f"Predicted Properties: Yield Strength={best_pred[0]:.2f} MPa, "
              f"Melting Temp={best_pred[1]:.2f} °C, Conductivity={best_pred[2]:.2f} MS/m")
        print(f"Estimated Cheapest Cost: ${best_cost:.2f}")
    else:
        print("\n❌ No feasible composition found with given requirements.")

except Exception as e:
    print("Error:", e)