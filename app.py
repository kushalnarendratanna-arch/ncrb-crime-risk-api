from flask import Flask, jsonify, request, render_template
import pandas as pd
import joblib
import os

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# -----------------------------
# App config
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates"
)

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["100 per minute"]  # relaxed for demo
)

# -----------------------------
# File paths
# -----------------------------
CSV_FILE = os.path.join(BASE_DIR, "NCRB_Crime_Against_Childrens.csv")

# -----------------------------
# Globals
# -----------------------------
LAST_LOADED = 0
df = None

model = None
encoder = None
scaler = None
imputer = None
max_total = None

# -----------------------------
# Load ML models lazily
# -----------------------------
def load_models():
    global model, encoder, scaler, imputer, max_total
    if model is not None:
        return

    model = joblib.load(os.path.join(BASE_DIR, "risk_model.pkl"))
    encoder = joblib.load(os.path.join(BASE_DIR, "risk_encoder.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    imputer = joblib.load(os.path.join(BASE_DIR, "imputer.pkl"))
    max_total = joblib.load(os.path.join(BASE_DIR, "max_total.pkl"))

# -----------------------------
# Load CSV with auto reload
# -----------------------------
def load_csv():
    global df, LAST_LOADED

    mtime = os.path.getmtime(CSV_FILE)
    if df is not None and mtime == LAST_LOADED:
        return

    temp = pd.read_csv(CSV_FILE)

    temp.rename(columns={
        "State/UT": "State",
        "State/UT/District": "District",
        "Total IPC Crimes Against Children - Col. ( 36)": "ipc",
        "Total SLL Crimes against Children - Col. ( 70)": "sll"
    }, inplace=True)

    temp["ipc"] = pd.to_numeric(temp["ipc"], errors="coerce").fillna(0)
    temp["sll"] = pd.to_numeric(temp["sll"], errors="coerce").fillna(0)

    temp["total"] = temp["ipc"] + temp["sll"]
    temp["ipc_ratio"] = temp["ipc"] / (temp["total"] + 1)
    temp["sll_ratio"] = temp["sll"] / (temp["total"] + 1)
    temp["crime_density"] = (
        temp["total"] /
        (temp.groupby("State")["total"].transform("mean") + 1)
    )

    df = temp
    LAST_LOADED = mtime

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/states")
def states():
    load_csv()
    return jsonify(sorted(df["State"].dropna().unique().tolist()))

@app.route("/districts")
def districts():
    load_csv()
    state = request.args.get("state")
    if not state:
        return jsonify([])
    d = df[df["State"] == state]["District"].dropna().unique().tolist()
    return jsonify(sorted(d))

@app.route("/predict", methods=["POST"])
@limiter.limit("30 per minute")
def predict():
    load_models()
    load_csv()

    data = request.json or {}
    state = data.get("state")
    district = data.get("district")

    if not state or not district:
        return jsonify({"error": "Please select both state and district"}), 400

    row = df[(df["State"] == state) & (df["District"] == district)]
    if row.empty:
        return jsonify({"error": "No data available"}), 404

    X = row[["ipc", "sll", "ipc_ratio", "sll_ratio", "crime_density"]]
    X = imputer.transform(X)
    X = scaler.transform(X)

    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0]

    risk_level = encoder.inverse_transform([pred])[0]
    confidence = round(float(max(probs)) * 100, 2)
    risk_score = round((row["total"].values[0] / max_total) * 100, 2)

    color = "green" if risk_level == "Low" else "orange" if risk_level == "Medium" else "red"

    return jsonify({
        "state": state,
        "district": district,
        "ipc": int(row["ipc"].values[0]),
        "sll": int(row["sll"].values[0]),
        "total": int(row["total"].values[0]),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "confidence": confidence,
        "color": color,
        "model": "RandomForestClassifier",
        "data_updated": "NCRB Portal (2024)"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)