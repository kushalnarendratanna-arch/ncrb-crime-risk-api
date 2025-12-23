from flask import Flask, jsonify, request, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "NCRB_Crime_Against_Childrens.csv")

model = joblib.load(os.path.join(BASE_DIR, "risk_model.pkl"))
encoder = joblib.load(os.path.join(BASE_DIR, "risk_encoder.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
max_total = joblib.load(os.path.join(BASE_DIR, "max_total.pkl"))

df = None

def load_data():
    global df
    if df is not None:
        return

    df = pd.read_csv(CSV_FILE)

    df.rename(columns={
        "State/UT": "State",
        "State/UT/District": "District",
        "Total IPC Crimes Against Children - Col. ( 36)": "ipc",
        "Total SLL Crimes against Children - Col. ( 70)": "sll"
    }, inplace=True)

    df["ipc"] = pd.to_numeric(df["ipc"], errors="coerce").fillna(0)
    df["sll"] = pd.to_numeric(df["sll"], errors="coerce").fillna(0)

    df["total"] = df["ipc"] + df["sll"]
    df["ipc_ratio"] = df["ipc"] / (df["total"] + 1)
    df["sll_ratio"] = df["sll"] / (df["total"] + 1)
    df["crime_density"] = df["total"] / (df.groupby("State")["total"].transform("mean") + 1)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/states")
def states():
    load_data()
    return jsonify(sorted(df["State"].dropna().unique().tolist()))

@app.route("/districts")
def districts():
    load_data()
    state = request.args.get("state", "")
    districts = df[df["State"] == state]["District"].dropna().unique().tolist()
    return jsonify(sorted(districts))

@app.route("/predict", methods=["POST"])
def predict():
    load_data()
    data = request.get_json()

    state = data.get("state")
    district = data.get("district")

    row = df[(df["State"] == state) & (df["District"] == district)]
    if row.empty:
        return jsonify({"error": "No data available"}), 404

    ipc = int(row["ipc"].values[0])
    sll = int(row["sll"].values[0])
    total = int(row["total"].values[0])

    X = row[["ipc", "sll", "ipc_ratio", "sll_ratio", "crime_density"]]
    X = scaler.transform(X)

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]

    risk_level = encoder.inverse_transform([pred])[0]
    confidence = round(max(prob) * 100, 2)
    risk_score = round((total / max_total) * 100, 2)

    color = "green" if risk_level == "Low" else "orange" if risk_level == "Medium" else "red"

    return jsonify({
        "state": state,
        "district": district,
        "ipc": ipc,
        "sll": sll,
        "total": total,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "confidence": confidence,
        "color": color
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
