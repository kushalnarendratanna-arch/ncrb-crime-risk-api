from flask import Flask, jsonify, request, render_template
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates"
)

CSV_FILE = os.path.join(BASE_DIR, "NCRB_Crime_Against_Childrens.csv")

df = None
model = encoder = scaler = imputer = max_total = None


def load_models():
    global model, encoder, scaler, imputer, max_total
    if model is not None:
        return

    model = joblib.load(os.path.join(BASE_DIR, "risk_model.pkl"))
    encoder = joblib.load(os.path.join(BASE_DIR, "risk_encoder.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    imputer = joblib.load(os.path.join(BASE_DIR, "imputer.pkl"))
    max_total = joblib.load(os.path.join(BASE_DIR, "max_total.pkl"))

    if not max_total or max_total == 0:
        max_total = 1


def load_csv():
    global df
    if df is not None:
        return

    temp = pd.read_csv(CSV_FILE)

    # 🔥 FLEXIBLE COLUMN HANDLING (FIX)
    col_map = {
        "State/UT": "State",
        "State/UT/District": "District",
        "Total IPC Crimes Against Children - Col. ( 36 )": "ipc",
        "Total IPC Crimes Against Children - Col. ( 36)": "ipc",
        "Total SLL Crimes against Children - Col. ( 70 )": "sll",
        "Total SLL Crimes against Children - Col. ( 70)": "sll",
    }

    temp.rename(columns={k: v for k, v in col_map.items() if k in temp.columns}, inplace=True)

    required = ["State", "District", "ipc", "sll"]
    for col in required:
        if col not in temp.columns:
            raise Exception(f"Missing column in CSV: {col}")

    temp["ipc"] = pd.to_numeric(temp["ipc"], errors="coerce").fillna(0)
    temp["sll"] = pd.to_numeric(temp["sll"], errors="coerce").fillna(0)

    temp["total"] = temp["ipc"] + temp["sll"]
    temp["ipc_ratio"] = temp["ipc"] / (temp["total"] + 1)
    temp["sll_ratio"] = temp["sll"] / (temp["total"] + 1)
    temp["crime_density"] = temp["total"] / (
        temp.groupby("State")["total"].transform("mean") + 1
    )

    df = temp


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
    return jsonify(sorted(df[df["State"] == state]["District"].dropna().unique().tolist()))


@app.route("/predict", methods=["POST"])
def predict():
    try:
        load_models()
        load_csv()

        data = request.get_json()
        state = data.get("state")
        district = data.get("district")

        if not state or not district:
            return jsonify({"error": "Invalid input"}), 400

        row = df[(df["State"] == state) & (df["District"] == district)]
        if row.empty:
            return jsonify({"error": "No data available"}), 404

        X = row[["ipc", "sll", "ipc_ratio", "sll_ratio", "crime_density"]]
        X = imputer.transform(X)
        X = scaler.transform(X)

        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0]

        risk_level = encoder.inverse_transform([pred])[0]
        confidence = round(float(max(prob)) * 100, 2)
        risk_score = round((row["total"].values[0] / max_total) * 100, 2)

        color = "green" if risk_level == "Low" else "orange" if risk_level == "Medium" else "red"

        return jsonify({
            "state": state,
            "district": district,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "confidence": confidence,
            "color": color
        })

    except Exception as e:
        return jsonify({"error": "Server error", "details": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)