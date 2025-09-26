from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

# --------------------
# Load model and scaler
# --------------------
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")  # if you used scaling

# Average Indian values (can adjust based on dataset)
AVG_GLUCOSE = 120
AVG_INSULIN = 85
AVG_DPF = 0.5

# --------------------
# Initialize Flask
# --------------------
app = Flask(__name__)
CORS(app)  # ✅ Enable CORS for all routes

# --------------------
# Prediction route
# --------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Get user input with default fallback
    pregnancies = data.get("Pregnancies", 1)
    bmi = data.get("BMI", 25.0)
    age = data.get("Age", 30)

    # Prepare DataFrame for model
    input_df = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [AVG_GLUCOSE],
        "Insulin": [AVG_INSULIN],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [AVG_DPF],
        "Age": [age]
    })

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Make prediction
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    return jsonify({
        "prediction": int(pred),
        "probability": float(prob)
    })

# --------------------
# Run server
# --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
