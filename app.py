from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import json
from datetime import datetime

app = Flask(__name__)

# Load model and artifacts
model = joblib.load('model/rf_model.pkl')
le = joblib.load('model/label_encoder.pkl')
with open('model/model_stats.json', 'r') as f:
    stats = json.load(f)

feature_cols = ['amount', 'transaction_hour', 'merchant_encoded', 'foreign_transaction',
                'location_mismatch', 'device_trust_score', 'velocity_last_24h', 'cardholder_age']

def risk_flags(amount, hour, device_trust, loc_mismatch, foreign, velocity):
    flags = []
    if 1 <= hour <= 4:
        flags.append("Late-night transaction")
    if device_trust < 40:
        flags.append("Low device trust score")
    if loc_mismatch == 1:
        flags.append("Location mismatch")
    if foreign == 1:
        flags.append("Foreign transaction")
    if velocity > 8:
        flags.append("High velocity (many transactions)")
    if amount > 800:
        flags.append("High amount")
    return flags

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stats', methods=['GET'])
def get_stats():
    return jsonify(stats)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        # required fields
        amount = float(data['amount'])
        hour = int(data['transaction_hour'])
        merchant = data['merchant_category']
        foreign = int(data['foreign_transaction'])
        loc_mismatch = int(data['location_mismatch'])
        device_trust = int(data['device_trust_score'])
        velocity = int(data['velocity_last_24h'])
        age = int(data['cardholder_age'])
        tx_id = data.get('transaction_id', f"TXN-{datetime.now().strftime('%Y%m%d')}-{np.random.randint(1000,9999)}")
    except (KeyError, TypeError, ValueError):
        return jsonify({'error': 'Invalid or missing fields'}), 400

    # Validate ranges
    if not (0 <= hour <= 23):
        return jsonify({'error': 'transaction_hour must be 0-23'}), 400
    if not (0 <= device_trust <= 100):
        return jsonify({'error': 'device_trust_score must be 0-100'}), 400
    if not (0 <= velocity <= 100):
        return jsonify({'error': 'velocity_last_24h must be 0-100'}), 400
    if not (18 <= age <= 90):
        return jsonify({'error': 'cardholder_age must be 18-90'}), 400
    if amount < 0:
        return jsonify({'error': 'amount must be positive'}), 400

    # Encode merchant
    try:
        merchant_enc = le.transform([merchant])[0]
    except ValueError:
        return jsonify({'error': f'Unknown merchant category: {merchant}'}), 400

    # Build feature array
    features = pd.DataFrame([[amount, hour, merchant_enc, foreign, loc_mismatch,
                              device_trust, velocity, age]], columns=feature_cols)
    prob_fraud = model.predict_proba(features)[0][1]
    flags = risk_flags(amount, hour, device_trust, loc_mismatch, foreign, velocity)
    is_fraud = (prob_fraud >= 0.15) or (len(flags) >= 4)
    prediction = "Fraud" if is_fraud else "Legitimate"

    response = {
        'transaction_id': tx_id,
        'prediction': prediction,
        'is_fraud': bool(is_fraud),
        'fraud_prob': round(prob_fraud, 4),
        'safe_prob': round(1 - prob_fraud, 4),
        'risk_flags': flags
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)