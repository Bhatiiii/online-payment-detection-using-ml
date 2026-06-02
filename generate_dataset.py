import pandas as pd
import numpy as np
import os

np.random.seed(42)
n = 10_000

# Amount (gamma distribution)
amount = np.random.gamma(shape=2, scale=200, size=n).astype(int)
amount = np.clip(amount, 10, 10000)

# Transaction hour – probabilities now sum to 1.0 exactly
hour_probs = [
    0.02, 0.02, 0.02, 0.03,  # 0-3
    0.04, 0.05, 0.06, 0.07,  # 4-7
    0.07, 0.06, 0.05, 0.05,  # 8-11
    0.05, 0.05, 0.05, 0.05,  # 12-15
    0.05, 0.04, 0.04, 0.03,  # 16-19
    0.03, 0.02, 0.02, 0.03   # 20-23  (last value changed from 0.02 to 0.03)
]
# Verify sum = 1.0
assert abs(sum(hour_probs) - 1.0) < 0.001, f"Sum is {sum(hour_probs)}"

hour = np.random.choice(range(24), size=n, p=hour_probs)

# Merchant category
merchant = np.random.choice(['Electronics','Food','Clothing','Grocery','Travel'], size=n, p=[0.2,0.3,0.15,0.25,0.1])

# Foreign transaction
foreign = np.random.choice([0,1], size=n, p=[0.92,0.08])

# Location mismatch
loc_mismatch = np.random.choice([0,1], size=n, p=[0.95,0.05])

# Device trust score
device_trust = np.random.beta(a=4, b=1.5, size=n) * 100
device_trust = np.clip(device_trust, 0, 100).astype(int)

# Velocity
velocity = np.random.poisson(lam=2, size=n)
velocity = np.clip(velocity, 0, 30)

# Age
age = np.random.normal(loc=38, scale=12, size=n).astype(int)
age = np.clip(age, 18, 90)

# Fraud rule
def is_fraud(row):
    score = 0.0
    if row['hour'] in [1,2,3,4]:
        score += 0.3
    if row['device_trust'] < 40:
        score += 0.4
    if row['loc_mismatch'] == 1:
        score += 0.3
    if row['foreign'] == 1:
        score += 0.2
    if row['velocity'] > 8:
        score += 0.2
    if row['amount'] > 800:
        score += 0.1
    if row['loc_mismatch'] == 1 and row['foreign'] == 1:
        score += 0.2
    if row['hour'] in [1,2,3,4] and row['device_trust'] < 40:
        score += 0.15
    prob = min(0.99, score)
    return np.random.binomial(1, prob)

fraud_label = np.array([is_fraud({'hour': h, 'device_trust': dt, 'loc_mismatch': lm, 
                                   'foreign': f, 'velocity': v, 'amount': a}) 
                        for h, dt, lm, f, v, a in zip(hour, device_trust, loc_mismatch, foreign, velocity, amount)])

# Ensure fraud rate ~1.5%
current_rate = fraud_label.mean()
if current_rate < 0.012:
    idx_high_risk = np.where((device_trust<30) & (loc_mismatch==1))[0]
    needed = int(0.015*n - fraud_label.sum())
    for i in np.random.choice(idx_high_risk, min(needed, len(idx_high_risk)), replace=False):
        fraud_label[i] = 1
elif current_rate > 0.018:
    fraud_idx = np.where(fraud_label==1)[0]
    to_flip = int((current_rate - 0.015) * n)
    for i in np.random.choice(fraud_idx, min(to_flip, len(fraud_idx)), replace=False):
        fraud_label[i] = 0

df = pd.DataFrame({
    'amount': amount,
    'transaction_hour': hour,
    'merchant_category': merchant,
    'foreign_transaction': foreign,
    'location_mismatch': loc_mismatch,
    'device_trust_score': device_trust,
    'velocity_last_24h': velocity,
    'cardholder_age': age,
    'is_fraud': fraud_label
})

os.makedirs('dataset', exist_ok=True)
df.to_csv('dataset/credit_card_fraud_10k.csv', index=False)
print(f"Dataset saved: dataset/credit_card_fraud_10k.csv")
print(f"Fraud rate: {df['is_fraud'].mean():.3%} ({df['is_fraud'].sum()} frauds)")