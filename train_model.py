import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load data
df = pd.read_csv('dataset/credit_card_fraud_10k.csv')

# Encode merchant category
le = LabelEncoder()
df['merchant_encoded'] = le.fit_transform(df['merchant_category'])

# Feature columns
feature_cols = ['amount', 'transaction_hour', 'merchant_encoded', 'foreign_transaction',
                'location_mismatch', 'device_trust_score', 'velocity_last_24h', 'cardholder_age']
X = df[feature_cols]
y = df['is_fraud']

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred).tolist()  # [[TN, FP], [FN, TP]]

# Feature importance (normalized)
importances = rf.feature_importances_
feat_imp = [{'name': col, 'importance': float(imp)} for col, imp in zip(feature_cols, importances)]
feat_imp.sort(key=lambda x: x['importance'], reverse=True)

# Prepare stats
stats = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'confusion_matrix': {
        'tn': cm[0][0], 'fp': cm[0][1],
        'fn': cm[1][0], 'tp': cm[1][1]
    },
    'feature_importance': feat_imp
}

# Save artifacts
os.makedirs('model', exist_ok=True)
joblib.dump(rf, 'model/rf_model.pkl')
joblib.dump(le, 'model/label_encoder.pkl')
with open('model/model_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print("Training completed.")
print(f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
print(f"Confusion matrix (test set):")
print(f"  TN={cm[0][0]}, FP={cm[0][1]}")
print(f"  FN={cm[1][0]}, TP={cm[1][1]}")