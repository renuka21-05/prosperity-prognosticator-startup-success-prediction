# ============================================
# STARTUP SUCCESS PREDICTION - INTERNSHIP VERSION
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------
df = pd.read_csv("startup_funding.csv")   # Replace with your CSV path

# Keep only relevant columns
df = df[['Startup Name', 'Industry Vertical', 'City  Location', 'Amount in USD']]

# --------------------------------------------
# 2. CLEAN AMOUNTInUSD COLUMN
# --------------------------------------------
df = df[df['Amount in USD'].notna()]  # Remove NaN
df = df[~df['Amount in USD'].str.lower().isin(['unknown', 'undisclosed'])]  # Remove invalid strings

# Remove commas, $, +, non-numeric chars
def clean_amount(x):
    if isinstance(x, str):
        x = re.sub(r'[^\d.]', '', x)
        if x == '':
            return None
        return float(x)
    return x

df['Amount in USD'] = df['Amount in USD'].apply(clean_amount)
df = df[df['Amount in USD'].notna()]

# --------------------------------------------
# 3. CREATE TARGET COLUMN
# --------------------------------------------
# Success = 1 if funding >= 500k USD, else 0
df['Success'] = df['Amount in USD'].apply(lambda x: 1 if x >= 500000 else 0)

# --------------------------------------------
# 4. DROP AmountInUSD FROM FEATURES
# --------------------------------------------
# We remove AmountInUSD from features to avoid data leakage
df_features = df.drop(columns=['Amount in USD', 'Startup Name'])

# Fill missing categorical values
df_features['Industry Vertical'].fillna('Other', inplace=True)
df_features['City  Location'].fillna('Other', inplace=True)

# --------------------------------------------
# 5. ENCODE CATEGORICAL COLUMNS
# --------------------------------------------
le_industry = LabelEncoder()
le_city = LabelEncoder()

df_features['Industry Vertical'] = le_industry.fit_transform(df_features['Industry Vertical'])
df_features['City  Location'] = le_city.fit_transform(df_features['City  Location'])

# Features and target
X = df_features[['Industry Vertical', 'City  Location']]
y = df_features['Success']

# --------------------------------------------
# 6. SPLIT DATA
# --------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------------------------
# 7. TRAIN MODEL
# --------------------------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# --------------------------------------------
# 8. PREDICT
# --------------------------------------------
y_pred = model.predict(X_test)

# --------------------------------------------
# 9. EVALUATE
# --------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# --------------------------------------------
# 10. FEATURE IMPORTANCE
# --------------------------------------------
importances = model.feature_importances_
plt.figure(figsize=(6,4))
plt.barh(X.columns, importances)
plt.title("Feature Importance")
plt.show()

# --------------------------------------------
# 11. SAVE MODEL
# --------------------------------------------
joblib.dump(model, "startup_model.pkl")
print("\nModel saved successfully as startup_model.pkl")

# --------------------------------------------
# 12. PREDICT NEW STARTUP (INTERACTIVE EXAMPLE)
# --------------------------------------------
# Example: a new startup
new_startup = pd.DataFrame({
    'Industry Vertical': ['Tech'],
    'City  Location': ['Bangalore']
})

# Encode categorical
new_startup['Industry Vertical'] = le_industry.transform(new_startup['Industry Vertical'])
new_startup['City  Location'] = le_city.transform(new_startup['City  Location'])

# Predict success probability
success_prob = model.predict_proba(new_startup)[:,1]
print("\nPredicted Probability of Success for new startup:", success_prob[0])

import joblib

joblib.dump(le_industry, "le_industry.pkl")
joblib.dump(le_city, "le_city.pkl")
