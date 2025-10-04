"""
Blood Donation Prediction — Logistic Regression Model
Author: Dan Li
Dataset: UCI Blood Transfusion Service Center
Description:
    This script analyzes donor behavior using logistic regression.
    It generates visual outputs (PNG) and saves evaluation metrics.
"""


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc

# --- Set up output directory ---
output_dir = "/Users/danli/Documents/healthcare_analysis/blood-donation-analysis/outputs"
os.makedirs(output_dir, exist_ok=True)  # create folder if it doesn't exist

# --- Load Data ---
df = pd.read_csv("/Users/danli/Documents/healthcare_analysis/blood-donation-analysis/SQL_blood.csv")

print(df.head())
print(df.info())
print(df['donate'].value_counts(normalize=True))
print(df.describe())
print(df.groupby('donate')[['recency','frequency','monetary','months']].mean())
print(df.corr())

# --- Model ---
X = df[['recency', 'frequency', 'monetary', 'months']]
y = df['donate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))

# --- 1. Donation class distribution ---
plt.figure(figsize=(6,4))
sns.countplot(x='donate', data=df, palette='viridis')
plt.title("Donation Class Distribution")
plt.xlabel("Donate (0=No, 1=Yes)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "donation_distribution.png"), dpi=300)
plt.close()

# --- 2. Correlation Heatmap ---
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=300)
plt.close()

# --- 3. Recency by Donation Status ---
plt.figure(figsize=(6,4))
sns.boxplot(x='donate', y='recency', data=df, palette='Set2')
plt.title("Recency by Donation Status")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "recency_vs_donation.png"), dpi=300)
plt.close()

# --- 4. ROC Curve ---
y_pred_prob = model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0,1],[0,1],'--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Donation Prediction")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=300)
plt.close()

print(f"✅ All charts saved successfully in: {output_dir}")
