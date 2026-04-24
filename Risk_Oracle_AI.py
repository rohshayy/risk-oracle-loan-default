import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, precision_recall_curve, auc

# ==========================================
# 1. DATA SIMULATION (Imbalanced Universe)
# ==========================================
np.random.seed(42)
n = 2000

data = {
    'income': np.random.normal(50000, 15000, n),
    'debt_ratio': np.random.uniform(0.1, 0.8, n),
    'credit_score': np.random.normal(650, 100, n),
    'late_payments': np.random.poisson(0.5, n)
}
df = pd.DataFrame(data)

# Logic: Default (1) is rare. Only happens if score is low AND debt is high.
df['default'] = ((df['debt_ratio'] * 2) - (df['credit_score'] / 400) + (df['late_payments'] * 0.5) > 0.5).astype(int)

# Check imbalance (You'll see far more 0s than 1s)
print(f"Target Distribution:\n{df['default'].value_counts(normalize=True)}")

X = df.drop('default', axis=1)
y = df['default']

# Stratified split is vital for imbalanced data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# ==========================================
# 2. SENIOR STEP: HYPERPARAMETER TUNING
# ==========================================
# We don't guess depth; we let the computer find the best one.
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

# GridSearchCV runs the model multiple times to find the "Goldilocks" settings
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='f1')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# ==========================================
# 3. EVALUATION: PRECISION-RECALL
# ==========================================
y_probs = best_model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'PR Curve (AUC = {auc(recall, precision):.2f})')
plt.xlabel('Recall (Ability to find defaulters)')
plt.ylabel('Precision (Ability to be correct)')
plt.title('Senior Level Evaluation: Precision-Recall Curve')
plt.legend()
plt.show()

# ==========================================
# 4. EXPLAINABILITY: FEATURE IMPORTANCE
# ==========================================
importances = pd.Series(best_model.feature_importances_, index=X.columns)
print("\n--- The Mathematical Predictors (Feature Importance) ---")
print(importances.sort_values(ascending=False))