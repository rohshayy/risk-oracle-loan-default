# Risk Oracle: Predictive Loan Default Engine

## **Project Overview**
Risk Oracle is a machine learning solution designed to identify high-risk loan applicants within highly imbalanced datasets. In financial environments, defaults are "rare events"; this project addresses that challenge through **stratified sampling**, **ensemble learning**, and **rigorous threshold optimization**.

## **The Engineering Engine: Random Forest**
Instead of a single decision tree, this project utilizes a **Random Forest Classifier** to reduce variance and prevent overfitting.
* **Bagging (Bootstrap Aggregating):** The model trains 100+ individual trees on random subsets of the data.
* **Collective Intelligence:** By averaging the results of these trees, the engine creates a stable "Committee of Experts" that is significantly more robust to noise than a single estimator.

## **Mathematical Optimization**

### **1. Hyperparameter Tuning (Grid Search)**
To find the optimal model configuration, I implemented `GridSearchCV` to perform an exhaustive search for the **Global Maximum** of the F1-score.
* **Cross-Validation:** The engine evaluates various tree depths and estimators across multiple "folds" to ensure the parameters generalize to unseen data.
* **Optimization Goal:** Balancing model complexity with predictive power to avoid the bias-variance trade-off.

### **2. Precision-Recall Analysis**
Since accuracy is a misleading metric for rare-event detection, the model is evaluated using the **Precision-Recall Curve**.
* **Area Under Curve (AUC):** Achieved an AUC of 0.81, demonstrating high discriminatory power.
* **Recall Focus:** Prioritizing the model's ability to "catch" defaulters while maintaining a professional level of precision to avoid excessive false alarms.

### **3. Explainability: Gini Importance**
To maintain transparency (crucial for financial regulations), the model extracts **Feature Importance**. This ranks variables based on how much they reduce the total **Entropy** (disorder) in the dataset, identifying key predictors such as `debt_ratio` and `credit_score`.

## **Technical Stack**
* **Language:** Python
* **Libraries:** Scikit-Learn (Ensemble, Model Selection, Metrics), NumPy, Pandas
* **Visualization:** Matplotlib
