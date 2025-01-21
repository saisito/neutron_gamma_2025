import os
import joblib
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, log_loss,
    roc_curve, roc_auc_score, f1_score, precision_recall_curve, auc, precision_score, recall_score, make_scorer
)
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from scipy.optimize import minimize

# Directory for preprocessed data
data_dir = "preprocessed_data_pure"

# Load data from CSV files
X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"), header=None).to_numpy().copy()
y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv"), header=None).squeeze().copy()  # Convert to Series
X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"), header=None).to_numpy().copy()
y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv"), header=None).squeeze().copy()  # Convert to Series

print("Data loaded successfully.")

def train_and_tune_model(model, param_grid, X_train, y_train, model_name, n_iter=50, cv=5):
    """
    Train and tune a model using RandomizedSearchCV.
    
    Args:
        model: The machine learning model to be tuned.
        param_grid: The hyperparameter grid to search.
        X_train: Training features.
        y_train: Training labels.
        model_name: Name of the model (used for saving).
        n_iter: Number of iterations for RandomizedSearchCV.
        cv: Number of cross-validation folds.
        
    Returns:
        The best fitted model.
    """
    print(f"Training and tuning {model_name} using RandomizedSearchCV...")
    try:
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            scoring=make_scorer(f1_score, average="weighted"),
            n_iter=n_iter,
            random_state=42,
            n_jobs=-1,
            cv=cv,
            verbose=2
        )
        search.fit(X_train, y_train)

        print(f"{model_name} best params: {search.best_params_}")
        print(f"{model_name} best score: {search.best_score_}")

        os.makedirs("results_pure/models", exist_ok=True)
        model_path = f"results_pure/models/{model_name}_tuned_pure.pkl"
        joblib.dump(search.best_estimator_, model_path)
        print(f"Model saved to {model_path}")

        best_model = search.best_estimator_
        del search
        gc.collect()

        return best_model

    except Exception as e:
        print(f"An error occurred while tuning {model_name}: {e}")
        return None

def load_and_predict(file, model, scaler, threshold=0.5):
    """
    Load data from a file, scale it, and make predictions using a trained model.
    
    Args:
        file: Path to the data file.
        model: Trained machine learning model.
        scaler: Pre-fitted scaler for data normalization.
        threshold: Probability threshold for binary classification.
        
    Returns:
        neutron_pulses: Data predicted as neutrons.
        gamma_pulses: Data predicted as gammas.
    """
    data = np.loadtxt(file, delimiter=',')
    data_scaled = scaler.transform(data)
    y_proba_all = model.predict_proba(data_scaled)[:, 1]  # Probability of class 1 (neutrons)
    y_pred_all = (y_proba_all >= threshold).astype(int)
    data_descaled = scaler.inverse_transform(data_scaled)
    neutron_pulses = data_descaled[y_pred_all == 1]
    gamma_pulses = data_descaled[y_pred_all == 0]
    
    return neutron_pulses, gamma_pulses

def save_pulses_to_csv(neutron_pulses, gamma_pulses, neutron_file, gamma_file):
    """
    Save neutron and gamma pulses to CSV files.
    
    Args:
        neutron_pulses: Data predicted as neutrons.
        gamma_pulses: Data predicted as gammas.
        neutron_file: Path to save neutron pulses.
        gamma_file: Path to save gamma pulses.
    """
    pd.DataFrame(neutron_pulses).to_csv(neutron_file, index=False, header=False)
    pd.DataFrame(gamma_pulses).to_csv(gamma_file, index=False, header=False)
    print(f'Neutron pulses saved to: {neutron_file}')
    print(f'Gamma pulses saved to: {gamma_file}')

param_grids = {
    'catboost': {
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'depth': [4, 6, 8, 10, 12],
        'iterations': [100, 300, 500, 1000, 1500],
        'l2_leaf_reg': [1, 3, 5, 7, 9],
        'bagging_temperature': [0.1, 0.5, 1.0, 2.0, 3.0],
        'border_count': [32, 64, 128, 256],
        'auto_class_weights': ['Balanced', 'SqrtBalanced', None],
        'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide']
    },
    'rf': {
        'n_estimators': [100, 200, 300, 500, 1000],
        'max_depth': [10, 20, 30, 40, None],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False],
        'class_weight': ['balanced', 'balanced_subsample', None]
    },
    'xgb': {
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'n_estimators': [100, 300, 500, 1000, 1500],
        'max_depth': [3, 6, 9, 12, 15],
        'min_child_weight': [1, 3, 5, 7, 9],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.3, 0.5, 1],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [1.0, 1.5, 2.0, 3.0]
    }
}

# Train and tune models
rf_model = train_and_tune_model(
    RandomForestClassifier(random_state=42),
    param_grids['rf'],
    X_train, 
    y_train, 
    "rf",   
    n_iter=100,  
    cv=5
)

catboost_model = train_and_tune_model(
    CatBoostClassifier(random_state=42, verbose=0),
    param_grids['catboost'],
    X_train, 
    y_train, 
    "catboost",
    n_iter=100,
    cv=5
)

xgb_model = train_and_tune_model(
    XGBClassifier(random_state=42),
    param_grids['xgb'],
    X_train, 
    y_train, 
    "xgboost",
    n_iter=100,
    cv=5
)

# Voting Classifier
voting_clf = VotingClassifier(
    estimators=[
        ('catboost', catboost_model),
        ('rf', RandomForestClassifier(random_state=42)),
        ('xgb', xgb_model)
    ],
    voting='soft', 
    n_jobs=-1
)

print("Training ensemble...")
voting_clf.fit(X_train, y_train)
joblib.dump(voting_clf, 'results_pure/models/voting_pure.pkl')

# Predictions
y_proba = voting_clf.predict_proba(X_test)[:, 1]

# Threshold optimization
thresholds = np.linspace(0, 1, 100)
tp_percentages, tn_percentages = [], []

neutrons = y_test.value_counts()[1]
gammas = y_test.value_counts()[0]

for t in thresholds:
    y_pred = (y_proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    tp_percentages.append(tp / neutrons * 100)
    tn_percentages.append(tn / gammas * 100)

central_range = slice(10, -10)
differences = np.abs(np.array(tp_percentages) - np.array(tn_percentages))
optimal_index = np.argmin(differences[central_range]) + 10
optimal_threshold = thresholds[optimal_index]

plt.figure(figsize=(8, 6))
plt.plot(thresholds, tp_percentages, label="TP", color="red")
plt.plot(thresholds, tn_percentages, label="TN", color="blue")
plt.axvline(x=optimal_threshold, color="green", linestyle="--", label=f"Optimal Threshold = {optimal_threshold:.2f}", linewidth=3)
plt.xlabel("Threshold", fontsize=24)
plt.ylabel("Percentage (%)", fontsize=24)
plt.legend(loc='lower right', fontsize=16)
plt.tick_params(axis='both', labelsize=20)
plt.grid(alpha=0.3)
plt.savefig('results_pure/thresholds_pure.png', dpi=300)

print(f"Optimal Threshold: {optimal_threshold:.2f}")

# Evaluate final metrics
y_pred = (y_proba >= optimal_threshold).astype(int)
cm = confusion_matrix(y_test, y_pred)
TP, FP, FN, TN = cm.ravel()

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * (precision * recall) / (precision + recall)
logloss = -np.mean(y_test * np.log(y_proba) + (1 - y_test) * np.log(1 - y_proba))
roc_auc = roc_auc_score(y_test, y_proba)

metrics_dict = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "Log Loss", "ROC-AUC"],
    "Value": [accuracy, precision, recall, f1, logloss, roc_auc]
}
metrics_df = pd.DataFrame(metrics_dict)
output_csv_path = "results_pure/models/voting_pure_metrics.csv"
metrics_df.to_csv(output_csv_path, index=False)
print(f"Metrics saved to: {output_csv_path}")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_percent, annot=True, fmt=".2f", cmap="Blues", 
            xticklabels=["Neutron", "Gamma"], yticklabels=["Neutron", "Gamma"],
            annot_kws={"size": 16})
plt.title("Confusion Matrix (%)", fontsize=24)
plt.xlabel("Prediction", fontsize=20)
plt.ylabel("Actual", fontsize=20)
plt.tick_params(axis='both', labelsize=16)
plt.savefig('results_pure/confusion_pure.png', dpi=300)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Voting Classifier (AUC = {roc_auc:.2f})', color='blue', linewidth=3)
plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier (AUC = 0.5)", linewidth=2)
plt.title("ROC Curve", fontsize=28)
plt.xlabel("FPR", fontsize=24)
plt.ylabel("TPR", fontsize=24)
plt.grid(alpha=0.4)
plt.legend(loc="lower right", fontsize=16)
plt.tick_params(axis='both', labelsize=20)
plt.savefig('results_pure/roc_pure.png', dpi=300)

scaler = joblib.load('results_pure/models/scaler_pure.pkl')

# Assuming the pipeline has been trained and saved, and you have a pre-fitted scaler
best_model = voting_clf

# Define data file
desnuda_file = 'filtered_AmBe_pure/AmBe_desnuda.csv'

# Load and predict
neutron_pulses_desnuda, gamma_pulses_desnuda = load_and_predict(
    desnuda_file, best_model, scaler, threshold=optimal_threshold
)

# Save the descaled pulses to CSV files
os.makedirs("predicted_pure", exist_ok=True)
save_pulses_to_csv(neutron_pulses_desnuda, gamma_pulses_desnuda, 'predicted_pure/neutron_pulses_pure.csv', 'predicted_pure/gamma_pulses_pure.csv')

# Print results
print(f'Number of neutrons in AmBe desnuda: {neutron_pulses_desnuda.shape[0]}')
print(f'Number of gammas in AmBe desnuda: {gamma_pulses_desnuda.shape[0]}\n')