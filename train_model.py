import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

def main():
    # 1. Load Data
    data_path = 'credit_data.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset '{data_path}' not found. Please run generate_data.py first.")
        
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Define features and target
    X = df.drop(columns=['Creditworthy'])
    y = df['Creditworthy']
    
    # Identify numerical and categorical columns
    num_cols = [
        'Age', 'Annual Income', 'Loan Amount', 'Existing Debts', 
        'Credit Utilization Ratio', 'Number of Credit Accounts', 
        'Payment History', 'Credit History Length', 'Number of Late Payments', 
        'Debt-to-Income Ratio'
    ]
    cat_cols = ['Employment Status', 'Loan Purpose']
    
    print("\n--- Features Summary ---")
    print(f"Numerical features ({len(num_cols)}): {num_cols}")
    print(f"Categorical features ({len(cat_cols)}): {cat_cols}")
    
    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # 3. Preprocessing Pipelines
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ]
    )
    
    # Fit preprocessor on training data
    print("\nPreprocessing data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after preprocessing
    ohe_categories = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols)
    feature_names = num_cols + list(ohe_categories)
    
    # Save preprocessor and feature names
    os.makedirs('models', exist_ok=True)
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    joblib.dump(feature_names, 'models/feature_names.joblib')
    
    # 4. Model Training & Comparison
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    print("\n--- Training Baseline Models ---")
    for name, model in models.items():
        model.fit(X_train_processed, y_train)
        y_pred = model.predict(X_test_processed)
        y_prob = model.predict_proba(X_test_processed)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        results[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc
        }
        print(f"{name:20} - Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        
    # Save baseline comparison
    pd.DataFrame(results).T.to_csv('models/model_comparison.csv')
    
    # 5. Hyperparameter Tuning on Random Forest
    print("\n--- Tuning Random Forest Hyperparameters ---")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [6, 10, 14, None],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 'log2', None]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train_processed, y_train)
    
    best_rf = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
    
    # Evaluate best model
    y_pred_best = best_rf.predict(X_test_processed)
    y_prob_best = best_rf.predict_proba(X_test_processed)[:, 1]
    
    best_acc = accuracy_score(y_test, y_pred_best)
    best_prec = precision_score(y_test, y_pred_best)
    best_rec = recall_score(y_test, y_pred_best)
    best_f1 = f1_score(y_test, y_pred_best)
    best_auc = roc_auc_score(y_test, y_prob_best)
    
    print("\n--- Best Model (Tuned Random Forest) Performance ---")
    print(classification_report(y_test, y_pred_best))
    print(f"Accuracy  : {best_acc:.4f}")
    print(f"Precision : {best_prec:.4f}")
    print(f"Recall    : {best_rec:.4f}")
    print(f"F1-Score  : {best_f1:.4f}")
    print(f"ROC-AUC   : {best_auc:.4f}")
    
    # Save best model
    joblib.dump(best_rf, 'models/best_model.joblib')
    
    # Save a small sample of test data (original & preprocessed) for the dashboard and SHAP
    # We will save the first 100 test samples of X_test and X_test_processed
    joblib.dump(X_test.head(100), 'models/X_test_sample.joblib')
    joblib.dump(X_test_processed[:100], 'models/X_test_processed_sample.joblib')
    joblib.dump(y_test.head(100), 'models/y_test_sample.joblib')
    
    # 6. Generate Metrics & Plots
    # A. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Creditworthy', 'Creditworthy'], yticklabels=['Not Creditworthy', 'Creditworthy'])
    plt.title('Confusion Matrix - Best Random Forest')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png', dpi=300)
    plt.close()
    
    # B. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob_best)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {best_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('models/roc_curve.png', dpi=300)
    plt.close()
    
    # C. Feature Importance
    importances = best_rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Save feature importances to CSV
    feat_imp_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in indices],
        'Importance': importances[indices]
    })
    feat_imp_df.to_csv('models/feature_importances.csv', index=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_imp_df.head(15), palette='viridis')
    plt.title('Top 15 Most Important Features - Tuned Random Forest')
    plt.xlabel('Relative Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('models/feature_importance.png', dpi=300)
    plt.close()
    
    print("\nAll assets, metrics, and figures saved to the 'models/' directory.")

if __name__ == '__main__':
    main()
