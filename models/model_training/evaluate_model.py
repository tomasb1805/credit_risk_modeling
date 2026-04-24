# Python libraries
import matplotlib.pyplot as plt

# Machine learning + eval libraries
from sklearn.metrics import roc_auc_score, auc, roc_curve



def evaluate_model(fitted_pipeline, X_test, y_test):
    """
    Computes the ROC AUC metrics using the optimized estimator.
    """
    y_pred_proba = fitted_pipeline.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"Out-of-sample Test ROC-AUC: {test_auc:.4f}")

    return y_test, y_pred_proba

def plot_roc_auc(y_test, y_pred_proba):
    """
    Generate ROC AUC plot.
    """
    plt.figure(figsize=(7, 5))

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f'XGBoost (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for XGBoost Model')
    plt.legend(loc="lower right")
    plt.show()
