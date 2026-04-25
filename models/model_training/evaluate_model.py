# Python libraries
import matplotlib.pyplot as plt

# metrics libraries
from sklearn.metrics import roc_auc_score, auc, roc_curve, precision_recall_curve, average_precision_score



def evaluate_model(fitted_pipeline, X_test, y_test):
    """
    Computes the ROC-AUC metrics using the optimized estimator.
    """
    y_pred_proba = fitted_pipeline.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"Out-of-sample Test ROC-AUC: {test_auc:.4f}")

    return y_test, y_pred_proba

def plot_roc_auc(y_test, y_pred_proba):
    """
    Plots the ROC-AUC curve.
    """
    fig = plt.figure(figsize=(7, 5))

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f'XGBoost (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for XGBoost Model')
    plt.legend(loc="lower right")
    plt.show()
    plt.close(fig)


def plot_precision_recall(fitted_pipeline, X_test, y_test):
    """
    Plots the Precision-Recall curve.
    """
    y_proba    = fitted_pipeline.predict_proba(X_test)[:, 1]
    avg_prec   = average_precision_score(y_test, y_proba)
    baseline   = y_test.mean() 

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    
    # due to the synthetic and incomplete nature of the data
    # a boolean mask is needed to assure the of the precision-recall curve
    # can be easily interpreted
    mask = recall >= 0.01

    plt.figure(figsize=(7, 5))
    plt.plot(recall[mask], precision[mask], label=f'XGBoost (AP = {avg_prec:.3f})', color='steelblue')
    plt.axhline(y=baseline, color='red', linestyle='--',
                label=f'Random Classifier (baseline = {baseline:.3f})')

    plt.xlabel('Recall  (fraction of defaulters correctly flagged)')
    plt.ylabel('Precision  (fraction of flagged who actually default)')
    plt.title('Precision-Recall Curve — XGBoost Credit Risk Model')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    print(f"Average Precision (AP): {avg_prec:.4f}")
    return avg_prec

