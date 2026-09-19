import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc

logger = logging.getLogger(__name__)

def evaluate_fraud_metrics(model, X_test, y_test):
    """
    Evaluates a trained model pipeline on the test set using fraud-specific metrics.
    """
    try:
        logger.info("Starting model evaluation on test data...")
        
        # 1. Generate predictions
        y_pred = model.predict(X_test)
        
        # Extract probabilities for AUC calculations
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            # Fallback for models without predict_proba (like some raw deep learning steps)
            y_proba = model.decision_function(X_test)
            
        logger.info("Predictions and probabilities generated successfully.")

        # 2. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        logger.info("\n=== CONFUSION MATRIX ===")
        logger.info(f"True Negatives (Legit caught): {tn}")
        logger.info(f"False Positives (False Alarms): {fp}")
        logger.info(f"False Negatives (Missed Fraud!): {fn}")
        logger.info(f"True Positives (Fraud caught!): {tp}")

        # 3. Classification Report (Precision, Recall, F1-Score)
        report = classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud'])
        print("\n=== CLASSIFICATION REPORT ===")
        print(report)

        # 4. ROC-AUC and PR-AUC
        # PR-AUC is much more informative than ROC-AUC for highly imbalanced fraud datasets
        roc_auc = roc_auc_score(y_test, y_proba)
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall_vals, precision_vals)

        logger.info(f"ROC-AUC Score: {roc_auc:.4f}")
        logger.info(f"PR-AUC (Precision-Recall AUC) Score: {pr_auc:.4f}")

        metrics_dict = {
            "confusion_matrix": cm,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "missed_fraud": fn,
            "caught_fraud": tp
        }
        
        return metrics_dict

    except Exception as e:
        logger.exception("An error occurred during metrics evaluation.")
        return None
