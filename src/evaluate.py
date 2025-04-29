import pickle
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

from config import Config
from model import LitModel


def evaluate_transformer():
    """
    Loads the best Transformer checkpoint and evaluates it on the test dataset.
    Prints accuracy, precision, recall, F1, confusion matrix, ROC, and PR curves.
    """
    # Load test dataset
    with open(Config.PROCESSED_DATA_DIR / "test_dataset.pkl", "rb") as f:
        test_dataset = pickle.load(f)

    # Create DataLoader
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, num_workers=4
    )

    # Load trained model from best checkpoint
    ckpt_path = Path(Config.CHECKPOINT_DIR) / "best-checkpoint-v17.ckpt"
    print(f"Loading Transformer checkpoint from: {ckpt_path}")
    model = LitModel.load_from_checkpoint(str(ckpt_path))
    model.eval()

    # Accumulate predictions
    y_true, y_pred_probs = [], []

    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x)
            y_true.extend(y.cpu().numpy())
            y_pred_probs.extend(torch.sigmoid(logits).cpu().numpy())

    # Convert to binary predictions
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = y_pred_probs > 0.5

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)

    print(f"\nModel Metrics:")
    print(f" Accuracy:  {acc:.4f}")
    print(f" Precision: {prec:.4f}")
    print(f" Recall:    {rec:.4f}")
    print(f" F1 Score:  {f1:.4f}")

    def print_metrics(y_true, y_pred, y_probs, task_name="Model"):

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{task_name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", lw=2)
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {task_name}")
        plt.legend()
        plt.show()

        # Precision-Recall Curve
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_probs)
        pr_auc = auc(recall_vals, precision_vals)
        plt.figure()
        plt.plot(recall_vals, precision_vals, label=f"PR AUC = {pr_auc:.2f}", lw=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve - {task_name}")
        plt.legend()
        plt.show()

    print_metrics(y_true, y_pred, y_pred_probs)


def evaluate_baselines():
    """
    Loads each trained baseline model (.joblib) from the 'models' folder
    and evaluates on the test dataset, printing accuracy, precision, recall, and F1 results.
    """
    # Load test dataset
    with open(Config.PROCESSED_DATA_DIR / "test_dataset.pkl", "rb") as f:
        test_dataset = pickle.load(f)

    # Convert test dataset to numpy
    test_features = np.array([sample[0] for sample in test_dataset])
    test_targets = np.array([sample[1] for sample in test_dataset])

    # List of known baseline model filenames
    possible_models = [
        "svm.joblib",
        "decision_tree.joblib",
        "random_forest.joblib",
        "logistic_regression.joblib",
        "knn.joblib",
        "naive_bayes.joblib",
        "mlp.joblib",
        "xgboost.joblib",
        "lightgbm.joblib",
    ]
    evaluations = []
    for filename in possible_models:
        model_path = Config.MODEL_DIR / filename
        if not model_path.exists():
            print(f"Model file not found: {model_path}")
            continue

        model = joblib.load(model_path)
        model_name = filename.replace(".joblib", "").title().replace("_", " ")

        # Evaluate on 'diagnosis' (column 0)
        diag_preds = model.predict(test_features)
        diag_acc = accuracy_score(test_targets[:, 0], diag_preds)
        diag_prec = precision_score(test_targets[:, 0], diag_preds)
        diag_rec = recall_score(test_targets[:, 0], diag_preds)
        diag_f1 = f1_score(test_targets[:, 0], diag_preds)

        # Evaluate on 'classification' (column 1)
        # For consistency, we do not want to re-fit the model on test data,
        # but if you want each baseline to be separately trained for classification:
        # just call model.fit(train_features, train_targets[:, 1]) in train.py
        # or load a second saved model. For demonstration, we'll just do .predict here.
        class_preds = model.predict(test_features)
        class_acc = accuracy_score(test_targets[:, 1], class_preds)
        class_prec = precision_score(test_targets[:, 1], class_preds)
        class_rec = recall_score(test_targets[:, 1], class_preds)
        class_f1 = f1_score(test_targets[:, 1], class_preds)

        evaluations.append(
            {
                "model_name": model_name,
                "diagnosis_accuracy": diag_acc,
                "diagnosis_precision": diag_prec,
                "diagnosis_recall": diag_rec,
                "diagnosis_f1": diag_f1,
                "class_accuracy": class_acc,
                "class_precision": class_prec,
                "class_recall": class_rec,
                "class_f1": class_f1,
            }
        )
    return evaluations
