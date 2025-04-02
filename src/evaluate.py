import pickle
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config
from model import LitModel
import joblib
from pathlib import Path


def evaluate_transformer():
    """
    Loads the best Transformer checkpoint and evaluates it on the test dataset.
    Prints accuracy, precision, recall, F1, confusion matrices, ROC, and PR curves.
    """
    # Load test dataset
    with open(Config.PROCESSED_DATA_DIR / "test_dataset.pkl", "rb") as f:
        test_dataset = pickle.load(f)

    # Create DataLoader
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, num_workers=4
    )

    # Load trained model (update checkpoint name if necessary)
    model_ckpt = Path(Config.CHECKPOINT_DIR) / "best-checkpoint-v13.ckpt"
    print(f"Loading Transformer checkpoint from: {model_ckpt}")
    model = LitModel.load_from_checkpoint(str(model_ckpt))
    model.eval()

    # Accumulate predictions
    y_true_diag, y_true_class = [], []
    y_pred_diag, y_pred_class = [], []

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            diag_logits, class_logits = model(x)

            y_true_diag.extend(y[:, 0].cpu().numpy())
            y_true_class.extend(y[:, 1].cpu().numpy())

            y_pred_diag.extend(torch.sigmoid(diag_logits).cpu().numpy())
            y_pred_class.extend(torch.sigmoid(class_logits).cpu().numpy())

    # Convert probabilities to binary predictions
    y_pred_diag = np.array(y_pred_diag) > 0.5
    y_pred_class = np.array(y_pred_class) > 0.5
    y_true_diag = np.array(y_true_diag)
    y_true_class = np.array(y_true_class)

    def print_metrics(y_true, y_pred, task_name):
        """Print confusion matrix, accuracy, precision, recall, and F1 for a given task."""
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)

        print(f"\n{task_name} Metrics:")
        print(f" Accuracy:  {acc:.4f}")
        print(f" Precision: {prec:.4f}")
        print(f" Recall:    {rec:.4f}")
        print(f" F1 Score:  {f1:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{task_name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC for {task_name}")
        plt.legend(loc="lower right")
        plt.show()

        # Precision-Recall curve
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred)
        pr_auc = auc(recall_vals, precision_vals)
        plt.figure()
        plt.plot(
            recall_vals,
            precision_vals,
            color="blue",
            lw=2,
            label=f"PR AUC = {pr_auc:.2f}",
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall for {task_name}")
        plt.legend(loc="upper right")
        plt.show()

        return acc, prec, rec, f1

    # Print and plot metrics for both tasks
    diag_acc, diag_prec, diag_rec, diag_f1 = print_metrics(
        y_true_diag, y_pred_diag, "Diagnosis"
    )
    class_acc, class_prec, class_rec, class_f1 = print_metrics(
        y_true_class, y_pred_class, "Classification"
    )

    # Classification reports
    print("\nDiagnosis Classification Report:")
    print(classification_report(y_true_diag, y_pred_diag, target_names=["Neg", "Pos"]))
    print("ASD Classification Report:")
    print(
        classification_report(y_true_class, y_pred_class, target_names=["Neg", "Pos"])
    )

    # Return dictionary with final metrics
    return [
        {
            "model_name": "Transformer",
            "diagnosis_accuracy": diag_acc,
            "diagnosis_precision": diag_prec,
            "diagnosis_recall": diag_rec,
            "diagnosis_f1": diag_f1,
            "class_accuracy": class_acc,
            "class_precision": class_prec,
            "class_recall": class_rec,
            "class_f1": class_f1,
        }
    ]


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
