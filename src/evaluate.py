from pathlib import Path
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

from config import Config
from model import LitModel


def evaluate():
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
    ckpt_path = Path(Config.CHECKPOINT_DIR) / "best-checkpoint.ckpt"
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

    # ─── Find optimal threshold by maximizing F1 ───
    thresholds = np.linspace(0.1, 0.9, 81)
    f1s = [f1_score(y_true, (y_pred_probs > t).astype(int)) for t in thresholds]
    best_t = thresholds[np.argmax(f1s)]
    print(f"\nBest F1 threshold on test set: {best_t:.2f}")

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


if __name__ == "__main__":
    evaluate()
