import pickle
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config
from model import LitModel


def evaluate_model():
    # Load test dataset
    with open(Config.PROCESSED_DATA_DIR / "test_dataset.pkl", "rb") as f:
        test_dataset = pickle.load(f)

    # Create data loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, num_workers=4
    )

    # Load trained model
    model = LitModel.load_from_checkpoint(
        str(Config.CHECKPOINT_DIR / "best-checkpoint.ckpt")
    )
    model.eval()

    # Get predictions
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

    # Convert to binary predictions
    y_pred_diag = np.array(y_pred_diag) > 0.5
    y_pred_class = np.array(y_pred_class) > 0.5
    y_true_diag = np.array(y_true_diag)
    y_true_class = np.array(y_true_class)

    # Calculate metrics
    def print_metrics(y_true, y_pred, task_name):
        print(f"\n{task_name} Metrics:")
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
        print(f"Precision: {precision_score(y_true, y_pred):.4f}")
        print(f"Recall: {recall_score(y_true, y_pred):.4f}")
        print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
        )
        plt.title(f"{task_name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    print_metrics(y_true_diag, y_pred_diag, "Diagnosis")
    print_metrics(y_true_class, y_pred_class, "Classification")


if __name__ == "__main__":
    evaluate_model()
