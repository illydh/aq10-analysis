import pickle
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns

# from config import Config
# from model import LitModel


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
        str(Config.CHECKPOINT_DIR / "best-checkpoint-v6.ckpt")
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

        ### Confusion matrix
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

        ### plot_roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve for {task_name}")
        plt.legend(loc="lower right")
        plt.show()

        ### plot_precision_recall_curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(8, 6))
        plt.plot(
            recall,
            precision,
            color="blue",
            lw=2,
            label=f"PR curve (AUC = {pr_auc:.2f})",
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve for {task_name}")
        plt.legend(loc="upper right")
        plt.show()

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
        }

    evals_diag = print_metrics(y_true_diag, y_pred_diag, "Diagnosis")
    evals_class = print_metrics(y_true_class, y_pred_class, "Classification")

    ### Accuracy Bar plot
    accuracies = {
        "Diagnosis": evals_diag["accuracy"],
        "Classification": evals_class["accuracy"],
    }
    plt.figure(figsize=(8, 5))
    plt.bar(accuracies.keys(), accuracies.values(), color=["blue", "green"])
    plt.title("Model Accuracy on Test Data")
    plt.xlabel("Task")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    for i, v in enumerate(accuracies.values()):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=12)
    plt.show()

    ### classification report
    print("Classification Report for Diagnosis:")
    print(
        classification_report(
            y_true_diag, y_pred_diag, target_names=["Negative", "Positive"]
        )
    )
    print("Classification Report for Classification:")
    print(
        classification_report(
            y_true_class, y_pred_class, target_names=["Negative", "Positive"]
        )
    )

    ### Summary Data
    data = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Diagnosis": [
            evals_diag["accuracy"],
            evals_diag["precision"],
            evals_diag["recall"],
            evals_diag["f1"],
        ],
        "Classification": [
            evals_class["accuracy"],
            evals_class["precision"],
            evals_class["recall"],
            evals_class["f1"],
        ],
    }
    print("Summary table:\n", pd.DataFrame(data))


if __name__ == "__main__":
    evaluate_model()
