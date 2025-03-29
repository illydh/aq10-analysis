import matplotlib.pyplot as plt
import seaborn as sns


def plot_comparison(results_df):
    """Plot model comparison"""
    plt.figure(figsize=(12, 6))

    # Diagnosis F1 Comparison
    plt.subplot(1, 2, 1)
    sns.barplot(data=results_df, x="Model", y="Diagnosis F1")
    plt.title("Diagnosis F1 Score Comparison")
    plt.xticks(rotation=45)

    # Classification F1 Comparison
    plt.subplot(1, 2, 2)
    sns.barplot(data=results_df, x="Model", y="Classification F1")
    plt.title("Classification F1 Score Comparison")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("reports/figures/model_comparison.png")
    plt.show()
