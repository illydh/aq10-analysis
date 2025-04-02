from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import joblib


class BaselineModels:
    def __init__(self, train_features, train_targets, test_features, test_targets):
        self.models = {
            "SVM": SVC(probability=True, kernel="rbf"),
            "Decision Tree": DecisionTreeClassifier(max_depth=5),
            "Random Forest": RandomForestClassifier(n_estimators=100),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "Naive Bayes": GaussianNB(),
            "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
            "LightGBM": LGBMClassifier(),
        }
        self.train_features = train_features
        self.train_targets = train_targets
        self.test_features = test_features
        self.test_targets = test_targets
        self.results = []

    def train_and_evaluate(self):
        """Train all baseline models and evaluate performance"""
        for name, model in self.models.items():
            print(f"\nTraining {name}...")

            # Train on diagnosis task
            model.fit(self.train_features, self.train_targets[:, 0])
            diag_preds = model.predict(self.test_features)

            # Train on classification task
            model.fit(self.train_features, self.train_targets[:, 1])
            class_preds = model.predict(self.test_features)

            # Store results
            self.results.append(
                {
                    "Model": name,
                    "Diagnosis Accuracy": accuracy_score(
                        self.test_targets[:, 0], diag_preds
                    ),
                    "Diagnosis F1": f1_score(self.test_targets[:, 0], diag_preds),
                    "Classification Accuracy": accuracy_score(
                        self.test_targets[:, 1], class_preds
                    ),
                    "Classification F1": f1_score(self.test_targets[:, 1], class_preds),
                }
            )

            # Save models
            joblib.dump(
                model, Config.MODEL_DIR / f"{name.lower().replace(' ', '_')}.joblib"
            )

        return pd.DataFrame(self.results)

    def generate_report(self, transformer_metrics):
        """Generate comparison report with transformer model"""
        baseline_df = pd.DataFrame(self.results)

        # Add transformer results
        transformer_row = {
            "Model": "Transformer",
            "Diagnosis Accuracy": transformer_metrics["diagnosis_accuracy"],
            "Diagnosis F1": transformer_metrics["diagnosis_f1"],
            "Classification Accuracy": transformer_metrics["class_accuracy"],
            "Classification F1": transformer_metrics["class_f1"],
        }

        return pd.concat(
            [baseline_df, pd.DataFrame([transformer_row])], ignore_index=True
        )
