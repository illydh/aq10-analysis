import pickle
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from config import Config
from data_processing import DataProcessor
from dataset import ASDDataset
from model import LitModel
from baseline_models import BaselineModels


def train_baselines(train_dataset, test_dataset):
    """
    Trains the baseline models.
    Uses the entire train_dataset for training and test_dataset for evaluation.
    Results are printed and models are saved as .joblib files.
    """
    # Convert train_dataset to NumPy
    train_features = np.array([sample[0] for sample in train_dataset])
    train_targets = np.array([sample[1] for sample in train_dataset])

    # Convert test_dataset to NumPy
    test_features = np.array([sample[0] for sample in test_dataset])
    test_targets = np.array([sample[1] for sample in test_dataset])

    # Initialize BaselineModels
    baselines = BaselineModels(
        train_features, train_targets, test_features, test_targets
    )
    # Train and evaluate
    df_results = baselines.train_and_evaluate()

    print("\nBaseline Models Training Complete.")
    print("Baseline performance on test set:\n", df_results)


def train():
    # Setup directories
    Config.setup_directories()

    # Load and process data
    print("Loading and processing data...")
    processor = DataProcessor()
    raw_df = processor.load_raw_data()
    splits = processor.split_data(raw_df)

    # Create datasets
    train_dataset = ASDDataset(splits["train_features"], splits["train_targets"])
    val_dataset = ASDDataset(splits["val_features"], splits["val_targets"])
    test_dataset = ASDDataset(splits["test_features"], splits["test_targets"])

    # Save processed datasets
    with open(Config.PROCESSED_DATA_DIR / "train_dataset.pkl", "wb") as f:
        pickle.dump(train_dataset, f)
    with open(Config.PROCESSED_DATA_DIR / "val_dataset.pkl", "wb") as f:
        pickle.dump(val_dataset, f)
    with open(Config.PROCESSED_DATA_DIR / "test_dataset.pkl", "wb") as f:
        pickle.dump(test_dataset, f)

    # Create data loaders for Transformer
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, num_workers=4)

    # Initialize Transformer model
    model = LitModel()

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=Config.CHECKPOINT_DIR,
        filename="best-checkpoint",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=3, mode="min")

    # Train Transformer
    print("Starting Transformer training...")
    trainer = Trainer(
        max_epochs=Config.EPOCHS,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10,
    )
    trainer.fit(model, train_loader, val_loader)
    print("Transformer training complete.")

    # Evaluate Transformer on the test set
    print("Evaluating Transformer on test set...")
    test_results = trainer.test(model, test_loader)
    print("Transformer test results:", test_results)

    # -----------------------------
    # Train baseline models
    # -----------------------------
    print("\nTraining baseline models...")
    train_baselines(train_dataset, test_dataset)
    print("All training routines complete.")


if __name__ == "__main__":
    train()
