import pickle
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch

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


def compute_pos_weight(targets_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute positive weight for BCEWithLogitsLoss to address class imbalance.
    pos_weight = num_neg / num_pos
    """
    total = targets_tensor.numel()
    pos = torch.sum(targets_tensor)
    neg = total - pos
    weight = neg / (pos + 1e-6)
    return torch.tensor(min(weight.item(), 5.0))  # clip to prevent over-weighting


def train():
    # Setup
    Config.setup_directories()
    processor = DataProcessor()

    print("Loading and processing raw data...")
    raw_df = processor.load_raw_data()
    splits = processor.split_data(raw_df)

    # Convert diagnosis labels to torch tensors for weight calc
    train_targets_tensor = torch.tensor(
        splits["train_targets"]["diagnosis"].to_numpy(), dtype=torch.float32
    )
    pos_weight = compute_pos_weight(train_targets_tensor)

    # Build datasets
    train_dataset = ASDDataset(
        splits["train_features"], splits["train_targets"], augment=True
    )
    val_dataset = ASDDataset(splits["val_features"], splits["val_targets"])
    test_dataset = ASDDataset(splits["test_features"], splits["test_targets"])

    # Save preprocessed datasets
    for name, ds in zip(
        ["train", "val", "test"], [train_dataset, val_dataset, test_dataset]
    ):
        with open(Config.PROCESSED_DATA_DIR / f"{name}_dataset.pkl", "wb") as f:
            pickle.dump(ds, f)

    # Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, num_workers=4)

    # Model
    model = LitModel(pos_weight=pos_weight)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=Config.CHECKPOINT_DIR,
        filename="best-checkpoint",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    # Train
    print("Starting training...")
    trainer = Trainer(
        max_epochs=Config.EPOCHS,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10,
        logger=False,
    )
    trainer.fit(model, train_loader, val_loader)

    # Evaluate
    print("Testing model on held-out set...")
    results = trainer.test(model, test_loader)
    print("Test results:", results)

    # Train baseline models
    # print("\nTraining baseline models...")
    # train_baselines(train_dataset, test_dataset)
    # print("All training routines complete.")
