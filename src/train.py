import pickle
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from config import Config
from data_processing import DataProcessor
from dataset import ASDDataset
from model import LitModel


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

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, num_workers=4)

    # Initialize model
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

    # Train model
    trainer = Trainer(
        max_epochs=Config.EPOCHS,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader, val_loader)

    # Evaluate on test set
    results = trainer.test(model, test_loader)
    return results


if __name__ == "__main__":
    train()
