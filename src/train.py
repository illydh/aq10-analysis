import pickle
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, WeightedRandomSampler
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import optuna

from config import Config
from data_preprocessing import DataProcessor
from dataset import ASDDataset
from model import LitModel


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

    # Compute pos_weight
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

    # ── OVERSAMPLING ──
    labels = [int(y.item()) for _, y in train_dataset]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[l] for l in labels]
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # Save preprocessed datasets
    for name, ds in zip(
        ["train", "val", "test"], [train_dataset, val_dataset, test_dataset]
    ):
        with open(Config.PROCESSED_DATA_DIR / f"{name}_dataset.pkl", "wb") as f:
            pickle.dump(ds, f)

    # DataLoaders (use sampler for train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,
        num_workers=4,
        drop_last=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, num_workers=4)

    # Model
    model = LitModel(pos_weight=pos_weight)

    # ── Loggers ──
    tb_logger = TensorBoardLogger(
        save_dir="tb_logs", name="ASD_Transformer", default_hp_metric=False
    )
    csv_logger = CSVLogger(save_dir="csv_logs", name="ASD_Transformer")

    # Callbacks: checkpoint on best val_f1, early stop on val_f1
    checkpoint_callback = ModelCheckpoint(
        dirpath=Config.CHECKPOINT_DIR,
        filename="best-{epoch:02d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
    )
    early_stop_callback = EarlyStopping(monitor="val_f1", patience=5, mode="max")

    # Trainer: attach both loggers
    trainer = Trainer(
        max_epochs=Config.EPOCHS,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=[tb_logger, csv_logger],
        log_every_n_steps=1,
    )

    # ── Training ──
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)

    # ── Plot training curves from CSV logs ──
    metrics_path = f"{csv_logger.log_dir}/metrics.csv"
    df = pd.read_csv(metrics_path)
    epoch_df = df.groupby("epoch").mean().reset_index()

    plt.figure()
    plt.plot(epoch_df["epoch"], epoch_df["train_loss_step"], label="Train Loss")
    plt.plot(epoch_df["epoch"], epoch_df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss vs Epoch")
    plt.show()

    plt.figure()
    plt.plot(epoch_df["epoch"], epoch_df["train_f1"], label="Train F1")
    plt.plot(epoch_df["epoch"], epoch_df["val_f1"], label="Val F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.title("F1 Score vs Epoch")
    plt.show()

    # ── Final Evaluation ──
    print("Testing model on held-out set...")
    results = trainer.test(model, test_loader)
    print("Test results:", results)


def objective(trial):
    # 1) Suggest a learning rate between 1e-5 and 1e-2
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)

    # 2) Reload data splits (or cache outside this function if you prefer)
    processor = DataProcessor()
    raw_df = processor.load_raw_data()
    splits = processor.split_data(raw_df)
    train_ds = ASDDataset(
        splits["train_features"], splits["train_targets"], augment=True
    )
    val_ds = ASDDataset(splits["val_features"], splits["val_targets"])

    # 3) Compute pos_weight for FocalLoss or BCEWithLogitsLoss
    train_labels = [int(y.item()) for _, y in train_ds]
    pos_weight = (len(train_labels) - sum(train_labels)) / (sum(train_labels) + 1e-6)
    pos_weight = np.clip(pos_weight, 1.0, 5.0)
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32)

    # 4) (Optional) Oversample positives
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,
        num_workers=4,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, num_workers=4)

    # 5) Instantiate model with this trial's LR
    model = LitModel(pos_weight=pos_weight)
    model.hparams.lr = lr  # override default LR

    # 6) Set up a Trainer that logs val_loss
    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        dirpath=str(Config.CHECKPOINT_DIR),
        filename="optuna-{epoch:02d}-{val_loss:.4f}",
    )
    earlystop_cb = EarlyStopping(monitor="val_loss", patience=3, mode="min")
    trainer = Trainer(
        max_epochs=Config.EPOCHS,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_cb, earlystop_cb],
        logger=False,
    )

    # 7) Train & fetch validation loss
    trainer.fit(model, train_loader, val_loader)
    val_loss = trainer.callback_metrics["val_loss"].item()

    # Report intermediate result to Optuna
    trial.report(val_loss, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return val_loss


if __name__ == "__main__":
    train()

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=20)

    print("Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f" • {k}: {v}")
