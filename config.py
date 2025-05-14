from pathlib import Path
import torch


class Config:
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    PROJECT_PATH = Path.home() / "Projects" / "aq10-analysis"
    RAW_ADULT_DATA_PATH = PROJECT_PATH / "dataset/raw/autism_screening.csv"
    RAW_CHILD_DATA_PATH = PROJECT_PATH / "dataset/raw/child_data.csv"
    PROCESSED_DATA_DIR = PROJECT_PATH / "dataset/processed"
    CHECKPOINT_DIR = PROJECT_PATH / "checkpoints"
    MODEL_DIR = PROJECT_PATH / "models"

    MODEL_NAME = "bert-base-uncased"
    MODEL_DIM = 64
    NUM_LAYERS = 4
    DROPOUT = 0.2
    LEARNING_RATE = 1.9e-3
    BATCH_SIZE = 64
    EPOCHS = 30
    FEATURE_LEN = 12

    @classmethod
    def setup_directories(cls):
        for d in (cls.PROCESSED_DATA_DIR, cls.CHECKPOINT_DIR, cls.MODEL_DIR):
            d.mkdir(parents=True, exist_ok=True)
