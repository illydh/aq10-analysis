from pathlib import Path

class Config:
    # Data configuration
    RAW_DATA_PATH = Path("data/raw/autism_screening.csv")
    PROCESSED_DATA_DIR = Path("data/processed")
    CHECKPOINT_DIR = Path("checkpoints")
    
    # Model hyperparameters
    MODEL_DIM = 64
    NUM_LAYERS = 4
    DROPOUT = 0.2
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 64
    EPOCHS = 20
    
    # Data processing
    FEATURE_LEN = 13  # AQ10 scores + age + gender + result
    
    @classmethod
    def setup_directories(cls):
        cls.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)