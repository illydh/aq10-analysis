from pathlib import Path
import torch


class Config:
    """
    Configuration class for the project.
    """

    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Data configuration
    PROJECT_PATH = Path("/content/drive/MyDrive/Projects/Spring-2025-Research")

    #   source: https://www.kaggle.com/datasets/andrewmvd/autism-screening-on-adults
    RAW_ADULT_DATA_PATH = PROJECT_PATH / Path("dataset/autism_screening.csv")
    #   source: https://www.kaggle.com/datasets/uppulurimadhuri/dataset/data
    RAW_CHILD_DATA_PATH = PROJECT_PATH / Path("dataset/data_csv.csv")

    PROCESSED_DATA_DIR = PROJECT_PATH / Path("dataset/processed")
    CHECKPOINT_DIR = PROJECT_PATH / Path("checkpoints")
    MODEL_DIR = PROJECT_PATH / Path("models")

    # Model hyperparameters
    MODEL_NAME = "bert-base-uncased"
    MODEL_DIM = 64
    NUM_LAYERS = 4
    DROPOUT = 0.2
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 64
    EPOCHS = 20

    # Data processing
    FEATURE_LEN = 12  # AQ10 scores + age + gender

    @classmethod
    def setup_directories(cls):
        """
        Setup directories for storing processed data, model checkpoints, and model files.
        """

        cls.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
