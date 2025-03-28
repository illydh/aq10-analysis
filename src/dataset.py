from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Dict
import multiprocessing as mp
from tqdm import tqdm


class ASDDataset(Dataset):
    """PyTorch Dataset for autism screening data."""

    def __init__(self, feature_df, tgt_df):
        self.feature_df = feature_df.to_pandas(use_pyarrow_extension_array=True)
        self.tgt_df = tgt_df.to_pandas(use_pyarrow_extension_array=True)

        # Compute normalization parameters
        self.age_min = self.feature_df["age"].min()
        self.age_max = self.feature_df["age"].max()
        self.result_min = self.feature_df["result"].min()
        self.result_max = self.feature_df["result"].max()

        self.keys = self.feature_df.index.tolist()
        self._precompute_features()

    def _precompute_features(self):
        """Precompute features for faster training."""
        self.tgt_arrays: Dict[int, np.ndarray] = {}
        self.feature_arrays: Dict[int, np.ndarray] = {}

        with mp.Pool(processes=min(8, mp.cpu_count())) as pool:
            results = pool.map(
                self._process_key,
                tqdm(self.keys, desc="Precomputing features", total=len(self.keys)),
            )

            for key, tgt_array, feature_array in results:
                self.tgt_arrays[key] = tgt_array
                self.feature_arrays[key] = feature_array

    def _process_key(self, key: int):
        """Process a single data sample."""
        tgt_array = self._transform_target(self.tgt_df.loc[key])
        feature_array = self._transform_features(self.feature_df.loc[key])
        return key, tgt_array, feature_array

    def _transform_features(self, row: pd.Series) -> np.ndarray:
        """Transform input features into numpy array."""
        features = row[
            [f"A{i}_Score" for i in range(1, 11)] + ["age", "gender", "result"]
        ]
        features["age"] = (features["age"] - self.age_min) / (
            self.age_max - self.age_min
        )
        features["result"] = (features["result"] - self.result_min) / (
            self.result_max - self.result_min
        )
        return features.values.astype(np.float32)

    def _transform_target(self, row: pd.Series) -> np.ndarray:
        """Transform target into numpy array."""
        return np.array([row["diagnosis"], row["classification"]], dtype=np.float32)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        return self.feature_arrays[key], self.tgt_arrays[key]
