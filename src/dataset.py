from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch


class ASDDataset(Dataset):
    """PyTorch Dataset for autism screening data with optional augmentation."""

    def __init__(self, feature_df, tgt_df, augment: bool = False):
        self.feature_df = feature_df.to_pandas(use_pyarrow_extension_array=True)
        self.tgt_df = tgt_df.to_pandas(use_pyarrow_extension_array=True)
        self.augment = augment

        self.age_min = self.feature_df["age"].min()
        self.age_max = self.feature_df["age"].max()
        self.keys = self.feature_df.index.tolist()
        self._precompute_features()

    def _precompute_features(self):
        """Precompute features for faster training."""
        self.tgt_arrays = {}
        self.feature_arrays = {}

        for key in self.keys:
            self.tgt_arrays[key] = self._transform_target(self.tgt_df.loc[key])
            self.feature_arrays[key] = self._transform_features(
                self.feature_df.loc[key]
            )

    def _transform_features(self, row: pd.Series) -> np.ndarray:
        features = row[[f"A{i}_Score" for i in range(1, 11)] + ["age", "gender"]].copy()
        features["age"] = (features["age"] - self.age_min) / (
            self.age_max - self.age_min
        )
        return features.values.astype(np.float32)

    def _transform_target(self, row: pd.Series) -> np.ndarray:
        return np.array([row["diagnosis"]], dtype=np.float32)

    def _augment_features(self, features: np.ndarray) -> np.ndarray:
        # Gaussian noise only for AQ10 + age (exclude gender)
        aq_age = features[:-1]
        gender = features[-1:]

        noise = np.random.normal(loc=0.0, scale=0.05, size=aq_age.shape)
        aq_age = aq_age + noise

        # Optional: randomly mask one AQ score (simulate missing answer)
        if np.random.rand() < 0.2:
            idx = np.random.randint(0, 10)
            aq_age[idx] = 0.0

        return np.concatenate([aq_age, gender])

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        features = self.feature_arrays[key]
        target = self.tgt_arrays[key]
        if self.augment:
            features = self._augment_features(features)

        features = torch.tensor(features, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        return features, target
