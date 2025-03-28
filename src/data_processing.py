import polars as pl
from typing import Dict
from config import Config


class DataProcessor:
    """Handles loading and preprocessing of autism screening data."""

    @staticmethod
    def load_raw_data() -> pl.DataFrame:
        """Load and preprocess raw data."""
        return (
            pl.read_csv(Config.RAW_DATA_PATH)
            .drop(
                [
                    "age_desc",
                    "ethnicity",
                    "jundice",
                    "contry_of_res",
                    "used_app_before",
                    "relation",
                ]
            )
            .with_columns(
                [
                    pl.when(pl.col("gender") == "m")
                    .then(1)
                    .otherwise(0)
                    .alias("gender"),
                    pl.when(pl.col("austim") == "yes")
                    .then(1)
                    .otherwise(0)
                    .alias("diagnosis"),
                    pl.when(pl.col("Class/ASD") == "YES")
                    .then(1)
                    .otherwise(0)
                    .alias("classification"),
                ]
            )
            .drop(["austim", "Class/ASD"])
            .filter(pl.col("age").is_not_null())
        )

    @staticmethod
    def split_data(df: pl.DataFrame) -> Dict[str, pl.DataFrame]:
        """Split data into train/val/test sets (70/15/15)."""
        total_rows = df.height
        train_size = int(0.7 * total_rows)
        val_size = int(0.15 * total_rows)

        # Shuffle the DataFrame
        df_shuffled = df.sample(fraction=1, seed=42)
        features_df = df_shuffled.drop(["diagnosis", "classification"])
        tgt_df = df_shuffled.select(["diagnosis", "classification"])

        return {
            "train_features": features_df.slice(0, train_size),
            "train_targets": tgt_df.slice(0, train_size),
            "val_features": features_df.slice(train_size, val_size),
            "val_targets": tgt_df.slice(train_size, val_size),
            "test_features": features_df.slice(train_size + val_size, total_rows),
            "test_targets": tgt_df.slice(train_size + val_size, total_rows),
        }
