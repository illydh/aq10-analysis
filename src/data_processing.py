import polars as pl
from typing import Dict, List, Tuple
import torch
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedShuffleSplit

from config import Config


class DataProcessor:
    """Handles loading, preprocessing, and splitting of autism screening data using Polars."""

    def __init__(
        self,
        model_name: str = Config.MODEL_NAME,
    ):
        """
        Initialize the DataProcessor, set up tokenizer, etc.

        :param model_name: Name of the transformer model (e.g., 'bert-base-uncased')
        :param additional_special_tokens: Optional list of additional tokens to add to the tokenizer
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": ["[DEMOGRAPHIC] ", "[/DEMOGRAPHIC]"]}
        )

    @staticmethod
    def load_raw_data() -> pl.DataFrame:
        """
        Load and preprocess raw data from CSV using Polars.
        Drops irrelevant columns and encodes certain fields as numeric.
        Return the cleaned Polars DataFrame.
        """
        adult_df = (
            pl.read_csv(Config.RAW_ADULT_DATA_PATH)
            .drop(
                [
                    "age_desc",
                    "ethnicity",
                    "jundice",
                    "contry_of_res",
                    "used_app_before",
                    "relation",
                    "Class/ASD",
                ]
            )
            .rename(
                {
                    "austim": "diagnosis",
                    # "jundice" : "jaundice"
                }
            )
            .with_columns(
                [
                    pl.when(pl.col("gender") == "m")
                    .then(1)
                    .otherwise(0)
                    .alias("gender"),
                    pl.when(pl.col("diagnosis") == "yes")
                    .then(1)
                    .otherwise(0)
                    .alias("diagnosis"),
                    pl.col("age").cast(pl.Int64),
                ]
            )
            .filter(pl.col("age").is_not_null())
        )

        child_df = (
            pl.read_csv(Config.RAW_CHILD_DATA_PATH)
            .drop(
                [
                    "CASE_NO_PATIENT'S",
                    "Social_Responsiveness_Scale",
                    "Qchat_10_Score",
                    "Speech Delay/Language Disorder",
                    "Learning disorder",
                    "Genetic_Disorders",
                    "Depression",
                    "Global developmental delay/intellectual disability",
                    "Social/Behavioural Issues",
                    "Childhood Autism Rating Scale",
                    "Anxiety_disorder",
                    "Ethnicity",
                    "Jaundice",
                    "Family_mem_with_ASD",
                    "Who_completed_the_test",
                ]
            )
            .rename(
                {
                    "A1": "A1_Score",
                    "A2": "A2_Score",
                    "A3": "A3_Score",
                    "A4": "A4_Score",
                    "A5": "A5_Score",
                    "A6": "A6_Score",
                    "A7": "A7_Score",
                    "A8": "A8_Score",
                    "A9": "A9_Score",
                    "A10_Autism_Spectrum_Quotient": "A10_Score",
                    "Age_Years": "age",
                    "Sex": "gender",
                    # "Jaundice": "jaundice",
                    # "Ethnicity": "ethnicity",
                    "ASD_traits": "diagnosis",
                }
            )
            .with_columns(
                [
                    pl.when(pl.col("gender") == "M")
                    .then(1)
                    .otherwise(0)
                    .alias("gender"),
                    pl.when(pl.col("diagnosis") == "Yes")
                    .then(1)
                    .otherwise(0)
                    .alias("diagnosis"),
                    pl.col("age").cast(pl.Int64),
                ]
            )
        )

        common_cols = list(set(adult_df.columns).intersection(set(child_df.columns)))
        adult_df = adult_df.select(common_cols)
        child_df = child_df.select(common_cols)

        raw_df = pl.concat([adult_df, child_df], how="vertical")

        counts = raw_df["diagnosis"].value_counts().to_dict()
        print(f"Class distribution: {counts}")

        return raw_df

    @staticmethod
    def split_data(df: pl.DataFrame) -> Dict[str, pl.DataFrame]:
        """
        Split data into stratified train/val/test sets (70/15/15),
        preserving the diagnosis class distribution.
        """
        # Convert Polars DataFrame to pandas for StratifiedShuffleSplit
        pdf = df.to_pandas()

        # First split: train (70%) vs temp (30%)
        sss1 = StratifiedShuffleSplit(
            n_splits=1, test_size=0.3, random_state=Config.SEED
        )
        train_idx, temp_idx = next(sss1.split(pdf, pdf["diagnosis"]))
        train_pdf = pdf.iloc[train_idx]
        temp_pdf = pdf.iloc[temp_idx]

        # Second split: temp → val (15%) and test (15%) of original
        sss2 = StratifiedShuffleSplit(
            n_splits=1, test_size=0.5, random_state=Config.SEED
        )
        val_idx, test_idx = next(sss2.split(temp_pdf, temp_pdf["diagnosis"]))
        val_pdf = temp_pdf.iloc[val_idx]
        test_pdf = temp_pdf.iloc[test_idx]

        # Convert back to Polars
        train_df = pl.from_pandas(train_pdf)
        val_df = pl.from_pandas(val_pdf)
        test_df = pl.from_pandas(test_pdf)

        # Prepare feature / target splits
        return {
            "train_features": train_df.drop(["diagnosis"]),
            "train_targets": train_df.select(["diagnosis"]),
            "val_features": val_df.drop(["diagnosis"]),
            "val_targets": val_df.select(["diagnosis"]),
            "test_features": test_df.drop(["diagnosis"]),
            "test_targets": test_df.select(["diagnosis"]),
        }

    @staticmethod
    def format_demographic_data(row: dict) -> str:
        """
        Convert demographic data into a string with special tokens.
        Adjust fields as appropriate for your data.
        E.g., row["age"], row["gender"], etc.
        """
        demographic_str = (
            f"[DEMOGRAPHIC] "
            f"Age: {row.get('age', '')}, "
            f"Gender: {row.get('gender', '')} "
            f"[/DEMOGRAPHIC]"
        )
        return demographic_str

    @staticmethod
    def format_aq10_responses(row: dict) -> str:
        """
        Convert the row's question responses (Q1..Q10) into a text format.
        Adjust as needed if your data has different question names.
        """
        responses = []
        for i in range(1, 11):
            col_name = f"A{i}_Score"
            if col_name in row:
                responses.append(f"Q{i}: {row[col_name]}")
            else:
                # If missing, you can adapt how to handle it (e.g., blank)
                responses.append(f"Q{i}: ")
        return " ".join(responses)

    def preprocess_data(
        self, df: pl.DataFrame, label_cols: List[str] = None
    ) -> Tuple[Dict[str, torch.Tensor], List]:
        """
        Preprocess the Polars DataFrame into tokenized input_ids & attention_masks.
        If label_cols is provided (e.g. ["diagnosis", "classification"]),
        returns labels; otherwise returns None for labels.

        :param df: Polars DataFrame
        :param label_cols: List of columns to treat as labels, e.g. ["diagnosis"].
        :return: (encodings, labels)
        """
        if label_cols is None:
            label_cols = []

        # Convert to list of dicts for easy iteration
        dict_rows = df.to_dicts()

        texts = []
        labels = []

        for row in dict_rows:
            # Format demographic info
            demographic_text = self.format_demographic_data(row)
            # Format AQ10 responses
            aq_text = self.format_aq10_responses(row)

            # Combine into one string
            full_text = demographic_text + " " + aq_text
            texts.append(full_text)

            # Gather labels if present
            if label_cols:
                # If label_cols is ["diagnosis"], we take row["diagnosis"]
                # If multiple columns, we store them as a tuple
                row_labels = tuple(row[col] for col in label_cols)
                labels.append(row_labels if len(label_cols) > 1 else row_labels[0])
            else:
                labels.append(None)

        # Tokenize the texts
        encodings = self.tokenizer(
            texts, truncation=True, padding=True, return_tensors="pt"
        )

        return encodings, labels
