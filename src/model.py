import torch
import torch.nn as nn
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall
from pytorch_lightning import LightningModule
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from config import Config


class Model(nn.Module):
    def __init__(self):
        """
        Initialize the model with the specified configuration.

        The model consists of a feature normalizer, a feature embedding layer,
        a positional embedding layer, a transformer encoder, a dropout layer,
        and a classification head.

        The feature normalizer normalizes the input data to have zero mean and
        unit variance.

        The feature embedding layer embeds the input data into a higher-dimensional
        space using a learnable linear transformation.

        The positional embedding layer adds learnable positional embeddings to the
        input data.

        The transformer encoder applies a multi-layer transformer to the input
        data.

        The dropout layer applies dropout to the output of the transformer encoder.

        The classification head applies a linear transformation to the output of the
        dropout layer to produce the final output.
        """
        super().__init__()
        dim_feedforward = Config.MODEL_DIM * 4
        num_heads = max(2, Config.MODEL_DIM // 16)

        self.feature_norm = nn.BatchNorm1d(Config.FEATURE_LEN)

        self.feature_embedding = nn.Sequential(
            nn.Linear(1, Config.MODEL_DIM),
            nn.GELU(),
            nn.LayerNorm(Config.MODEL_DIM),
        )

        # Learnable positional embeddings per feature index
        self.positional_embedding = nn.Parameter(
            torch.randn(Config.FEATURE_LEN, Config.MODEL_DIM)
        )

        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=Config.MODEL_DIM,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=Config.DROPOUT,
                batch_first=True,
            ),
            num_layers=Config.NUM_LAYERS,
        )

        self.dropout = nn.Dropout(Config.DROPOUT)

        self.classifier_head = nn.Sequential(
            nn.LayerNorm(Config.MODEL_DIM),
            nn.Linear(Config.MODEL_DIM, Config.MODEL_DIM // 2),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.MODEL_DIM // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Applies feature normalization, feature embedding, and adds learnable 
        positional embeddings to the input tensor. The transformer encoder processes
        the embedded input, and a dropout layer is applied to the output. The first 
        feature's representation is used as an anchor token to generate logits 
        through the classification head.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, feature_len].

        Returns:
            torch.Tensor: Logits tensor of shape [batch_size], representing the 
            model's predictions.
        """
        x = self.feature_norm(x)  # [B, 12]
        x = x.unsqueeze(-1)  # [B, 12, 1]
        x = self.feature_embedding(x)  # [B, 12, D]

        # Add learnable positional embedding
        x = x + self.positional_embedding.unsqueeze(0)  # [1, 12, D]

        x = self.encoder(x)
        x = self.dropout(x)
        x_cls = x[:, 0]  # Use first feature (e.g. AQ1) as anchor token
        logits = self.classifier_head(x_cls).squeeze(-1)
        return logits


class LitModel(LightningModule):
    def __init__(self, pos_weight):
        """
        Initializes the LitModel class.

        Args:
            pos_weight (torch.Tensor): A tensor to handle class imbalance by assigning 
                more weight to the positive class in the binary classification task.

        Attributes:
            model (Model): The core model used for predictions.
            loss_fn (nn.BCEWithLogitsLoss): Loss function with class imbalance adjustment.
            f1 (BinaryF1Score): Metric to compute F1 score for evaluation.
            precision (BinaryPrecision): Metric to compute precision for evaluation.
            recall (BinaryRecall): Metric to compute recall for evaluation.
        """

        super().__init__()
        self.save_hyperparameters()
        self.model = Model()

        # Address class imbalance
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # TorchMetrics for logging
        self.f1 = BinaryF1Score(threshold=0.5)
        self.precision = BinaryPrecision(threshold=0.5)
        self.recall = BinaryRecall(threshold=0.5)

    def forward(self, x):
        return self.model(x)

    def smooth_labels(self, y: torch.Tensor, smoothing=0.05) -> torch.Tensor:
        return y * (1 - smoothing) + 0.5 * smoothing

    def step(self, batch, batch_idx, stage: str):
        x, y = batch
        logits = self(x)

        # Apply label smoothing during training only
        # if stage == "train":
        #     y = self.smooth_labels(y)

        loss = self.loss_fn(logits, y.squeeze())
        preds = torch.sigmoid(logits) > 0.5
        acc = (preds == y.bool()).float().mean()

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=Config.LEARNING_RATE)
