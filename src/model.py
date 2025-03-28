import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from pytorch_lightning import LightningModule
from config import Config


class Model(nn.Module):
    """Transformer model for autism diagnosis prediction."""

    def __init__(self):
        super().__init__()
        dim_feedforward = Config.MODEL_DIM * 4
        num_heads = max(2, Config.MODEL_DIM // 16)

        # Feature processing
        self.feature_norm = nn.BatchNorm1d(Config.FEATURE_LEN)
        self.feature_embedding = nn.Sequential(
            nn.Linear(1, Config.MODEL_DIM),
            nn.GELU(),
            nn.LayerNorm(Config.MODEL_DIM),
            nn.Dropout(Config.DROPOUT),
        )

        # Transformer encoder
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

        # Prediction heads
        self.diagnosis_head = self._build_head()
        self.classification_head = self._build_head()

    def _build_head(self):
        return nn.Sequential(
            nn.Linear(Config.MODEL_DIM, Config.MODEL_DIM // 2),
            nn.GELU(),
            nn.LayerNorm(Config.MODEL_DIM // 2),
            nn.Linear(Config.MODEL_DIM // 2, 1),
        )

    def forward(self, x):
        x = self.feature_norm(x)
        x = x.unsqueeze(-1)
        x = self.feature_embedding(x)
        x = self.encoder(x)
        x = x.mean(dim=1)  # Average pooling

        diagnosis = self.diagnosis_head(x).squeeze(-1)
        classification = self.classification_head(x).squeeze(-1)
        return diagnosis, classification


class LitModel(LightningModule):
    """Lightning module for training the autism prediction model."""

    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = Model()
        self.diag_loss = nn.BCEWithLogitsLoss()
        self.class_loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        diag_logits, class_logits = self.model(x)

        diag_loss = self.diag_loss(diag_logits, y[:, 0])
        class_loss = self.class_loss(class_logits, y[:, 1])
        total_loss = (diag_loss + class_loss) / 2

        self.log_dict(
            {
                "train_loss": total_loss,
                "train_diag_loss": diag_loss,
                "train_class_loss": class_loss,
            },
            prog_bar=True,
        )

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        diag_logits, class_logits = self.model(x)

        diag_loss = self.diag_loss(diag_logits, y[:, 0])
        class_loss = self.class_loss(class_logits, y[:, 1])
        total_loss = (diag_loss + class_loss) / 2

        self.log_dict(
            {
                "val_loss": total_loss,
                "val_diag_loss": diag_loss,
                "val_class_loss": class_loss,
            },
            prog_bar=True,
        )

        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=Config.LEARNING_RATE)
