import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall
from pytorch_lightning import LightningModule
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from config import Config


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        loss = self.alpha * (1 - p_t) ** self.gamma * bce
        return loss.mean() if self.reduction == "mean" else loss.sum()


class Model(nn.Module):
    def __init__(
        self,
        use_text: bool = False,
        text_model_name: str = "bert-base-uncased",
        ablate_demographic: bool = False,
        pooling: str = "first",
    ):
        super().__init__()
        # 1) Numeric features setup
        base_len = Config.FEATURE_LEN  # e.g. 12
        self.feature_len = base_len - 2 if ablate_demographic else base_len
        self.ablate_demographic = ablate_demographic
        self.pooling = pooling

        self.feature_norm = nn.BatchNorm1d(self.feature_len)
        self.feature_embedding = nn.Sequential(
            nn.Linear(1, Config.MODEL_DIM),
            nn.GELU(),
            nn.LayerNorm(Config.MODEL_DIM),
        )
        self.positional_embedding = nn.Parameter(
            torch.randn(self.feature_len, Config.MODEL_DIM)
        )

        dim_feedforward = Config.MODEL_DIM * 4
        num_heads = max(2, Config.MODEL_DIM // 16)
        encoder_layer = TransformerEncoderLayer(
            d_model=Config.MODEL_DIM,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=Config.DROPOUT,
            batch_first=True,
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=Config.NUM_LAYERS)
        self.dropout = nn.Dropout(Config.DROPOUT)

        # 2) Optional text encoder
        self.use_text = use_text
        if use_text:
            self.text_encoder = AutoModel.from_pretrained(text_model_name)
            text_hidden = self.text_encoder.config.hidden_size
        else:
            text_hidden = 0

        # 3) Classification head on fused dimension
        fused_dim = Config.MODEL_DIM + text_hidden
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(fused_dim // 2, 1),
        )

    def forward(
        self,
        x_numeric: torch.Tensor,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ):
        # Optionally drop demographic columns
        x = x_numeric[:, : self.feature_len] if self.ablate_demographic else x_numeric

        # Numeric transformer pathway
        x = self.feature_norm(x)
        x = x.unsqueeze(-1)
        x = self.feature_embedding(x)
        x = x + self.positional_embedding.unsqueeze(0)
        x = self.encoder(x)
        x = self.dropout(x)
        if self.pooling == "first":
            num_rep = x[:, 0]
        elif self.pooling == "mean":
            num_rep = x.mean(dim=1)
        else:
            raise ValueError(f"Unknown pooling mode {self.pooling}")

        # Text pathway + fusion
        if self.use_text:
            txt_out = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
            txt_rep = txt_out.last_hidden_state[:, 0, :]
            feat = torch.cat([num_rep, txt_rep], dim=1)
        else:
            feat = num_rep

        logits = self.classifier_head(feat).squeeze(-1)
        return logits


class LitModel(LightningModule):
    def __init__(
        self,
        pos_weight: torch.Tensor,
        gamma: float = 2.0,
        use_text: bool = False,
        text_model_name: str = "bert-base-uncased",
        ablate_demographic: bool = False,
        pooling: str = "first",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = Model(
            use_text=use_text,
            text_model_name=text_model_name,
            ablate_demographic=ablate_demographic,
            pooling=pooling,
        )
        self.loss_fn = FocalLoss(alpha=pos_weight, gamma=gamma)
        self.f1 = BinaryF1Score(threshold=0.5)
        self.precision = BinaryPrecision(threshold=0.5)
        self.recall = BinaryRecall(threshold=0.5)

    def forward(self, x_numeric, input_ids=None, attention_mask=None):
        return self.model(x_numeric, input_ids=input_ids, attention_mask=attention_mask)

    def step(self, batch, batch_idx, stage: str):
        if self.hparams.use_text:
            x_num, enc, y = batch
            logits = self(x_num, **enc)
        else:
            x_num, y = batch
            logits = self(x_num)

        loss = self.loss_fn(logits, y.squeeze())
        preds = torch.sigmoid(logits) > 0.5

        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_f1", self.f1(preds, y.squeeze()), prog_bar=True)
        self.log(f"{stage}_precision", self.precision(preds, y.squeeze()))
        self.log(f"{stage}_recall", self.recall(preds, y.squeeze()))

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=Config.LEARNING_RATE)
