import os
import torch
from config import Config
from src.model import LitModel

# Load trained Lightning checkpoint once at startup
def load_model():
    ckpt_path = os.getenv("MODEL_CKPT_PATH", "best.ckpt")
    model = LitModel.load_from_checkpoint(ckpt_path, map_location=Config.DEVICE)
    model.eval()
    return model

model = load_model()

def predict_asd(features: list[float]) -> dict:
    # features = [A1,...,A10, age, gender]
    x = torch.tensor([features], dtype=torch.float32, device=Config.DEVICE)
    with torch.no_grad():
        logits = model.model(x)
        prob = torch.sigmoid(logits).item()
    return {"probability": prob, "decision": prob > 0.5}