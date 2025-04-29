from train import train as train_transformer
from evaluate import evaluate_transformer
import pandas as pd

if __name__ == "__main__":
    train_transformer()

    t_metrics = evaluate_transformer()
    
    df = pd.DataFrame(t_metrics)
    print("\n\nModel results:\n", df)
