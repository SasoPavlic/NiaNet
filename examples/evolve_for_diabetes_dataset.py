import pandas as pd
import torch

from sklearn.datasets import load_diabetes
from datetime import datetime
from nianet.problem import find_architecture

if __name__ == '__main__':
    start = datetime.now().strftime("%H:%M:%S-%d/%m/%Y")
    print(f"Program start... {start}")

    dataset = load_diabetes()
    df_dataset = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
    df_dataset['target'] = pd.Series(dataset.target)

    model = find_architecture(df_dataset)
    MODEL_PATH = f"model_{str(datetime.now())}_best_model.pth"
    torch.save(model, MODEL_PATH)

    end = datetime.now().strftime("%H:%M:%S-%d/%m/%Y")
    print(f"Program end... {end}")
