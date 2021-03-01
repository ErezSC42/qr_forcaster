import os
import pandas as pd
from model import Encoder
from MyDataset import ElDataset

DATA_PATH = os.path.join("data", "LD2011_2014.txt")

if __name__ == '__main__':
    samples = 1
    hist_hours = 168
    pred_horizon = 24
    # load data
    df = pd.read_csv(DATA_PATH,
                     parse_dates=[0],
                     delimiter=";",
                     decimal=",")
    df.rename({"Unnamed: 0": "timestamp"}, axis=1, inplace=True)

    el_dataset = ElDataset(df=df, samples=samples, hist_hours=hist_hours, future_hours=pred_horizon)

    encoder = Encoder(1, 50, 1, hist_hours)

    dummy = el_dataset[0][0].unsqueeze(dim=0)
    encoder(dummy)
    print(el_dataset)

