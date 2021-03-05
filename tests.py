import os
import pandas as pd
from model import Encoder, ForecasterQR
from MyDataset import ElDataset

DATA_PATH = os.path.join("data", "LD2011_2014.txt")

if __name__ == '__main__':
    num_samples = 2
    hist_hours = 168
    pred_horizon = 24
    # load data
    df = pd.read_csv(DATA_PATH,
                     parse_dates=[0],
                     delimiter=";",
                     decimal=",")
    df.rename({"Unnamed: 0": "timestamp"}, axis=1, inplace=True)

    el_dataset = ElDataset(df=df, num_samples=num_samples, hist_hours=hist_hours, future_hours=pred_horizon)

    quantiles = [.1, .2, .3, .4, .5, .6, .7, .8, .9, .95]

    model = ForecasterQR(
        x_dim=0,
        y_dim=1,
        input_max_squence_len=168,
        encoder_hidden_dim=8,
        encoder_num_layers=1,
        decoder_context_dim=4,
        quantiles=quantiles,
        horizons=24)

    # test data
    dummy = el_dataset[0][0].unsqueeze(dim=0)
    model(dummy)

    print(el_dataset)
