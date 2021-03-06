import os
import pickle
import random
import datetime
import numpy as np
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from DatasetHandler import DatasetHandler
from model import ForecasterQR

data_path = os.path.join("data", "LD2011_2014.txt")

TRAIN_DL_PATH = os.path.join("dataloaders", "train_dl.pkl")
TEST_DL_PATH = os.path.join("dataloaders", "test_dl.pkl")


def predict(model, dataset, index):
    household, start_timestamp = dataset.get_mapping(index)
    data_sample = dataset[index]
    features = data_sample[0]
    future_series = data_sample[1]
    past_series = features[0].unsqueeze(0)
    x_tensor = features[1].unsqueeze(0)
    x_future_tensor = features[2].unsqueeze(0)
    res = model(y_tensor=past_series, x_tensor=x_tensor, x_future_tensor=x_future_tensor).squeeze()
    return household, start_timestamp, past_series, future_series, res


def plot_prediction(start_ts, household, model_output, y_past, y_future):
    res = model_output.cpu().detach().numpy()
    past_ts_index = [start_ts + datetime.timedelta(hours=x) for x in range(168)]
    future_ts_index = [start_ts + datetime.timedelta(hours=(168 + x)) for x in range(24)]

    quantiles_num = len(model.quantiles)
    half = (quantiles_num - 1) // 2
    fig = plt.figure()
    plt.title(f"Consumption prediction for household {household}")
    plt.plot(past_ts_index, y_past.squeeze(), label="past consumption")
    plt.plot(future_ts_index, y_future, label="actual consumption")
    plt.plot(future_ts_index, res[:, 5], label="median prediction")
    for i in range(half):
        alph = 0.05 + 0.5 * (i / len(model.quantiles))
        plt.fill_between(future_ts_index, res[:, i], res[:, -(i + 1)],
                         color="g", alpha=alph)
    fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    trained_model_list = os.listdir("trained_models")
    best_model_index = np.argmin([int(name.split("=")[2].split(".")[0]) for name in trained_model_list])
    TRAINED_MODEL_NAME = os.listdir("trained_models")[best_model_index]
    TRAINED_MODEL_PATH = os.path.join("trained_models", TRAINED_MODEL_NAME)

    model = ForecasterQR.load_from_checkpoint(TRAINED_MODEL_PATH)
    model.eval()

    print(f"loaded model: {TRAINED_MODEL_PATH}")

    # test data
    with open(TEST_DL_PATH, "rb") as fp:
        pred_dl = pickle.load(fp)

    pred_dataset = pred_dl.dataset
    index = random.randint(0, len(pred_dataset))
    print(f"sampled index: {index}")

    household, start_timestamp, past_series, future_series, res = predict(model, dataset=pred_dataset, index=index)
    plot_prediction(start_timestamp, household, res, y_past=past_series, y_future=future_series)
