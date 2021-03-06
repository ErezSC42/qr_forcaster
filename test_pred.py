import os
import pickle
import random
import numpy as np
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from DatasetHandler import DatasetHandler
from model import ForecasterQR

data_path = os.path.join("data", "LD2011_2014.txt")

TRAIN_DL_PATH = os.path.join("dataloaders", "train_dl.pkl")
TEST_DL_PATH = os.path.join("dataloaders", "test_dl.pkl")


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
        train_dl = pickle.load(fp)

    non_zero_indexes = [46369]
    train_dataset = train_dl.dataset
    index = random.randint(0, len(train_dataset))

    print(f"sampled index: {index}")

    data_sample = train_dataset[index]
    features = data_sample[0]
    future_series = data_sample[1]
    y_tensor = features[0].unsqueeze(0)
    x_tensor = features[1].unsqueeze(0)
    x_future_tensor = features[2].unsqueeze(0)

    res = model(y_tensor=y_tensor, x_tensor=x_tensor, x_future_tensor=x_future_tensor).squeeze()
    res = res.cpu().detach().numpy()
    print(res.shape)

    past_index = np.arange(0, 168)
    future_index = np.arange(168, 192)

    plt.figure()
    plt.plot(past_index ,y_tensor.squeeze(), label="past consumption")
    plt.plot(future_index, future_series, label="actual consumption")
    plt.plot(future_index, res[:, 5], label="model prediction")
    plt.legend()
    plt.tight_layout()
    plt.show()







