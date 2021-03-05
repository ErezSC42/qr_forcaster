import os

import pytorch_lightning as pl
import torch
from torch import nn

from DatasetHandler import DatasetHandler
from model import ForecasterQR

data_path = os.path.join("data", "LD2011_2014.txt")

if __name__ == '__main__':
    train_el_dataloader, val_el_dataloader = DatasetHandler(
        data_path,
        num_samples=100,
        hist_hours=168,
        pred_horizon=24,
        batch_size=128).load_dataset()
    quantiles = [.1, .2, .3, .4, .5, .6, .7, .8, .9, .95]
    model = ForecasterQR(
        x_dim=0,
        y_dim=1,
        input_max_squence_len=168,
        encoder_hidden_dim=80,
        encoder_num_layers=1,
        decoder_context_dim=40,
        quantiles=quantiles,
        horizons=24,
        device="gpu")

    trainer = pl.Trainer(gpus=1)
    trainer.fit(model, train_el_dataloader, val_el_dataloader)

# TODO config file? num_gpus, data_path, output(logs+visualization) path etc
