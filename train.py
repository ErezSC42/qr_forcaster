import os
import pickle
import pytorch_lightning as pl

from DatasetHandler import DatasetHandler
from model import ForecasterQR

data_path = os.path.join("data", "LD2011_2014.txt")
TRAINED_MODEL_PATH = os.path.join("trained_models")
DATALOADERS_PATH = os.path.join("dataloaders")

if __name__ == '__main__':
    train_el_dataloader, val_el_dataloader = DatasetHandler(
        data_path,
        num_samples=400,
        hist_hours=168,
        pred_horizon=24,
        batch_size=256).load_dataset()

    # save dataloaders for predictions
    train_dl_path = os.path.join(DATALOADERS_PATH, "train_dl.pkl")
    test_dl_path = os.path.join(DATALOADERS_PATH, "test_dl.pkl")
    with open(train_dl_path, "wb") as fp:
        pickle.dump(train_el_dataloader, fp)
    with open(test_dl_path, "wb") as fp:
        pickle.dump(train_el_dataloader, fp)

    quantiles = [.1, .2, .3, .4, .5, .6, .7, .8, .9, .95]
    model = ForecasterQR(
        x_dim=3,
        y_dim=4,
        input_max_squence_len=168,
        encoder_hidden_dim=128,
        encoder_num_layers=1,
        decoder_context_dim=64,
        quantiles=quantiles,
        horizons=24,
        device="gpu")

    # model checkpoint callback
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=TRAINED_MODEL_PATH,
        monitor="val_loss",
        filename="model-{epoch:02d}-{val_loss:.2f}"
    )

    trainer = pl.Trainer(gpus=1, checkpoint_callback=checkpoint_cb)
    trainer.fit(model, train_el_dataloader, val_el_dataloader)

# TODO config file? num_gpus, data_path, output_path(logs+visualization) etc
