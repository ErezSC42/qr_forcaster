import os
import nni
import torch
import pickle
from argparse import Namespace
import pytorch_lightning as pl

from config import quantiles
from model import ForecasterQR
from arguments import get_params
from DatasetHandler import DatasetHandler


data_path = os.path.join("data", "LD2011_2014.txt")
TRAINED_MODEL_PATH = os.path.join("trained_models")
DATALOADERS_PATH = os.path.join("dataloaders")


def main(args):
    forking = args.use_forking_sequences
    forking_total_seq_length = 500 if forking else None
    train_el_dataloader, val_el_dataloader = DatasetHandler(
        data_path,
        num_samples=args.dataset_num_samples,
        hist_hours=args.max_sequence_len,
        pred_horizon=args.forcast_horizons,
        batch_size=args.batch_size,  # with forking, use lower batch size!
        forking_total_seq_length=forking_total_seq_length).load_dataset()

    # save dataloaders for predictions
    os.makedirs(DATALOADERS_PATH, exist_ok=True)
    train_dl_path = os.path.join(DATALOADERS_PATH, "train_dl.pkl")
    test_dl_path = os.path.join(DATALOADERS_PATH, "test_dl.pkl")
    with open(train_dl_path, "wb") as fp:
        pickle.dump(train_el_dataloader, fp)
    with open(test_dl_path, "wb") as fp:
        pickle.dump(val_el_dataloader, fp)

    # quantiles = [.1, .2, .3, .4, .5, .6, .7, .8, .9, .95]
    # quantiles = [.2, .4, .5, .6, .8]

    model = ForecasterQR(
        x_dim=3,
        y_dim=4,
        input_max_squence_len=args.max_sequence_len,
        encoder_hidden_dim=args.encoder_hidden_dim,
        encoder_num_layers=args.encoder_layer_count,
        decoder_context_dim=args.decoder_context_dim,
        quantiles=quantiles,
        horizons=args.forcast_horizons,
        device="gpu",
        init_learning_rate=args.learning_rate,
        init_weight_decay=args.weight_decay,
        sequence_forking=forking is not None
    )

    # model checkpoint callback
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=TRAINED_MODEL_PATH,
        monitor="val_loss",
        filename="model-{epoch:02d}-{val_loss:.2f}"
    )

    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.epochs,
        checkpoint_callback=checkpoint_cb,
        num_sanity_val_steps=0)

    trainer.fit(model, train_el_dataloader, val_el_dataloader)
    val_loss = trainer.callback_metrics["val_loss"].item()
    nni.report_final_result({"default": val_loss})


if __name__ == '__main__':
    try:
        # get parameters from tuner
        namespace_params = get_params()
        if namespace_params.use_nni:
            print("nni activated.")
            tuner_params = nni.get_next_parameter()
            params = vars(namespace_params)
            print("TUNER PARAMS: " + str(tuner_params))
            print("params:" + str(params))
            params.update(tuner_params)
            namespace_params = Namespace(**params)
        main(namespace_params)
    except Exception as ex:
        torch.cuda.empty_cache()
        raise
