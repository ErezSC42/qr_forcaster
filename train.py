import os
from pathlib import Path
import random
import nni
import numpy as np
import torch
import pickle
from argparse import Namespace
import pytorch_lightning as pl

from config import quantiles
from model import ForecasterQR
from arguments import get_params
from DatasetHandler import DatasetHandler


df_name = "raw_df"
NAME_EXP = f'{df_name}_nosplit'
#df_name = 'df_with_feat'
data_path = Path(f"/home/roxane/fintica/code/qr_forcaster/{df_name}")
TRAINED_MODEL_PATH = Path("trained_models")
DATALOADERS_PATH = Path("dataloaders")
LOSS_PATH = Path("lightning_losses")
SPLIT_DF = True

# TRAINED_MODEL_PATH = os.path.join("trained_models")
# DATALOADERS_PATH = os.path.join("dataloaders")

def set_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def debug_datasets(dataset, name, outdir):
    return

def main(args):
    set_seeds(args.seed)

    forking = args.use_forking_sequences
    forking_total_seq_length = 500 if forking else None
    data_handler = DatasetHandler(
        data_path,
        num_samples=args.dataset_num_samples,
        hist_days=args.max_sequence_len,
        pred_horizon=args.forcast_horizons,
        batch_size=args.batch_size,  # with forking, use lower batch size!
        forking_total_seq_length=forking_total_seq_length
    )
    df, dict_df_features = data_handler.load_df()    
    train_loader, val_loader = data_handler.load_dataset(df,dict_df_features, split_assets=SPLIT_DF)

    if args.debug:
        debug_datasets(df, "train", DATALOADERS_PATH)

    # save dataloaders for predictions
    os.makedirs(DATALOADERS_PATH, exist_ok=True)
    train_dl_path = os.path.join(DATALOADERS_PATH, "train_dl.pkl")
    test_dl_path = os.path.join(DATALOADERS_PATH, "test_dl.pkl")
    with open(train_dl_path, "wb") as fp:
        pickle.dump(train_loader, fp)
    with open(test_dl_path, "wb") as fp:
        pickle.dump(val_loader, fp)

    # quantiles = [.1, .2, .3, .4, .5, .6, .7, .8, .9, .95]
    # quantiles = [.2, .4, .5, .6, .8]
    
    futur_input_dim=len(train_loader.dataset.calendar_features)
    input_features_dim = len(list(train_loader.dataset.dict_features.values())[0].columns) if list(train_loader.dataset.dict_features.values())[0] is not None else 0
    data_dim = 1 + futur_input_dim + input_features_dim
    
    
    model = ForecasterQR(
        x_future_dim=futur_input_dim,
        data_dim=data_dim,
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
    NAME_ARG = f'his{args.max_sequence_len}_for{args.forcast_horizons}_h{args.encoder_hidden_dim}_d{args.decoder_context_dim}_sa{args.dataset_num_samples}_lr{args.learning_rate}_ba{args.batch_size}_ep{args.epochs}'
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(TRAINED_MODEL_PATH, f'{NAME_EXP}',f'{NAME_ARG}'),
        monitor="val_loss",
        filename="model-{epoch:02d}-{val_loss:.2f}",

    )
    logger = pl.loggers.tensorboard.TensorBoardLogger(save_dir = LOSS_PATH,name = NAME_EXP, \
                                                      version = NAME_ARG )
    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.epochs,
        callbacks=checkpoint_cb,
        num_sanity_val_steps=0,
        logger = logger)
    trainer.fit(model, train_loader, val_loader)
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
            namnespace_params = Namespace(**params)
        main(namespace_params)
    except Exception as ex:
        torch.cuda.empty_cache()
        raise
