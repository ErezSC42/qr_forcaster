import os
import pickle
import random
import datetime
import torch
import pandas as pd
from config import prod_model_path
from model import ForecasterQR
import matplotlib.pyplot as plt

data_path = os.path.join("data", "LD2011_2014.txt")

TRAIN_DL_PATH = os.path.join("dataloaders", "train_dl.pkl")
TEST_DL_PATH = os.path.join("dataloaders", "test_dl.pkl")
SAVE_FIG = os.path.join("figures")


def predict(model, dataset, index):
    asset, start = dataset.get_mapping(index)
    data_sample = dataset[index]
    features = data_sample[0]
    future_series = data_sample[1]
    past_series = features[0].unsqueeze(0)
    x_tensor = features[1].unsqueeze(0)
    x_features_past = features[2].unsqueeze(0) if not isinstance(features[2] ,float) else None
    x_future_tensor = features[3].unsqueeze(0)
    x_tensor = torch.cat((x_tensor,x_features_past),dim=-1) if x_features_past is not None else x_tensor 
    res = model(y_tensor=past_series, x_tensor=x_tensor,x_future_tensor=x_future_tensor).squeeze()
    return asset, start, past_series, future_series, res


def plot_prediction(asset, preds, y_past, y_future, save_dir=None, index=None):
    fig = plt.figure()
    plt.title(f"prediction for asset {asset}")
    plt.plot(y_past.index, y_past.values, label="past value")
    plt.plot(y_future.index, y_future.values, label="actual value")
    for i, q in enumerate(preds.columns):
        plt.plot(future_ts, preds.loc[:, q], label=f"q={q}")
    half = (preds.shape[1] - 1) // 2
    for i in range(half):
        alph = 0.0 + 2 * (i / len(model.quantiles))
        plt.fill_between(preds.index, preds.iloc[:, i], preds.iloc[:, -(i + 1)], color="g", alpha=alph)
    fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
    plt.legend(loc="upper left")
    plt.xlabel("time")
    plt.ylabel("asset")
    plt.grid()
    plt.tight_layout()
    os.makedirs(os.path.join(save_dir, asset), exist_ok=True)
    plt.savefig(os.path.join(save_dir, asset, f'{index}.png'))
    plt.close()
   # plt.show()


if __name__ == '__main__':
    # trained_model_list = os.listdir("trained_models")
    # best_model_index = np.argmin([int(name.split("=")[2].split(".")[0]) for name in trained_model_list])
    # TRAINED_MODEL_NAME = os.listdir("trained_models")[best_model_index]
    # TRAINED_MODEL_PATH = os.path.join("trained_models", TRAINED_MODEL_NAME)

    # TRAINED_MODEL_PATH = os.path.join("trained_models/model-epoch=49-val_loss=410.90.ckpt")
    # TRAINED_MODEL_PATH = os.path.join("trained_models", "model-epoch=04-val_loss=709.13.ckpt")
  #  path1 ='/home/roxane/fintica/code/qr_forcaster/trained_models/df_with_feat/his126_for5_h32_d32_sa500_lr0.05_ba512_ep300/model-epoch=90-val_loss=0.18.ckpt'
    path2 = '/home/roxane/fintica/code/qr_forcaster/trained_models/raw_df/his126_for5_h32_d32_sa500_lr0.05_ba512_ep300/model-epoch=90-val_loss=0.03.ckpt'
    path3='/home/roxane/fintica/code/qr_forcaster/trained_models/raw_df_nosplit/his126_for5_h32_d32_sa500_lr0.05_ba512_ep300/model-epoch=265-val_loss=0.02.ckpt'
    for name_path, prod_model_path in {'split':path2,'no_split':path3}.items():
        TRAINED_MODEL_PATH = prod_model_path
        model = ForecasterQR.load_from_checkpoint(TRAINED_MODEL_PATH)
        model.eval()

        print(f"loaded model: {TRAINED_MODEL_PATH}")

        # test data
        with open(TEST_DL_PATH, "rb") as fp:
            pred_dl = pickle.load(fp)

        pred_dataset = pred_dl.dataset

        hist_quantiles = {q: pred_dataset.raw_data.expanding().quantile(q) for q in model.quantiles}


        asset_name = 'cac40_d0.1'
        ordered_mapping = sorted([(ts, idx) for idx, (a, ts) in pred_dataset.mapping.items() if a == asset_name])
        ordered_mapping = [m[1] for m in ordered_mapping]
        for index in ordered_mapping:
            # index = random.randint(0, len(pred_dataset))
            # print(f"sampled index: {index}")

            asset, hist_start, past_series, future_series, preds = predict(model, 
                dataset=pred_dataset, 
                index=index
            )
            hist_end = hist_start + pred_dataset.hist_days
            future_start = hist_end + 1
            future_end = future_start + pred_dataset.future_days
            hist_ts = pred_dataset.raw_data.index[hist_start:hist_end]
            future_ts = pred_dataset.raw_data.index[future_start:future_end]

            past_df = pd.DataFrame(index=hist_ts, columns=[asset], data=past_series.detach().numpy().squeeze())
            future_df = pd.DataFrame(index=future_ts, columns=[asset], data=future_series.detach().numpy().squeeze())
            preds_df = pd.DataFrame(index=future_ts, columns=map(str, model.quantiles), data=preds.detach().numpy().squeeze())

            plot_prediction(asset, preds_df, y_past=past_df, y_future=future_df, save_dir =os.path.join(SAVE_FIG, name_path), index=index )

            
