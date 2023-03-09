import os
from pathlib import Path
import nni
import numpy as np
import pandas as pd
import torch
import pickle
from argparse import Namespace
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import ForecasterQR
from arguments import get_params


TRAINED_MODEL_PATH = os.path.join("trained_models")
# DATALOADERS_PATH = "/home/villqrd/repos/mqrnn/dataloaders"
DATALOADERS_PATH = "/home/villqrd/repos/mqrnn/dataloaders"

def build_low_bound_strategy(asset, low_bound):
    strat = asset * 0 + 1
    strat = pd.concat([strat, asset, low_bound], axis=1).dropna()
    strat.iloc[strat.iloc[:, 1] < strat.iloc[:, 2], 0] = 0
    return strat.iloc[:, 0].rename("REGIMES_1").astype(int), strat.iloc[:, 1]

def eval_modal(model, loader, forward_shift=2, quantiles=(0.05, 0.95)):
    model.eval()

    quantiles_indices = [list(model.quantiles).index(q) for q in quantiles]
    asset_names = ["XLI_d0.2", "goog_d0.25", "ftse100_d0.1"]
    
    with torch.no_grad():
        dataset = loader.dataset
        for asset_name in asset_names:
            if asset_name in dataset.calendar_features:
                continue
            last_pred_idx = len(dataset.raw_data.index) - dataset.full_length - forward_shift
            start_ts = range(0, last_pred_idx)

            preds = pd.DataFrame(
                index=dataset.raw_data.index[dataset.hist_days:], 
                columns=list(quantiles) + [asset_name]
            )
            
            for sts in tqdm(start_ts):
                (y_tensor, x_calendar_past, x_features_past, x_calendar_future), y_future, _ = \
                    dataset.get(asset_name, sts)
                x_tensor = model.get_x_tensor(x_features_past, x_calendar_past)
                pred = model(
                    y_tensor.unsqueeze(0),
                    x_tensor.unsqueeze(0), 
                    x_calendar_future.unsqueeze(0)
                )
                # hist_dates, future_dates = dataset.get_dates(sts)

                pred = pred[0, forward_shift, quantiles_indices].numpy()
                actual = y_future[forward_shift].numpy()
                preds.iloc[sts:sts+1] = np.hstack([pred, actual])

            preds = preds.dropna().astype(float)

            qt_diff = (preds.iloc[:,1] - preds.iloc[:,0])

            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax2 = ax1.twinx()

            ax1.fill_between(
                preds.index, 
                preds.iloc[:,0].values, 
                preds.iloc[:,1].values, 
                alpha=0.2,
            )

            ax1.plot(preds.index, preds[asset_name].values, 
                label=asset_name, color='black')
            
            calibration = {}
            for q in quantiles:
                below = preds[asset_name] < preds[q]
                cal = (below.sum() / preds.shape[0]).round(2)
                calibration[q] = cal

                to_plot = below
                if q > 0.5:
                    to_plot = ~below
                # need to convolve, otherwise it will not show the single values in fill_between
                where = np.convolve(to_plot, [1,1,1], mode='same').astype(bool)
                ax1.fill_between(to_plot.index, 0,1, where=where, alpha=0.2, 
                    transform=ax1.get_xaxis_transform(), 
                    label=f'{q} outliers ({cal})')

            ax2.plot(qt_diff.index, qt_diff.values, label='qt diff', color='blue')
            ax1.set_ylabel('quantiles')
            ax2.set_ylabel('sharpness (qt diff)')

            plt.legend()
            plt.savefig('indiv.jpg')
            plt.show()


if __name__ == '__main__':
    model_path = ""
    model = ForecasterQR.load_from_checkpoint(model_path)
    
    loaders = []
    name = "test"
    loader_path = Path(DATALOADERS_PATH) / f"{name}_dl.pkl"
    with loader_path.open("rb") as f:
        loader = pickle.load(f)    
        loaders.append(loader)

    eval_modal(model, loader)
