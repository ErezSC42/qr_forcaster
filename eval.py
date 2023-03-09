import os
from pathlib import Path
import nni
import numpy as np
import pandas as pd
import torch
import pickle
from argparse import Namespace
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import ForecasterQR
from arguments import get_params
from DatasetHandler import DatasetHandler


TRAINED_MODEL_PATH = os.path.join("trained_models")
# DATALOADERS_PATH = "/home/villqrd/repos/mqrnn/dataloaders"
DATALOADERS_PATH = "/home/villqrd/repos/mqrnn/dataloaders"

def build_low_bound_strategy(asset, low_bound):
    strat = asset * 0 + 1
    strat = pd.concat([strat, asset, low_bound], axis=1).dropna()
    strat.iloc[strat.iloc[:, 1] < strat.iloc[:, 2], 0] = 0
    return strat.iloc[:, 0].rename("REGIMES_1").astype(int), strat.iloc[:, 1]

def main(args):
    model = ForecasterQR.load_from_checkpoint("/home/villqrd/repos/mqrnn/trained_models/raw_df_nosplit/his252_for10_h32_d32_sa500_lr0.05_ba512_ep2000/model-epoch=301-val_loss=0.07.ckpt")
    
    loaders = []
    for name in ("train", "test"):
        loader_path = Path(DATALOADERS_PATH) / f"{name}_dl.pkl"
        with loader_path.open("rb") as f:
            loader = pickle.load(f)    
            loaders.append(loader)
    
    
    forward_shift = 2
    model.eval()

    names = ("train", "val")
    qt_idx = [1, -2]
    quantiles = np.array(model.quantiles)[qt_idx]
    with torch.no_grad():
        diffs = {}
        asset_names = ["XLI_d0.2", "goog_d0.25", "ftse100_d0.1"]
        for asset_name in asset_names:
            for name, loader in zip(names, loaders):
                if name == "train": 
                    continue
            
                dataset = loader.dataset
                if asset_name in dataset.calendar_features:
                    continue
                last_pred_idx = len(dataset.raw_data.index) - dataset.full_length - forward_shift
                # start_ts = range(0, last_pred_idx, dataset.future_days - forward_shift)
                start_ts = range(0, last_pred_idx)

                # columns = [f'q_{q}' for q in model.quantiles]
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

                    pred = pred[0, forward_shift, qt_idx].numpy()
                    actual = y_future[forward_shift].numpy()
                    preds.iloc[sts:sts+1] = np.hstack([pred, actual])

                preds = preds.dropna().astype(float)
                
                roll_vol = preds[asset_name].pct_change().rolling(20).std()*(252**0.5) 
                qt_diff = (preds.iloc[:,1] - preds.iloc[:,0])
                # dataset.raw_data[asset_name].plot()
                
                # plt.plot(preds.index, preds.iloc[:,1].values - preds.iloc[:,0].values)
                fig = plt.figure()
                ax1 = fig.add_subplot(111)
                ax2 = ax1.twinx()

                ax1.fill_between(
                    preds.index, 
                    preds.iloc[:,0].values, 
                    preds.iloc[:,1].values, 
                    alpha=0.2,
                )


                # asset = dataset.raw_data[asset_name].shift(-forward_shift)
                # ax1.plot(asset.index, asset.values, label=asset_name, color='black')
                ax1.plot(preds.index, preds[asset_name].values, label=asset_name, color='black')
                
                calibration = {}
                for q in quantiles:
                    below = preds[asset_name] < preds[q]
                    cal = (below.sum() / preds.shape[0]).round(2)
                    calibration[q] = cal

                    to_plot = below
                    if q == 0.95:
                        to_plot = ~below
                    where = np.convolve(to_plot, [1,1,1], mode='same').astype(bool)
                    ax1.fill_between(to_plot.index, 0,1, where=where, alpha=0.2, 
                        transform=ax1.get_xaxis_transform(), 
                        label=f'{q} outliers ({cal})')

                ax1.set_ylabel('quantiles')
                ax2.set_ylabel('sharpness (qt diff)')
                
                # for col in colufmns:
                #     plt.plot(preds.index, preds[col].values, label=col)
                plt.legend()
                plt.savefig('indiv.jpg')
                plt.show()

                diffs[name] = qt_diff
                fig = plt.figure()
                ax1 = fig.add_subplot(111)
                ax2 = ax1.twinx()

                ax1.plot(preds.iloc[:,0].index, preds.iloc[:,0].values, label='loqt')
                ax1.plot(preds.iloc[:,1].index, preds.iloc[:,1].values, label='hiqt')
                ax1.legend()
                ax2.legend()
                plt.show()
                # plt.savefig('indiv.jpg')

                # qt = hist_diff.expanding().quantile(0.9)
                qt = qt_diff.expanding().quantile(0.9)
                diff_strat = pd.concat([qt_diff, qt], axis=1).dropna()
                diff_strat = (diff_strat.iloc[:, 0] < diff_strat.iloc[:, 1]).astype(int)

                # plt.hist(strat.values, bins=60)
                # plt.show()
                
                # strat, asset = build_low_bound_strategy(dataset.raw_data[asset_name], low_bound)
                # diff_strat = diff_strat.shift(forward_shift)

                returns = dataset.raw_data[asset_name].pct_change()[diff_strat.index].dropna()
                bnh = (1 + returns).cumprod()

                fig, ax1 = plt.subplots()

                ax2 = ax1.twinx()
                ax1.plot(bnh.index, bnh, label=f'{asset_name}_bnh')

                eqty_curve = (1 + diff_strat * returns).cumprod()
                ax1.plot(eqty_curve.index, eqty_curve.values, label='strat')
                ax2.plot(qt_diff.index, qt_diff.values, label='qt diff', color='gray')
                ax1.set_xlabel("time")
                ax1.set_ylabel('eqty curve')
                ax2.set_ylabel('qt diff')
                plt.legend()
                plt.title(f"{asset_name} {name.upper()} qt diff (0.9 - 0.1)")
                plt.savefig(f"/home/villqrd/Downloads/{asset_name}_{name}.jpg")
                plt.close()

            for name, data in diffs.items():
                plt.hist(data, bins=50, alpha=0.2, label=name, density=True)
            plt.legend()
            plt.savefig(f"/home/villqrd/Downloads/{asset_name}_{name}_hist.jpg")
            plt.close() 



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
