import pytorch_lightning as pl
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import ForecasterQR


def eval_model(model, dataset, forward_shift=2, quantiles=(0.05, 0.95), show_preds_every_x_days=12):
    model.eval()

    quantiles_indices = [list(model.quantiles).index(q) for q in quantiles]
    asset_names = dataset.raw_data.columns[:-3]
    
    with torch.no_grad():
        for asset_name in asset_names:
            if asset_name in dataset.calendar_features:
                continue
            last_pred_idx = len(dataset.raw_data.index) - dataset.full_length - forward_shift
            start_ts = range(forward_shift, last_pred_idx)

            preds = pd.DataFrame(
                index=dataset.raw_data.index[dataset.hist_days:], 
                columns=list(quantiles) + [asset_name]
            )
            
            seqs = []
            for sts in tqdm(start_ts):
                (y_tensor, x_calendar_past, x_features_past, x_calendar_future), y_future, _ = \
                    dataset.get(asset_name, sts)
                x_tensor = model.get_x_tensor(x_features_past, x_calendar_past)
                if x_tensor is not None:
                    x_tensor = x_tensor.unsqueeze(0)
                pred = model(
                    y_tensor.unsqueeze(0),
                    x_tensor, 
                    x_calendar_future.unsqueeze(0)
                )
                if sts % show_preds_every_x_days == 0:
                    p = pred[0, :, quantiles_indices].numpy()
                    ind = preds.index[sts-forward_shift:sts+p.shape[0]-forward_shift]
                    seqs.append((ind, p))
                pred = pred[0, forward_shift, quantiles_indices].numpy()
                actual = y_future[forward_shift].numpy()
                preds.iloc[sts:sts+1] = np.hstack([pred, actual])

            preds = preds.dropna().astype(float)

            fig = plt.figure()
            ax1 = fig.add_subplot(111)

            ax1.fill_between(
                preds.index, 
                preds.iloc[:,0].values, 
                preds.iloc[:,1].values, 
                alpha=0.2,
            )

            ax1.plot(preds.index, preds[asset_name].values, 
                label=asset_name, color='black')
            for (ind, p) in seqs:
                ax1.plot(ind, p[:, 0], color='orange', marker='x')
                ax1.plot(ind, p[:, 1], color='green', marker='x')
            
            calibration = {}
            for q in quantiles:
                if q >= 0.5 or (1 - q) not in preds.columns:
                    continue
                below = preds[asset_name] < preds[q]
                above = preds[asset_name] >= preds[1-q]
                cal = (0.5 * (below.sum() + above.sum()) / preds.shape[0]).round(3)
                calibration[q] = cal

                losses = []
                errors = (preds[asset_name] - preds[q])
                losses.append(np.maximum((q - 1) * errors, q * errors))
                errors = (preds[asset_name] - preds[1 - q]).values
                losses.append(np.maximum((q - 1) * errors, q * errors))
                total_loss = np.mean(losses).round(6)

                # need to convolve, otherwise it will not show the single values in fill_between
                for to_plot in (below, above):
                    where = np.convolve(to_plot, [1,1,1], mode='same').astype(bool)
                    ax1.fill_between(to_plot.index, 0,1, where=where, alpha=0.2, 
                        transform=ax1.get_xaxis_transform(), 
                        label=f'{q} outliers ({cal})')

            ax1.set_ylabel('quantiles ')
            plt.title(f'({cal}) / {total_loss}')
            plt.legend()
            plt.show()


if __name__ == '__main__':
    model_path = ""
    loader_path = ""
    model = ForecasterQR.load_from_checkpoint(model_path)

    with Path(loader_path).open("rb") as f:
        loader = pickle.load(f)    
    eval_model(model, loader.dataset)
