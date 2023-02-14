from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from FinticaDataset import FinticaDataset


class DatasetHandler:
    def __init__(self, data_path, num_samples, hist_days, pred_horizon, batch_size, val_split_ratio=0.2,
                 forking_total_seq_length=None):
        self.data_path = Path(data_path)
        self.num_samples = num_samples
        self.hist_days = hist_days
        self.pred_horizon = pred_horizon
        self.batch_size = batch_size
        self.val_split_ratio = val_split_ratio
        self.forking_total_seq_length = forking_total_seq_length
        if forking_total_seq_length is not None:
            assert (self.forking_total_seq_length > self.hist_days + pred_horizon)

    def load_dataset(self, df: pd.DataFrame = None, split: bool = True, num_workers=0):
        if df is None:
            assets = []
            for p in self.data_path.iterdir():
                asset = pd.read_csv(p, parse_dates=True, index_col=0)
                asset_name = asset.columns[0]
                if 'd0.' in asset_name:
                    assets.append(asset)    
                else:
                    print(f'skipping {asset_name}')
            df = pd.concat(assets, axis=1)
            df = df.reset_index()
            df = df.ffill()
            df = df.dropna()
            df.rename({"Date": "timestamp"}, axis=1, inplace=True)
        if split:
            df_train, df_val = self.split_df(df, True)
            train_dataset = FinticaDataset(df=df_train, num_samples=self.num_samples, hist_days=self.hist_days,
                                      future_days=self.pred_horizon,
                                      forking_total_seq_length=self.forking_total_seq_length)
            val_dataset = FinticaDataset(df=df_val, num_samples=self.num_samples, hist_days=self.hist_days,
                                    future_days=self.pred_horizon)
            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=num_workers)
            val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=num_workers)
            return train_dataloader, val_dataloader

        dataset = FinticaDataset(df=df, num_samples=self.num_samples, hist_days=self.hist_days,
                            future_days=self.pred_horizon,
                            forking_total_seq_length=self.forking_total_seq_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=num_workers)
        return dataloader

    def split_df(self, df, split_assets: bool):
        num_rows, num_cols = df.shape
        if split_assets:
            print("splitting assets!")
            assets = df.columns[1:]
            train_assets, test_assets = train_test_split(assets, test_size=0.5)
            assert not set(train_assets).intersection(set(test_assets)) # make sure no leakage
            train_assets = train_assets.tolist()
            test_assets = test_assets.tolist()
            train_assets.insert(0, "timestamp")
            test_assets.insert(0, "timestamp")
            df_train = df[train_assets]
            df_test = df[test_assets]
        else:
            df_train = df.copy()
            df_test = df.copy()
        train_size = int(num_rows * self.val_split_ratio)
        df_train = df_train[:train_size]
        df_val = df_test[train_size:]
        return df_train, df_val
