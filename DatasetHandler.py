import pandas as pd
from torch.utils.data import DataLoader

from qr_forcaster.MyDataset import ElDataset


class DatasetHandler:
    def __init__(self, data_path, num_samples, hist_hours, pred_horizon, batch_size, val_split_ratio=0.2, device="cpu"):
        self.data_path = data_path
        self.num_samples = num_samples
        self.hist_hours = hist_hours
        self.pred_horizon = pred_horizon
        self.batch_size = batch_size
        self.val_split_ratio = val_split_ratio
        self.device = device

    def load_dataset(self):
        df = pd.read_csv(self.data_path,
                         parse_dates=[0],
                         delimiter=";",
                         decimal=",")
        df.rename({"Unnamed: 0": "timestamp"}, axis=1, inplace=True)
        df_train, df_val = self.split_df(df)
        train_dataset = ElDataset(df=df_train, num_samples=self.num_samples, hist_hours=self.hist_hours, future_hours=self.pred_horizon, device=self.device)
        val_dataset = ElDataset(df=df_val, num_samples=self.num_samples, hist_hours=self.hist_hours, future_hours=self.pred_horizon, device=self.device)
        train_dataloader, val_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=4), DataLoader(val_dataset, batch_size=self.batch_size, num_workers=4)
        return train_dataloader, val_dataloader

    def split_df(self, df):
        num_rows = df.shape[0]
        train_size = int(num_rows * self.val_split_ratio)
        df_train = df[:train_size]
        df_val = df[train_size:]
        return df_train, df_val
