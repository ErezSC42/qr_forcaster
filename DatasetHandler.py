import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from ElDataset import ElDataset


class DatasetHandler:
    def __init__(self, data_path, num_samples, hist_hours, pred_horizon, batch_size, val_split_ratio=0.2,
                 forking_total_seq_length=None):
        self.data_path = data_path
        self.num_samples = num_samples
        self.hist_hours = hist_hours
        self.pred_horizon = pred_horizon
        self.batch_size = batch_size
        self.val_split_ratio = val_split_ratio
        self.forking_total_seq_length = forking_total_seq_length
        if forking_total_seq_length is not None:
            assert (self.forking_total_seq_length > self.hist_hours + pred_horizon)

    def load_dataset(self, df: pd.DataFrame = None, split: bool = True):
        if df is None:
            df = pd.read_csv(self.data_path,
                             parse_dates=[0],
                             delimiter=";",
                             decimal=",")
            df.rename({"Unnamed: 0": "timestamp"}, axis=1, inplace=True)
        if split:
            df_train, df_val = self.split_df(df, True)
            train_dataset = ElDataset(df=df_train, num_samples=self.num_samples, hist_hours=self.hist_hours,
                                      future_hours=self.pred_horizon,
                                      forking_total_seq_length=self.forking_total_seq_length)
            val_dataset = ElDataset(df=df_val, num_samples=self.num_samples, hist_hours=self.hist_hours,
                                    future_hours=self.pred_horizon)
            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=4)
            val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=4)
            return train_dataloader, val_dataloader

        dataset = ElDataset(df=df, num_samples=self.num_samples, hist_hours=self.hist_hours,
                            future_hours=self.pred_horizon,
                            forking_total_seq_length=self.forking_total_seq_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=4)
        return dataloader

    def split_df(self, df, split_households: bool):
        num_rows, num_cols = df.shape
        if split_households:
            print("splitting households!")
            household_list = df.columns[1:]
            train_households, test_households = train_test_split(household_list, test_size=0.5)
            assert set(train_households).intersection(set(test_households)) == set() # make sure no leakage
            train_households = train_households.tolist()
            test_households = test_households.tolist()
            train_households.insert(0, "timestamp")
            test_households.insert(0, "timestamp")
            df_train = df[train_households]
            df_test = df[test_households]
        else:
            df_train = df.copy()
            df_test = df.copy()
        train_size = int(num_rows * self.val_split_ratio)
        df_train = df_train[:train_size]
        df_val = df_test[train_size:]
        return df_train, df_val
