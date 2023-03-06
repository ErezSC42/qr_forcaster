import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class FinticaDataset(Dataset):
    """Fintica dataset."""

    def __init__(self, df, num_samples = None, hist_days=126, future_days=20, forking_total_seq_length=None, dict_features=None):
        """
        Args:
            df: original electricity data (see HW intro for details).
            samples (int): number of sample to take per asset.
        """
        self.raw_data = df.set_index("timestamp").astype(float)
        self.dict_features = dict_features
        self.num_samples = num_samples
        self.hist_days = hist_days
        self.future_days = future_days
        if forking_total_seq_length is None:
            self.full_length = hist_days + future_days
        else:
            self.full_length = forking_total_seq_length
        self.timestamps = range(len(self.raw_data.index) - self.full_length)
        self.num_samples = min(num_samples, len(self.timestamps)) if num_samples else len(self.timestamps)
        self.forking_total_seq_length = forking_total_seq_length
        self.sample()

    def __len__(self):
        return  self.num_samples * (self.raw_data.shape[1] - len(self.calendar_features))

    def get(self, asset, start_ts):
        if self.forking_total_seq_length is None:
            hist_start = start_ts

            # for days we need to be using indices addition 
            # because some days are missing
            hist_end = start_ts + self.hist_days
            future_start = hist_end + 1
            future_end = future_start + self.future_days
            hist_slice = self.raw_data.iloc[hist_start:hist_end]
            

            y_past = torch.Tensor(hist_slice[asset].values).unsqueeze(-1)
            
            # ----------ADDITIONAL FEATURES
            
            feat = self.dict_features.get(asset)
            x_features_df = feat.iloc[hist_start:hist_end] if feat else None
            x_features_past = torch.Tensor(x_features_df.values) if x_features_df is not None else torch.nan
            # ----------A

            x_calendar_past = torch.stack(
                [
                    torch.Tensor(hist_slice["yearly_cycle"].values),
                    torch.Tensor(hist_slice["monthly_cycle"].values),
                    torch.Tensor(hist_slice["weekly_cycle"].values),
                ],
                axis=-1
            )
            future_slice = self.raw_data.iloc[future_start:future_end]
            x_calendar_future = torch.stack(
                [
                    torch.Tensor(future_slice["yearly_cycle"].values),
                    torch.Tensor(future_slice["monthly_cycle"].values),
                    torch.Tensor(future_slice["weekly_cycle"].values),
                ],
                axis=-1
            )
            y = torch.Tensor(future_slice[asset].values)

        else:  # forking
            start = start_ts
            end = start_ts + pd.Timedelta(days=self.forking_total_seq_length - 1)
            unsliced_data = torch.stack(
                [
                    torch.Tensor(self.raw_data.iloc[start:end, asset].values),
                    torch.Tensor(self.raw_data.iloc[start:end, "yearly_cycle"].values),
                    torch.Tensor(self.raw_data.iloc[start:end, "monthly_cycle"].values),
                    torch.Tensor(self.raw_data.iloc[start:end, "weekly_cycle"].values),
                ],
                axis=-1
            )  # shape forking_total_seq_length,4
            tot_samples = self.forking_total_seq_length - (self.hist_days + self.future_days)
            data = torch.zeros([tot_samples, self.hist_days + self.future_days, unsliced_data.shape[1]])
            # mask = torch.ones([tot_samples]) #handles the cases of fct>horizon-(hist_days+future_days) can be also solved by masking
            for fct in range(tot_samples):
                # slice = unsliced_data[fct:fct + self.hist_days + self.future_days, :]
                data[fct, :, :] = unsliced_data[fct:fct + self.hist_days + self.future_days, :]

            # data = data[mask, :]
            y_past = data[:, :self.hist_days, 0].unsqueeze(-1)
            x_calendar_past = data[:, :self.hist_days, 1:]
            x_calendar_future = data[:, self.hist_days:, 1:]
            y = data[:, self.hist_days:, 0]
            x_features_past =None
        return (y_past, x_calendar_past, x_features_past, x_calendar_future), y, asset

    def __getitem__(self, idx):
        """Yield one sample, according to `self.get_mapping(idx)`."""
        asset, start_ts = self.mapping[idx]
        return self.get(asset, start_ts)
        
    # TODO add static feature? (house number embedding?)

    def get_mapping(self, idx):
        """Mapping between dataset index `idx` and actual `(asset, start_ts)` pair."""
        return self.mapping[idx]

    def sample(self):
        """
        Create sampling. Note that we shuffle `idx`, otherwise we would yield assets in batches,
        i.e., `self.samples` samples from `MT_001` first, then `self.samples` samples from `MT_002`, and so on.
        """
        self.mapping = {}

        timestamps = range(len(self.raw_data.index) - self.full_length)
        if self.num_samples is None:
            self.num_samples = len(timestamps)
        idx = np.arange(self.num_samples * self.raw_data.shape[1])
        np.random.shuffle(idx) 
        
        pairs = []
        for asset in self.raw_data.columns:
            start_ts_idx = np.random.choice(self.timestamps, replace=False, size=self.num_samples)
            pairs.extend([(asset, sts_index) for sts_index in start_ts_idx])

        self.mapping = {idx[i]: pairs[i] for i in range(len(idx))}
        self.non_calendar_features = list(self.raw_data.columns)
        self.create_calender_features()

    def create_calender_features(self):
        self.raw_data["yearly_cycle"] = np.sin(2 * np.pi * self.raw_data.index.dayofyear / 366)
        self.raw_data["monthly_cycle"] = np.sin(2 * np.pi * self.raw_data.index.isocalendar().week % 4 / 3)
        self.raw_data["weekly_cycle"] = np.sin(2 * np.pi * self.raw_data.index.dayofweek / 7)
        self.calendar_features = ["yearly_cycle", "monthly_cycle", "weekly_cycle"]
