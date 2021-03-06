import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class ElDataset(Dataset):
    """Electricity dataset."""

    def __init__(self, df, num_samples, hist_hours=168, future_hours=24):
        """
        Args:
            df: original electricity data (see HW intro for details).
            samples (int): number of sample to take per household.
        """
        self.raw_data = self.el_resample(df).set_index("timestamp")
        self.num_samples = num_samples
        self.hist_hours = hist_hours
        self.future_hours = future_hours
        self.full_length = pd.Timedelta(hours=(hist_hours + future_hours))
        self.sample()

    def __len__(self):
        return self.num_samples * (self.raw_data.shape[1]-len(self.calendar_features))

    def __getitem__(self, idx):
        """Yield one sample, according to `self.get_mapping(idx)`."""

        household, start_ts = self.mapping[idx]

        hist_start = start_ts
        hist_end = start_ts + pd.Timedelta(hours=self.hist_hours - 1)
        future_start = hist_end + pd.Timedelta(hours=1)
        future_end = hist_end + pd.Timedelta(hours=self.future_hours)

        x_data = torch.Tensor(self.raw_data.loc[hist_start:hist_end, household].values).unsqueeze(-1)
        x_calendar_past = torch.stack(
            [
                torch.Tensor(self.raw_data.loc[hist_start:hist_end, "yearly_cycle"].values),
                torch.Tensor(self.raw_data.loc[hist_start:hist_end, "weekly_cycle"].values),
                torch.Tensor(self.raw_data.loc[hist_start:hist_end, "daily_cycle"].values),
            ],
            axis=-1
        )
        x_calendar_future = torch.stack(
            [
                torch.Tensor(self.raw_data.loc[future_start:future_end, "yearly_cycle"].values),
                torch.Tensor(self.raw_data.loc[future_start:future_end, "weekly_cycle"].values),
                torch.Tensor(self.raw_data.loc[future_start:future_end, "daily_cycle"].values),
            ],
            axis=-1
        )
        y = torch.Tensor(self.raw_data.loc[future_start:future_end, household].values)

        return (x_data, x_calendar_past, x_calendar_future), y

    # TODO add static feature? (house number embedding?)

    def get_mapping(self, idx):
        """Mapping between dataset index `idx` and actual `(household, start_ts)` pair."""
        return self.mapping[idx]

    def sample(self):
        """
        Create sampling. Note that we shuffle `idx`, otherwise we would yield households in batches,
        i.e., `self.samples` samples from `MT_001` first, then `self.samples` samples from `MT_002`, and so on.
        """

        self.mapping = {}
        timestamps = self.raw_data[:(self.raw_data.index.max() - self.full_length)].index.to_series()

        idx = np.arange(self.num_samples * self.raw_data.shape[1])
        np.random.shuffle(idx)

        pairs = []

        for household in self.raw_data.columns:
            start_ts = timestamps.sample(self.num_samples)
            pairs.extend([(household, sts) for sts in start_ts])

        self.mapping = {idx[i]: pairs[i] for i in range(len(idx))}

        self.create_calender_features()

    def create_calender_features(self):
        self.raw_data["yearly_cycle"] = np.sin(2 * np.pi * self.raw_data.index.dayofyear / 366)
        self.raw_data["weekly_cycle"] = np.sin(2 * np.pi * self.raw_data.index.dayofweek / 7)
        self.raw_data["daily_cycle"] = np.sin(2 * np.pi * self.raw_data.index.hour / 24)
        self.calendar_features = ["yearly_cycle", "weekly_cycle", "daily_cycle"]


    def el_resample(self, df):
        return df.resample("1H", on="timestamp").mean().reset_index()
