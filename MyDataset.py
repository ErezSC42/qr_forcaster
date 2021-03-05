import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def el_resample(df):
    return df.resample("1H", on="timestamp").mean().reset_index()


class ElDataset(Dataset):
    """Electricity dataset."""

    # TODO add calendaric features

    def __init__(self, df, num_samples, hist_hours=168, future_hours=24):
        """
        Args:
            df: original electricity data (see HW intro for details).
            samples (int): number of sample to take per household.
        """
        self.raw_data = el_resample(df).set_index("timestamp")
        self.num_samples = num_samples
        self.hist_hours = hist_hours
        self.future_hours = future_hours
        self.full_length = pd.Timedelta(hours=(hist_hours + future_hours))
        self.sample()

    def __len__(self):
        return self.num_samples * self.raw_data.shape[1]

    def __getitem__(self, idx):
        """Yield one sample, according to `self.get_mapping(idx)`."""

        household, start_ts = self.mapping[idx]

        hist_start = start_ts
        hist_end = start_ts + pd.Timedelta(hours=self.hist_hours - 1)
        future_start = hist_end + pd.Timedelta(hours=1)
        future_end = hist_end + pd.Timedelta(hours=self.future_hours)

        x = torch.Tensor(self.raw_data.loc[hist_start:hist_end, household].values)
        y = torch.Tensor(self.raw_data.loc[future_start:future_end, household].values)
        return x, y

    # TODO add static,seasunal features

    def get_mapping(self, idx):
        """Mapping between dataset index `idx` and actual `(household, start_ts)` pair."""
        return self.mapping[idx]

    def sample(self):
        """
        Create sampling. Note that we shuffle `idx`, otherwise we would yield households in batches,
        i. e. `self.samples` samples from `MT_001` first, then `self.samples` samples from `MT_002`, and so on.
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
