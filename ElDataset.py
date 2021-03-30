import torch
import datetime
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class ElDataset(Dataset):
    """Electricity dataset."""

    def __init__(self, df, num_samples = None, hist_hours=168, future_hours=24, forking_total_seq_length=None):
        """
        Args:
            df: original electricity data (see HW intro for details).
            samples (int): number of sample to take per household.
        """
        self.raw_data = self.el_resample(df).set_index("timestamp")
        self.num_samples = num_samples
        self.hist_hours = hist_hours
        self.future_hours = future_hours
        if forking_total_seq_length is None:
            self.full_length = pd.Timedelta(hours=(hist_hours + future_hours))
        else:
            self.full_length = pd.Timedelta(hours=(forking_total_seq_length))
        self.forking_total_seq_length = forking_total_seq_length
        self.sample()

    def __len__(self):
        if self.num_samples:
            return self.num_samples * (self.raw_data.shape[1] - len(self.calendar_features))
        return  self.num_samples * (self.raw_data.shape[1] - len(self.calendar_features))

    def __getitem__(self, idx):
        """Yield one sample, according to `self.get_mapping(idx)`."""

        household, start_ts = self.mapping[idx]
        if self.forking_total_seq_length is None:
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

        else:  # forking
            start = start_ts
            end = start_ts + pd.Timedelta(hours=self.forking_total_seq_length - 1)
            unsliced_data = torch.stack(
                [
                    torch.Tensor(self.raw_data.loc[start:end, household].values),
                    torch.Tensor(self.raw_data.loc[start:end, "yearly_cycle"].values),
                    torch.Tensor(self.raw_data.loc[start:end, "weekly_cycle"].values),
                    torch.Tensor(self.raw_data.loc[start:end, "daily_cycle"].values),
                ],
                axis=-1
            )  # shape forking_total_seq_length,4
            tot_samples = self.forking_total_seq_length - (self.hist_hours + self.future_hours)
            data = torch.zeros([tot_samples, self.hist_hours + self.future_hours, unsliced_data.shape[1]])
            # mask = torch.ones([tot_samples]) #handles the cases of fct>horizon-(hist_hours+future_hours) can be also solved by masking
            for fct in range(tot_samples):
                # slice = unsliced_data[fct:fct + self.hist_hours + self.future_hours, :]
                data[fct, :, :] = unsliced_data[fct:fct + self.hist_hours + self.future_hours, :]

            # data = data[mask, :]
            x_data = data[:, :self.hist_hours, 0].unsqueeze(-1)
            x_calendar_past = data[:, :self.hist_hours, 1:]
            x_calendar_future = data[:, self.hist_hours:, 1:]
            y = data[:, self.hist_hours:, 0]
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

        if self.num_samples:
            idx = np.arange(self.num_samples * self.raw_data.shape[1])
            np.random.shuffle(idx)
            timestamps = self.raw_data[:(self.raw_data.index.max() - self.full_length)].index.to_series()
        else:
            idx = np.arange(self.raw_data.shape[1])
            timestamps = pd.date_range(self.raw_data.index[0],
                          self.raw_data.index[0] + datetime.timedelta(hours=(self.future_hours + self.hist_hours - 1)),
                          freq='H')
            self.raw_data = self.raw_data.reindex(timestamps)

        pairs = []
        for household in self.raw_data.columns:
            start_ts = timestamps.sample(self.num_samples) if self.num_samples else [timestamps[0]]
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
