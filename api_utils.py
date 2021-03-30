import torch
import datetime
import numpy as np
import pandas as pd
from typing import List
from pydantic import BaseModel
from model import ForecasterQR
from test_pred import predict
from DatasetHandler import DatasetHandler
from fastapi.encoders import jsonable_encoder


class TimeSeriesData(BaseModel):
    household_name: str
    consumption: float
    timestamp: datetime.datetime


def api_predict(model: ForecasterQR, dataloader: torch.utils.data.DataLoader) -> np.array:
    household, start_timestamp, _ , _, y_pred = predict(model=model, dataset=dataloader.dataset, index=0)
    y_pred = y_pred.cpu().detach().numpy()
    pred_start_ts = start_timestamp + datetime.timedelta(hours=168)
    pred_end_ts = pred_start_ts + datetime.timedelta(hours=model.horizons-1)
    predicted_time = pd.date_range(pred_start_ts, pred_end_ts, freq='H')
    result = []
    for i, ts in enumerate(predicted_time):
        result.append(
            {
                "household_name": household,
                "timestamp": ts,
                "predicted_quantiles": y_pred[i,:].tolist()
            }
        )
    return result



def query_to_dataloader(data: List[TimeSeriesData]) -> pd.DataFrame:
    df = pd.DataFrame(jsonable_encoder(data))
    df = df.pivot_table(values="consumption", index="timestamp", columns="household_name")
    df.reset_index(inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    dataset_handler = DatasetHandler(
        data_path=None,
        num_samples=None,
        batch_size=len(df),
        val_split_ratio=0,
        pred_horizon=24,
        hist_hours=len(df) // 4,
        forking_total_seq_length=None)
    dataloader = dataset_handler.load_dataset(df, False)
    return dataloader