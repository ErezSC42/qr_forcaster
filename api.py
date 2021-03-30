import os
import uvicorn
from typing import List
from fastapi import FastAPI
from model import ForecasterQR
from config import prod_model_path

from api_utils import TimeSeriesData, query_to_dataloader, api_predict

app = FastAPI()

#TRAINED_MODEL_PATH = os.path.join("trained_models/model-epoch=49-val_loss=410.90.ckpt")
TRAINED_MODEL_PATH = prod_model_path
model = ForecasterQR.load_from_checkpoint(TRAINED_MODEL_PATH)
model.eval()


@app.get("/")
def read_root():
    return {
        "Hello": "World"
    }


@app.post("/forecast")
def forecast(data: List[TimeSeriesData]):
    dataloader = query_to_dataloader(data)
    result = api_predict(model, dataloader)
    return result


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
