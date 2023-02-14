import os
import pickle
import random
import datetime
from config import prod_model_path
from model import ForecasterQR
import matplotlib.pyplot as plt

data_path = os.path.join("data", "LD2011_2014.txt")

TRAIN_DL_PATH = os.path.join("dataloaders", "train_dl.pkl")
TEST_DL_PATH = os.path.join("dataloaders", "test_dl.pkl")


def predict(model, dataset, index):
    asset, start = dataset.get_mapping(index)
    data_sample = dataset[index]
    features = data_sample[0]
    future_series = data_sample[1]
    past_series = features[0].unsqueeze(0)
    x_tensor = features[1].unsqueeze(0)
    x_future_tensor = features[2].unsqueeze(0)
    res = model(y_tensor=past_series, x_tensor=x_tensor, x_future_tensor=x_future_tensor).squeeze()
    return asset, start, past_series, future_series, res


def plot_prediction(asset, model_output, y_past, y_future, hist_ts, future_ts):
    res = model_output.cpu().detach().numpy()
    quantiles_num = len(model.quantiles)
    half = (quantiles_num - 1) // 2
    fig = plt.figure()
    plt.title(f"prediction for asset {asset}")
    plt.plot(hist_ts, y_past.squeeze(), label="past consumption")
    plt.plot(future_ts, y_future, label="actual consumption")
    for i, q in enumerate(model.quantiles):
        plt.plot(future_ts, res[:, i], label=f"q={q}")
    res = res[:, 1:]
    for i in range(half):
        alph = 0.0 + 2 * (i / len(model.quantiles))
        plt.fill_between(future_ts, res[:, i], res[:, -(i + 1)],
                         color="g", alpha=alph)
    fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
    plt.legend(loc="upper left")
    plt.xlabel("time")
    plt.ylabel("asset")
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # trained_model_list = os.listdir("trained_models")
    # best_model_index = np.argmin([int(name.split("=")[2].split(".")[0]) for name in trained_model_list])
    # TRAINED_MODEL_NAME = os.listdir("trained_models")[best_model_index]
    # TRAINED_MODEL_PATH = os.path.join("trained_models", TRAINED_MODEL_NAME)

    # TRAINED_MODEL_PATH = os.path.join("trained_models/model-epoch=49-val_loss=410.90.ckpt")
    # TRAINED_MODEL_PATH = os.path.join("trained_models", "model-epoch=04-val_loss=709.13.ckpt")
    TRAINED_MODEL_PATH = prod_model_path
    model = ForecasterQR.load_from_checkpoint(TRAINED_MODEL_PATH)
    model.eval()

    print(f"loaded model: {TRAINED_MODEL_PATH}")

    # test data
    with open(TEST_DL_PATH, "rb") as fp:
        pred_dl = pickle.load(fp)

    pred_dataset = pred_dl.dataset
    index = random.randint(0, len(pred_dataset))
    print(f"sampled index: {index}")

    # asset = "MT_006"
    # s_index = 14000
    # s_len = 674
    # df_temp = pred_dl.dataset.raw_data[[asset]]
    # df_temp["asset_name"] = asset
    # df_temp.reset_index(inplace=True)
    # df_temp.rename(columns={asset: "consumption"}, inplace=True)
    # df_temp["timestamp"] = pd.to_datetime(df_temp["timestamp"])
    # df_temp = df_temp.iloc[s_index:s_index+s_len, :]
    # df_temp.to_json("dummmy.json", orient="records")

    asset, hist_start, past_series, future_series, res = predict(model, dataset=pred_dataset, index=index)
    hist_end = hist_start + pred_dataset.hist_days
    future_start = hist_end + 1
    future_end = future_start + pred_dataset.future_days
    hist_ts = pred_dataset.raw_data.index[hist_start:hist_end]
    future_ts = pred_dataset.raw_data.index[future_start:future_end]
    plot_prediction(asset, res, y_past=past_series, y_future=future_series, 
        hist_ts=hist_ts, future_ts=future_ts)
