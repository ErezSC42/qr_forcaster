import os

import pytorch_lightning as pl
import torch
from torch import nn

from qr_forcaster.DatasetHandler import DatasetHandler
from qr_forcaster.Metrics.Losses import DummyLoss
from qr_forcaster.model import ForecasterQR

data_path = os.path.join("data", "LD2011_2014.txt")


class DummyEncoderDecoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.decoder = nn.LSTM(input_size=50, hidden_size=24, num_layers=1, batch_first=False)
        self.loss_calculator = DummyLoss()

    def forward(self, x):
        o1, (h1, c1) = self.encoder(x)  # batch, sequence, feature
        o2, (h2, c2) = self.decoder(h1)  # sequence, batch, input_size
        return o1, o2, h1, h2

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        o1, o2, h1, h2 = self(x.view(x.shape[0], 168, 1))
        loss = self.loss_calculator.calc_loss(o2.view(o2.shape[1], o2.shape[2]), y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        o1, o2, h1, h2 = self(x.view(x.shape[0], 168, 1))
        loss = self.loss_calculator.calc_loss(o2.view(o2.shape[1], o2.shape[2]), y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == '__main__':
    train_el_dataloader, val_el_dataloader = DatasetHandler(data_path, num_samples=100, hist_hours=168, pred_horizon=24, batch_size=32, device="cpu").load_dataset()
    quantiles = [.1, .2, .3, .4, .5, .6, .7, .8, .9, .95]
    model = ForecasterQR(
        x_dim=0,
        y_dim=1,
        input_max_squence_len=168,
        encoder_hidden_dim=8,
        encoder_num_layers=1,
        decoder_context_dim=4,
        quantiles=quantiles,
        horizons=24)
    trainer = pl.Trainer()#gpus=1)
    trainer.fit(model, train_el_dataloader, val_el_dataloader)

# TODO config file? num_gpus, data_path, output(logs+visualization) path etc
