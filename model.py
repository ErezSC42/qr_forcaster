
import torch
from torch import nn
from typing import List
import pytorch_lightning as pl

from Metrics.Losses import QuantileLoss


class Encoder(pl.LightningModule):
    '''
    Encoder module for timeseries. creates encoded representation of input sequence using LSTM
    '''

    def __init__(
            self,
            data_dim: int,
            hidden_dim: int,
            num_layers: int,
            max_sequence_len: int):
        super(Encoder, self).__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_sequence_len = max_sequence_len
        self.encoder_lstm = nn.LSTM(input_size=data_dim, hidden_size=hidden_dim, batch_first=True,
                                    num_layers=num_layers)

    def forward(self, x):
        # TODO return all the states or only the last
        output, (henc, cenc) = self.encoder_lstm(x.view(x.shape[0], x.shape[1], 1))
        return henc[:, 0, :], cenc[:, 0, :]  # returns encoded state [batch_size, hidden_dim]


class DecoderGlobal(pl.LightningModule):
    '''
    Global decoder for output of encoder LSTM. single instance in full model
    '''

    def __init__(
            self,
            encoder_hidden_dim: int,
            context_dim: int,
            x_future_dim: int,
            k: int):
        super(DecoderGlobal, self).__init__()
        self.input_dim = encoder_hidden_dim + k * x_future_dim  # the dimension of the input data
        self.output_dim = (k + 1) * context_dim  # k local contexts and global context
        self.mlp = nn.Linear(in_features=self.input_dim, out_features=self.output_dim)

    def forward(self, x):
        x = torch.nn.functional.relu(self.mlp(x))
        return x


class DecoderLocal(pl.LightningModule):
    '''
    Local decoder for quantile prediction of specific timestep.
    parameters are shared between timestamps, meaning there is a single local decoder in the network
    '''

    def __init__(
            self,
            context_dim,
            future_data_dim: int,
            quantiles: List[float],
    ):
        super(DecoderLocal, self).__init__()
        self.input_dim = 2 * context_dim + future_data_dim  # local context, global context and future data
        quantiles_num = len(quantiles)
        self.output_dim = quantiles_num  # each local decoder outputs q values
        self.mlp = nn.Linear(in_features=self.input_dim, out_features=self.output_dim)

    def forward(self, context_vector, context_alpha_vector , x_future_data=None):
        if x_future_data:
            vec_list = [context_vector, context_alpha_vector, x_future_data]
        else:
            vec_list = [context_vector, context_alpha_vector]
        x = torch.cat(vec_list, dim=1)
        x = torch.nn.functional.relu(self.mlp(x))
        return x


class ForecasterQR(pl.LightningModule):
    '''
    Full class of forecaster module
    '''

    def __init__(
            self,
            y_dim: int,
            x_dim: int,
            input_max_squence_len: int,
            encoder_hidden_dim: int,
            encoder_num_layers: int,
            decoder_context_dim: int,
            quantiles: List[float],
            horizons: int,
            device: str
    ):
        super(ForecasterQR, self).__init__()
        self.encoder = Encoder(
            data_dim=y_dim,
            hidden_dim=encoder_hidden_dim,
            num_layers=encoder_num_layers,
            max_sequence_len=input_max_squence_len)

        self.horizons = horizons
        self.quantiles = quantiles
        self.context_dim = decoder_context_dim
        self.q = len(self.quantiles)
        self.device_ = device
        self.loss = QuantileLoss(quantiles, device=self.device_)

        # TODO correctly init decoders
        self.global_decoder = DecoderGlobal(encoder_hidden_dim=encoder_hidden_dim,
                                            context_dim=decoder_context_dim,
                                            x_future_dim=x_dim,
                                            k=horizons)

        # create a local decoder foreach output step
        self.local_decoder = DecoderLocal(context_dim=decoder_context_dim,
                                          future_data_dim=x_dim, quantiles=quantiles)

    def forward(self, y_tensor, x_tensor=None, x_future_tensor=None):
        '''
        :param y_tensor: time series data of past
        :param x_tensor: feature/calender data of the past
        :param x_future_tensor: feature/calaender data of the future
        :return:
        '''
        batch_size = y_tensor.shape[0]
        if x_tensor:
            past_vector = torch.cat([y_tensor, x_tensor], axis=1)
        else:
            past_vector = y_tensor
        encoded_hidden_state, _ = self.encoder(past_vector)
        global_state = self.global_decoder(encoded_hidden_state)

        # init output tensor in [batch_size, horizons, quantiles]
        output_tensor = torch.zeros([batch_size, self.horizons, self.q])

        # use local decoder k times to get the quantile outputs foreach horizon
        for k in range(self.horizons):
            # take the correct elements from the global_state vector, matching the current k
            c_alpha = global_state[:, -self.context_dim:]  # get c_alpha
            c_t_k = global_state[:, k * self.context_dim:(k + 1) * self.context_dim]
            output_tensor[:, k, :] = self.local_decoder(c_t_k, c_alpha, x_future_tensor).unsqueeze(dim=1)

        return output_tensor

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        pred = self(x)
        loss = self.loss(pred, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        pred = self(x)
        loss = self.loss(pred, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
