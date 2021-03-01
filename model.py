from torch import nn


class Encoder(nn.Module):
    '''
    Encoder module for timeseries. creates encoded representation of input sequence using LSTM
    '''

    def __init__(
            self,
            data_dim: int,
            hidden_dim: int,
            num_layers: int,
            max_sequence_len: int):
        super(Encoder).__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_sequence_len = max_sequence_len
        self.encoder_lstm = nn.LSTM(input_size=data_dim, hidden_size=hidden_dim, batch_first=True,
                                    num_layers=num_layers)

    def forward(self, x):
        # TODO returm all the states or only the last
        output, (henc, cenc) = self.encoder_lstm(x.view(x.shape[0], x.shape[1], 1))
        return henc[:, 0, :], cenc[:, 0, :]  # returns encoded state [batch_size, hidden_dim]


class DecoderGlobal(nn.Module):
    '''
    Global decoder for output of encoder LSTM. single instance in full model
    '''

    def __init__(
            self,
            encoder_hidden_dim: int,
            context_dim: int,
            x_future_dim: int,
            k: int):
        super(DecoderGlobal).__init__()
        self.input_dim = encoder_hidden_dim + k * x_future_dim  # the dimension of the input data
        self.output_dim = (k + 1) * context_dim  # k local contexts and global context
        self.mlp = nn.Linear(in_features=self.input_dim, out_features=self.output_dim)

    def forward(self, x):
        x = self.mlp(x)
        # TODO add activation and check dims
        return x


class DecoderLocal(nn.Module):
    '''
    Local decoder for quantile prediction of specific timestep. Instance a local encoder for each horizon dim.
    '''

    def __init__(
            self,
            context_dim,
            future_data_dim: int,
            quantiles_num: int,
    ):
        super(DecoderLocal).__init__()
        self.input_dim = 2 * context_dim + future_data_dim  # local context, global context and future data
        self.output_dim = quantiles_num  # each local decoder outputs q values
        self.mlp = nn.Linear(in_features=self.input_dim, out_features=self.output_dim)

    def forward(self, x):
        x = self.mlp(x)
        # TODO add activation and check dims
        return x


class ForecasterQR(nn.Module):
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
            quantiles: int,
            horizons: int,
    ):
        super(ForecasterQR, self).__init__()
        self.encoder = Encoder(
            data_dim=y_dim,
            hidden_dim=encoder_hidden_dim,
            num_layers=encoder_num_layers,
            max_sequence_len=input_max_squence_len)

        # TODO correctly init decoders
        self.global_decoder = DecoderGlobal(encoder_hidden_dim=encoder_hidden_dim,
                                            context_dim=decoder_context_dim,
                                            x_future_dim=x_dim,
                                            k=horizons)

        # create a local decoder foreach output step
        self.local_decoders = [DecoderLocal(context_dim=decoder_context_dim,
                                            future_data_dim=x_dim,
                                            quantiles_num=quantiles) for _ in range(horizons)]

    def forward(self, x):
        pass
