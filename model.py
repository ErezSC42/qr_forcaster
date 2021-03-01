from torch import nn


class Encoder(nn.Module):
    def __init__(
            self,
            data_dim: int,
            hidden_dim: int,
            num_layers: int,
            max_sequence_len: int):
        super().__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_sequence_len = max_sequence_len
        self.encoder_lstm = nn.LSTM(input_size=data_dim, hidden_size=hidden_dim, batch_first=True, num_layers=num_layers)

    def forward(self, x):
        output, (henc, cenc) = self.encoder_lstm(x.view(x.shape[0], x.shape[1], 1))
        return henc[:,0,:] # returns encoded state [batch_size, hidden_dim]


