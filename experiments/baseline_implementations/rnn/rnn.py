'''
Pytorch implementation of encoder with fully-connected layers and RNN decoder
'''
import torch
import torch.nn as nn


class Encoder(nn.Module):
    '''
    MLP encoder
    '''

    def __init__(self, config):
        super().__init__()
        n_features = config.n_features
        hidden_sizes = config.encoder.hidden_sizes
        dropout_p = config.encoder.dropout_p

        self.layers = []
        activation = nn.ReLU()

        self.layers.append(nn.Linear(n_features, hidden_sizes[0]))
        self.layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        self.layers.append(activation)

        for i in range(len(hidden_sizes) - 2):
            self.layers.append(torch.nn.Linear(
                hidden_sizes[i], hidden_sizes[i+1]))
            self.layers.append(torch.nn.BatchNorm1d(hidden_sizes[i+1]))
            self.layers.append(activation)
            self.layers.append(torch.nn.Dropout(dropout_p))

        self.layers.append(torch.nn.Linear(hidden_sizes[-2], hidden_sizes[-1]))
        self.encoder = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    '''
    RNN/LSTM decoder
    '''

    def __init__(self, config):
        super().__init__()
        self.max_len = config.max_len
        decoder_type = config.decoder_type
        input_dim = config.decoder.input_dim
        hidden_dim = config.decoder.hidden_dim
        output_dim = config.decoder.output_dim
        num_layers = config.decoder.num_layers
        dropout_p = config.decoder.dropout_p

        if num_layers == 1:
            dropout_p = 0.

        if decoder_type == 'rnn':
            self.decoder = nn.RNN(
                input_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout_p
            )
        elif decoder_type == 'lstm':
            self.decoder = nn.LSTM(
                input_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout_p
            )
        else:
            raise ValueError('Unknown decoder type...')
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        assert len(x.shape) == 2

        # repeat input for each time step
        x = x.unsqueeze(1)  # x.shape (batch_size, 1, hidden_dim)
        # x.shape (batch_size, max_len, hidden_dim)
        x = x.repeat(1, self.max_len, 1)

        out, _ = self.decoder(x)
        out = self.linear(out)
        return out.squeeze()


class RNN(nn.Module):
    '''
    RNN/LSTM encoder-decoder model
    '''

    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
