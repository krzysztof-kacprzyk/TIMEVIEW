from types import SimpleNamespace
import torch

ACTIVATION_FUNCTIONS = {
    'relu': torch.nn.ReLU,
    'sigmoid': torch.nn.Sigmoid,
    'tanh': torch.nn.Tanh,
    'leaky_relu': torch.nn.LeakyReLU,
    'elu': torch.nn.ELU,
    'selu': torch.nn.SELU
}

OPTIMIZERS = {
    'adam': torch.optim.Adam,
    'adagrad': torch.optim.Adagrad,
    'adadelta': torch.optim.Adadelta,
    'sgd': torch.optim.SGD,
    'rmsprop': torch.optim.RMSprop
}


class RNNConfig():

    def __init__(self,
                 decoder_type,
                 n_features=1,
                 seed=42,
                 max_len=10,
                 encoder={'hidden_sizes': [32, 64, 32],
                          'activation': 'relu', 'dropout_p': 0.2},
                 decoder={'input_dim': 32, 'hidden_dim': 32,
                          'output_dim': 1, 'num_layers': 1, 'dropout_p': 0.2},
                 training={'optimizer': 'adam', 'lr': 1e-3,
                           'batch_size': 32, 'weight_decay': 1e-5},
                device='cpu',
                num_epochs=200):

        assert encoder['hidden_sizes'][-1] == decoder['input_dim']

        if not isinstance(n_features, int):
            raise ValueError("n_features must be an integer > 0")
        if not isinstance(encoder['hidden_sizes'], list):
            raise ValueError(
                "encoder['hidden_sizes'] must be a list of integers")
        if not all([isinstance(x, int) for x in encoder['hidden_sizes']]):
            raise ValueError(
                "encoder['hidden_sizes'] must be a list of integers")
        if encoder['activation'] not in ACTIVATION_FUNCTIONS:
            raise ValueError("encoder['activation'] must be one of {}".format(
                list(ACTIVATION_FUNCTIONS.keys())))
        if training['optimizer'] not in OPTIMIZERS:
            raise ValueError("optimizer['name'] must be one of {}".format(
                list(OPTIMIZERS.keys())))
        if not isinstance(training['lr'], float):
            raise ValueError("optimizer['lr'] must be a float")
        if not isinstance(training['batch_size'], int):
            raise ValueError("training['batch_size'] must be an integer")
        if not isinstance(training['weight_decay'], float):
            raise ValueError("training['weight_decay'] must be a float")
       
        self.n_features = n_features
        self.seed = seed
        self.max_len = max_len
        self.decoder_type = decoder_type
        self.encoder = SimpleNamespace(**encoder)
        self.decoder = SimpleNamespace(**decoder)
        self.training = SimpleNamespace(**training)
        self.device = device
        self.num_epochs = num_epochs




class RNNTuningConfig():
    def __init__(
        self,
        trial,
        decoder_type,
        n_features=1,
        max_len=10,
        seed=42,
        device='cpu',
        num_epochs=200
    ):
        assert decoder_type in ['lstm', 'rnn']

        # define hyperparameter search space
        hidden_sizes = [trial.suggest_int(
            f'hidden_size_{i}', 16, 128) for i in range(3)]
        # the activation search range might be a bit excessive, but it's a good example
        activation = trial.suggest_categorical(
            'activation',
            ['relu', 'sigmoid', 'tanh', 'leaky_relu', 'elu', 'selu']
        )
        encoder_dropout_p = trial.suggest_float('dropout_p', 0.0, 0.5)

        decoder_hidden_dim = trial.suggest_int('decoder_hidden_dim', 16, 128)
        decoder_num_layers = trial.suggest_int('decoder_num_layers', 1, 3)
        decoder_dropout_p = trial.suggest_float('decoder_dropout_p', 0.0, 0.5)

        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [64, 128])
        weight_decay = trial.suggest_float(
            'weight_decay', 1e-6, 1e-3, log=True)

        encoder = {
            'hidden_sizes': hidden_sizes,
            'activation': activation,
            'dropout_p': encoder_dropout_p
        }
        decoder = {
            'input_dim': hidden_sizes[-1],
            'hidden_dim': decoder_hidden_dim,
            'output_dim': 1,
            'num_layers': decoder_num_layers,
            'dropout_p': decoder_dropout_p
        }

        training = {
            'optimizer': 'adam',
            'lr': lr,
            'batch_size': batch_size,
            'weight_decay': weight_decay
        }

        assert encoder['hidden_sizes'][-1] == decoder['input_dim']
        if not isinstance(n_features, int):
            raise ValueError("n_features must be an integer > 0")
        if not isinstance(encoder['hidden_sizes'], list):
            raise ValueError(
                "encoder['hidden_sizes'] must be a list of integers")
        if not all([isinstance(x, int) for x in encoder['hidden_sizes']]):
            raise ValueError(
                "encoder['hidden_sizes'] must be a list of integers")
        if encoder['activation'] not in ACTIVATION_FUNCTIONS:
            raise ValueError("encoder['activation'] must be one of {}".format(
                list(ACTIVATION_FUNCTIONS.keys())))
        if training['optimizer'] not in OPTIMIZERS:
            raise ValueError("optimizer['name'] must be one of {}".format(
                list(OPTIMIZERS.keys())))
        if not isinstance(training['lr'], float):
            raise ValueError("optimizer['lr'] must be a float")
        if not isinstance(training['batch_size'], int):
            raise ValueError("training['batch_size'] must be an integer")
        if not isinstance(training['weight_decay'], float):
            raise ValueError("training['weight_decay'] must be a float")


        self.n_features = n_features
        self.seed = seed
        self.max_len = max_len
        self.decoder_type = decoder_type
        self.encoder = SimpleNamespace(**encoder)
        self.decoder = SimpleNamespace(**decoder)
        self.training = SimpleNamespace(**training)
        self.device = device
        self.num_epochs = num_epochs
        

