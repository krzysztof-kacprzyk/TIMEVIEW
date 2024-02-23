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


class Config():

    def __init__(self,
                 n_features=1,
                 n_basis=5,
                 T=1,
                 seed=42,
                 encoder={'hidden_sizes': [32, 64, 32],
                          'activation': 'relu', 'dropout_p': 0.2},
                 training={'optimizer': 'adam', 'lr': 1e-3,
                           'batch_size': 32, 'weight_decay': 1e-5},
                 dataset_split={'train': 0.8, 'val': 0.1, 'test': 0.1},
                 dataloader_type='iterative',
                 device='cpu',
                 num_epochs=200,
                 internal_knots=None,
                 n_basis_tunable=False,
                 dynamic_bias=False):

        if not isinstance(n_features, int):
            raise ValueError("n_features must be an integer > 0")
        if n_basis < 4:
            raise ValueError("num_basis must be at least 4")
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
        if not isinstance(dataset_split['train'], float):
            raise ValueError("dataset_split['train'] must be a float")
        if not isinstance(dataset_split['val'], float):
            raise ValueError("dataset_split['val'] must be a float")
        if not isinstance(dataset_split['test'], float):
            raise ValueError("dataset_split['test'] must be a float")
        if dataset_split['train'] + dataset_split['val'] + dataset_split['test'] != 1.0:
            raise ValueError(
                "dataset_split['train'] + dataset_split['val'] + dataset_split['test'] must equal 1.0")
        if dataloader_type not in ['iterative', 'tensor']:
            raise ValueError(
                "dataloader_type must be one of ['iterative','tensor']")

        self.n_basis = n_basis
        self.n_features = n_features
        self.T = T
        self.seed = seed
        self.encoder = SimpleNamespace(**encoder)
        self.training = SimpleNamespace(**training)
        self.dataset_split = SimpleNamespace(**dataset_split)
        self.dataloader_type = dataloader_type
        self.device = device
        self.num_epochs = num_epochs
        self.internal_knots = internal_knots
        self.n_basis_tunable = n_basis_tunable
        self.dynamic_bias = dynamic_bias



class TuningConfig(Config):
    def __init__(
        self,
        trial,
        n_features=1,
        n_basis=5,
        T=1,
        seed=42,
        dataset_split={'train': 0.8, 'val': 0.1, 'test': 0.1},
        dataloader_type='iterative',
        device='cpu',
        num_epochs=200,
        internal_knots=None,
        n_basis_tunable=False,
        dynamic_bias=False
    ):

        # define hyperparameter search space
        hidden_sizes = [trial.suggest_int(
            f'hidden_size_{i}', 16, 128) for i in range(3)]
        # the activation search range might be a bit excessive, but it's a good example
        activation = trial.suggest_categorical(
            'activation',
            ['relu', 'sigmoid', 'tanh', 'leaky_relu', 'elu', 'selu']
        )
        dropout_p = trial.suggest_float('dropout_p', 0.0, 0.5)

        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
        batch_size = trial.suggest_categorical('batch_size', [64, 128])
        weight_decay = trial.suggest_float(
            'weight_decay', 1e-6, 1e-1, log=True)
        
        if n_basis_tunable:
            n_basis = trial.suggest_int('n_basis', 5, 16)

        encoder = {
            'hidden_sizes': hidden_sizes,
            'activation': activation,
            'dropout_p': dropout_p
        }
        training = {
            'optimizer': 'adam',
            'lr': lr,
            'batch_size': batch_size,
            'weight_decay': weight_decay
        }

        if not isinstance(n_features, int):
            raise ValueError("n_features must be an integer > 0")
        if n_basis < 4:
            raise ValueError("num_basis must be at least 4")
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
        if not isinstance(dataset_split['train'], float):
            raise ValueError("dataset_split['train'] must be a float")
        if not isinstance(dataset_split['val'], float):
            raise ValueError("dataset_split['val'] must be a float")
        if not isinstance(dataset_split['test'], float):
            raise ValueError("dataset_split['test'] must be a float")
        if dataset_split['train'] + dataset_split['val'] + dataset_split['test'] != 1.0:
            raise ValueError(
                "dataset_split['train'] + dataset_split['val'] + dataset_split['test'] must equal 1.0")
        if dataloader_type not in ['iterative', 'tensor']:
            raise ValueError(
                "dataloader_type must be one of ['iterative','tensor']")

        self.n_basis = n_basis
        self.n_features = n_features
        self.T = T
        self.seed = seed
        self.encoder = SimpleNamespace(**encoder)
        self.training = SimpleNamespace(**training)
        self.dataset_split = SimpleNamespace(**dataset_split)
        self.dataloader_type = dataloader_type
        self.device = device
        self.num_epochs = num_epochs
        self.internal_knots = internal_knots
        self.n_basis_tunable = n_basis_tunable
        self.dynamic_bias = dynamic_bias
        

