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
                 encoder={'hidden_sizes':[32,64,32],'activation':'relu'},
                 training={'optimizer':'adam','lr':1e-3,'batch_size':32,'num_epochs':100,'patience':10,'verbose':True},
                 dataset_split={'train':0.8,'val':0.1,'test':0.1},
                 dataloader_type='iterative'):
        
        if not isinstance(n_features,int):
            raise ValueError("n_features must be an integer > 0")
        if n_basis < 4:
            raise ValueError("num_basis must be at least 4")
        if not isinstance(encoder['hidden_sizes'],list):
            raise ValueError("encoder['hidden_sizes'] must be a list of integers")
        if not all([isinstance(x,int) for x in encoder['hidden_sizes']]):
            raise ValueError("encoder['hidden_sizes'] must be a list of integers")
        if encoder['activation'] not in ACTIVATION_FUNCTIONS:
            raise ValueError("encoder['activation'] must be one of {}".format(list(ACTIVATION_FUNCTIONS.keys())))   
        if training['optimizer'] not in OPTIMIZERS:
            raise ValueError("optimizer['name'] must be one of {}".format(list(OPTIMIZERS.keys())))
        if not isinstance(training['lr'],float):
            raise ValueError("optimizer['lr'] must be a float")
        if not isinstance(training['batch_size'],int):
            raise ValueError("training['batch_size'] must be an integer")
        if not isinstance(training['num_epochs'],int):
            raise ValueError("training['num_epochs'] must be an integer")
        if not isinstance(training['patience'],int):
            raise ValueError("training['patience'] must be an integer")
        if not isinstance(training['verbose'],bool):
            raise ValueError("training['verbose'] must be a boolean")
        if not isinstance(dataset_split['train'],float):
            raise ValueError("dataset_split['train'] must be a float")
        if not isinstance(dataset_split['val'],float):
            raise ValueError("dataset_split['val'] must be a float")
        if not isinstance(dataset_split['test'],float):
            raise ValueError("dataset_split['test'] must be a float")
        if dataset_split['train'] + dataset_split['val'] + dataset_split['test'] != 1.0:
            raise ValueError("dataset_split['train'] + dataset_split['val'] + dataset_split['test'] must equal 1.0")
        if dataloader_type not in ['iterative','tensor']:
            raise ValueError("dataloader_type must be one of ['iterative','tensor']")
        
        self.n_basis = n_basis
        self.n_features = n_features
        self.T = T
        self.seed = seed
        self.encoder = SimpleNamespace(**encoder)
        self.training = SimpleNamespace(**training)
        self.dataset_split = SimpleNamespace(**dataset_split)
        self.dataloader_type = dataloader_type


    

    
