import torch
import numpy as np

from tts.data import BaseDataset

def _pad_to_shape(a, shape):
    """
    This function pads a 1D, 2D or 3D numpy array with zeros to a specified shape
    Args:
        a: a numpy array
        shape: a tuple of integers
    Returns:
        a numpy array of shape shape
    """
    if a.shape == shape:
        return a
    if len(a.shape) == 1:
        assert a.shape[0] <= shape[0]
        b = np.zeros(shape)
        b[:a.shape[0]] = a
    elif len(a.shape) == 2:
        assert a.shape[0] <= shape[0]
        assert a.shape[1] <= shape[1]
        b = np.zeros(shape)
        b[:a.shape[0], :a.shape[1]] = a
    elif len(a.shape) == 3:
        assert a.shape[0] <= shape[0]
        assert a.shape[1] <= shape[1]
        assert a.shape[2] <= shape[2]
        b = np.zeros(shape)
        b[:a.shape[0], :a.shape[1], :a.shape[2]] = a
    return b

class NeuralODEDataset(torch.utils.data.Dataset):
    def __init__(self, config, X, ts, ys):
        self.X = X
        self.ts = ts
        self.ys = ys
        self.config = config
        self._process_data()
    
    def _process_data(self):

        self.X = torch.from_numpy(np.array(self.X)).float()

        self.D = self.X.shape[0]
        self.M = self.X.shape[1]
        self.Ns = [self.ts[i].shape[0] for i in range(len(self.ts))]
        self.N_max = max(self.Ns)

        if self.config.dataloader_type == 'tensor':
            # We pad ys and stack into a tensor
            self.Y = torch.stack([torch.from_numpy(_pad_to_shape(
                y, (self.N_max,))).float() for y in self.ys], dim=0)
            # We pad ts and stack into a tensor
            self.T = torch.stack([torch.from_numpy(_pad_to_shape(
                t, (self.N_max,))).float() for t in self.ts], dim=0)
            # We turn Ns into a tensor
            self.NS = torch.tensor(self.Ns)

        elif self.config.dataloader_type == 'iterative':
            # Convert Phi to a list of tensors
            self.ts = [torch.from_numpy(t).float() for t in self.ts]
            # Convert ys to a list of tensors
            self.ys = [torch.from_numpy(y).float() for y in self.ys]

    def __getitem__(self, idx):
        if self.config.dataloader_type == 'iterative':
            return self.X[idx, :], self.ts[idx], self.ys[idx]
        elif self.config.dataloader_type == 'tensor':
            return self.X[idx, :], self.T[idx, :], self.Y[idx, :], self.NS[idx]


    def __len__(self):
        return self.X.shape[0]
    
    def get_collate_fn(self):
        def collate_fn(batch):
            Xs = []
            ts = []
            ys = []
            for b in batch:
                Xs.append(b[0])
                ts.append(b[1])
                ys.append(b[2])
            return torch.stack(Xs, dim=0), ts, ys
        
        if self.config.dataloader_type == 'iterative':
            return collate_fn
        elif self.config.dataloader_type == 'tensor':
            return None
    
def create_neural_ode_dataloader(config, dataset, indices=None, shuffle=True):
    """
    Creates a dataloader for a subset of the dataset described by the list of indices
    Args:
        config: an instance of the Config class
        dataset: an instance of the TTSDataset class
        indices: a list of integers that are the indices of the dataset
        shuffle: a boolean that indicates whether to shuffle the indices
    Returns:
        dataloader: a torch.utils.data.DataLoader object
    """
    gen = torch.Generator()
    seed = config.seed
    gen.manual_seed(seed)

    if indices is None:
        subset = dataset
    else:
        subset = torch.utils.data.Subset(dataset, indices)
    collate_fn = dataset.get_collate_fn()
    dataloader = torch.utils.data.DataLoader(
        subset, batch_size=config.training.batch_size, shuffle=shuffle, generator=gen, collate_fn=collate_fn)
    return dataloader
