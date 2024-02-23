import torch
import random
import numpy as np
from .config import Config, TuningConfig
from .basis import BSplineBasis
import pandas as pd
from scipy.integrate import odeint
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from abc import abstractmethod, ABC
from types import SimpleNamespace


class BaseDataset(ABC):

    def __init__(self, **args):
        self.args = SimpleNamespace(**args)

    @abstractmethod
    def get_X_ts_ys(self):
        """
        Returns:
            X: numpy array of shape (D,M) where D is the number of samples and M is the number of static features
            ts: a list of D 1D numpy arrays of shape (N_i,) where N_i is the number of time steps for the i-th sample
            ys: a list of D 1D numpy arrays of shape (N_i,) where N_i is the number of time steps for the i-th sample
        """
        pass

    @abstractmethod
    def __len__(self):
        """
        This function returns the number of samples in the dataset
        """
        pass

    @abstractmethod
    def get_feature_ranges(self):
        """
        This function returns a dictionary that maps feature names to feature ranges.
        """
        pass

    @abstractmethod
    def get_feature_names(self):
        """
        This function returns a list of feature names in the same order as they appear in the dataset.
        """
        pass

    def get_single_matrix(self, indices):
        X, ts, ys = self.get_X_ts_ys()
        static_cols = list(X.columns)
        samples = []
        for i in indices:
            n_rows = len(ts[i])
            X_i_tiled = pd.concat([X.iloc[[i],:]] * n_rows, ignore_index=True, axis=0)
            X_i_tiled['t'] = ts[i]
            X_i_tiled['y'] = ys[i]
            samples.append(X_i_tiled)
        whole_df = pd.concat(samples, axis=0, ignore_index=True)
        return whole_df[static_cols + ['t', 'y']]

    def _extract_data_from_one_dataframe(self, df):
        """
        This function extracts the data from one dataframe
        Args:
            df a pandas dataframe with columns ['id','x1','x2',...,'xM','t','y'] where the first M columns are the static features and the last two columns are the time and the observation
        Returns:
            X: a pandas dataframe of shape (D,M) where D is the number of samples and M is the number of static features
            ts: a list of D 1D numpy arrays of shape (N_i,) where N_i is the number of time steps for the i-th sample
            ys: a list of D 1D numpy arrays of shape (N_i,) where N_i is the number of time steps for the i-th sample
        """
        # TODO: Validate data

        ids = df['id'].unique()
        X = []
        ts = []
        ys = []
        for id in ids:
            df_id = df[df['id'] == id].copy()
            X.append(
                df_id.iloc[[0], 1:-2])
            # print(X)
            df_id.sort_values(by='t', inplace=True)
            ts.append(df_id['t'].values.reshape(-1))
            ys.append(df_id['y'].values.reshape(-1))
        X = pd.concat(X, axis=0, ignore_index=True)
        return X, ts, ys
    
    def get_feature_types(self):
        """
        This function returns a dictionary that maps feature names to feature types.
        """
        feature_ranges = self.get_feature_ranges()
        feature_types = {}
        for feature_name in feature_ranges:
            feature_range = feature_ranges[feature_name]
            if isinstance(feature_range, tuple):
                feature_types[feature_name] = 'continuous'
            elif isinstance(feature_range, list):
                if len(feature_range) > 2:
                    feature_types[feature_name] = 'categorical'
                elif len(feature_range) == 2:
                    feature_types[feature_name] = 'binary'
        return feature_types
    
    def get_feature_type(self, feature_name):
        """
        This function returns the type of a feature
        Args:
            feature_name: a string
        Returns:
            a string that is either 'continuous', 'categorical' or 'binary'
        """
        feature_range = self.get_feature_ranges()[feature_name]
        if isinstance(feature_range, tuple):
            return 'continuous'
        elif isinstance(feature_range, list):
            if len(feature_range) > 2:
                return 'categorical'
            elif len(feature_range) == 2:
                return 'binary'
        raise ValueError('Invalid feature range')
    
    def get_default_column_transformer(self, keep_categorical=False):
        """
        Creates a default column transformer for the dataset
        Returns:
            a sklearn ColumnTransformer object
        """
        transformers = []
        for feature_index, feature_name in enumerate(self.get_feature_names()):
            if self.get_feature_type(feature_name) == 'continuous':
                transformer = StandardScaler()
            elif self.get_feature_type(feature_name) == 'categorical' or self.get_feature_type(feature_name) == 'binary':
                if keep_categorical:
                    transformer = OrdinalEncoder(categories=[self.get_feature_ranges()[feature_name]])
                else:
                    transformer = OneHotEncoder(categories=[self.get_feature_ranges()[feature_name]],sparse_output=False,drop='if_binary')
            transformers.append((f"{feature_name}_transformer", transformer, [feature_index]))
        transformer = ColumnTransformer(transformers=transformers, remainder='passthrough') # The remainder option is needed to pass the time column for static methods
        return transformer

class XTYDataset(BaseDataset):
    def __init__(self, X, ts, ys, feature_ranges=None, feature_names=None):
        self.X = X
        self.ts = ts
        self.ys = ys
        self.feature_ranges = feature_ranges
        self.feature_names = feature_names

    def get_X_ts_ys(self):
        return self.X, self.ts, self.ys
    
    def get_feature_ranges(self):
        return self.feature_ranges
    
    def get_feature_names(self):
        return self.feature_names

    def __len__(self):
        return len(self.X)
    
   


def _validate_data(X, ts, ys, T):
    """
    This function verifies that the data is valid
    Args:
        X: numpy array of shape (D,M) where D is the number of samples and M is the number of static features
        ts: a list of D 1D numpy arrays of shape (N_i,) where T_i is the number of time steps for the i-th sample
        ys: a list of D 1D numpy arrays of shape (N_i,) where T_i is the number of time steps for the i-th sample
        T: a real number that is the time horizon
    """
    assert X.shape[0] == len(ts) == len(ys)
    for i in range(len(ts)):
        assert len(ts[i].shape) == len(ys[i].shape) == 1
        assert ts[i].shape[0] == ys[i].shape[0]
        assert ts[i].max() <= T
        assert ts[i].min() >= 0


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


class TTSDataset(torch.utils.data.Dataset):

    def __init__(self, config, data):
        """
        This constructor is helpful when you have irregularly sampled trajectories.
        Args:
            config: an instance of the Config class
            data: data can be any of the following:
                tuple (X,ts,ys), where
                    X: numpy array of shape (D,M) where D is the number of samples and M is the number of static features
                    ts: a list of D 1D numpy arrays of shape (N_i,) where N_i is the number of time steps for the i-th sample
                    ys: a list of D 1D numpy arrays of shape (N_i,) where N_i is the number of time steps for the i-th sample
                tuple (df_static,df_trajectories), where
                    df_static: a pandas dataframe with columns ['id','x1','x2',...,'xM'] corresponding to the static features
                    df_trajectories: a pandas dataframe with columns ['id','t','y']
                single dataframe df with columns ['id','x1','x2',...,'xM','t','y'] where the first M columns correspond to the static features and the last two columns correspond to the trajectories
            T: a real number that is the time horizon
        """
        self.config = config
        T = config.T
        if isinstance(data, pd.DataFrame):
            self._save_data(*self._extract_data_from_one_dataframe(data), T)
        elif isinstance(data, tuple):
            if len(data) == 2:
                self._save_data(
                    *self._extract_data_from_two_dataframes(data[0], data[1]), T)
            elif len(data) == 3:
                _validate_data(data[0], data[1], data[2], T)
                self._save_data(data[0], data[1], data[2], T)

        self._process_data()

    def _extract_data_from_two_dataframes(self, df_static, df_trajectories):
        """
        This function extracts the data from two dataframes
        Args:
            df_static: a pandas dataframe with columns ['id','x1','x2',...,'xM'] corresponding to the static features
            df_trajectories: a pandas dataframe with columns ['id','t','y']
        """
        ids = df_static['id'].unique()
        X = []
        ts = []
        ys = []
        for id in ids:
            df_static_id = df_static[df_static['id'] == id]
            # there should be only one row for each id
            assert df_static_id.shape[0] == 1
            X.append(df_static_id[0, 1:].values.astype(
                np.float32).reshape(1, -1))
            df_trajectories_id = df_trajectories[df_trajectories['id'] == id].copy(
            )
            df_trajectories_id.sort_values(by='t', inplace=True)
            ts.append(df_trajectories_id['t'].values.reshape(-1))
            ys.append(df_trajectories_id['y'].values.reshape(-1))
        X = np.concatenate(X, axis=0)
        return [X, ts, ys]

    def _extract_data_from_one_dataframe(self, df):
        """
        This function extracts the data from one dataframe
        Args:
              df a pandas dataframe with columns ['id','x1','x2',...,'xM','t','y'] where the first M columns are the static features and the last two columns are the time and the observation
        """
        # TODO: Validate data

        ids = df['id'].unique()
        X = []
        ts = []
        ys = []
        for id in ids:
            df_id = df[df['id'] == id].copy()
            X.append(
                df_id.iloc[0, 1:-2].values.astype(np.float32).reshape(1, -1))
            # print(X)
            df_id.sort_values(by='t', inplace=True)
            ts.append(df_id['t'].values.reshape(-1))
            ys.append(df_id['y'].values.reshape(-1))
        X = np.concatenate(X, axis=0)
        return [X, ts, ys]

    def _save_data(self, X, ts, ys, T):
        self.X = X
        self.ts = ts
        self.ys = ys
        self.T = T

    def _process_data(self):

        self.X = torch.from_numpy(np.array(self.X)).float()

        self.D = self.X.shape[0]
        self.M = self.X.shape[1]
        self.Ns = [self.ts[i].shape[0] for i in range(len(self.ts))]
        self.N_max = max(self.Ns)

        self.Phis = self._compute_matrices()

        if self.config.dataloader_type == 'tensor':
            # We pad ys and stack into a tensor
            self.Y = torch.stack([torch.from_numpy(_pad_to_shape(
                y, (self.N_max,))).float() for y in self.ys], dim=0)
            # We pad Phis and stack into a tensor
            self.PHI = torch.stack([torch.from_numpy(_pad_to_shape(
                Phi, (self.N_max, self.config.n_basis))).float() for Phi in self.Phis], dim=0)
            # We turn Ns into a tensor
            self.NS = torch.tensor(self.Ns)

        elif self.config.dataloader_type == 'iterative':
            # Convert Phi to a list of tensors
            self.Phis = [torch.from_numpy(Phi).float() for Phi in self.Phis]
            # Convert ys to a list of tensors
            self.ys = [torch.from_numpy(y).float() for y in self.ys]

    def _compute_matrices(self):
        bspline = BSplineBasis(self.config.n_basis, (0, self.T), internal_knots=self.config.internal_knots)
        Phis = list(bspline.get_all_matrices(self.ts))
        return Phis

    def __len__(self):
        return self.D

    def __getitem__(self, idx):
        if self.config.dataloader_type == 'iterative':
            return self.X[idx, :], self.Phis[idx], self.ys[idx]
        elif self.config.dataloader_type == 'tensor':
            return self.X[idx, :], self.PHI[idx, :, :], self.Y[idx, :], self.NS[idx]

    def get_collate_fn(self):

        def iterative_collate_fn(batch):
            Xs = []
            ts = []
            ys = []
            for b in batch:
                Xs.append(b[0])
                ts.append(b[1])
                ys.append(b[2])
            return torch.stack(Xs, dim=0), ts, ys

        if self.config.dataloader_type == 'iterative':
            return iterative_collate_fn
        elif self.config.dataloader_type == 'tensor':
            return None

# class _IterativeDataset(torch.utils.data.Dataset, TTSDataset):

#     def __init__(self, config, X, ts, ys, T):
#         """
#         This constructor is helpful when you have irregularly sampled trajectories.
#         Args:
#             config: an instance of the Config class
#             X: numpy array of shape (D,M) where D is the number of samples and M is the number of static features
#             ts: a list of D 1D numpy arrays of shape (N_i,) where T_i is the number of time steps for the i-th sample
#             ys: a list of D 1D numpy arrays of shape (N_i,) where T_i is the number of time steps for the i-th sample
#             T: a real number that is the time horizon
#         """
#         TTSDataset.__init__(config, X, ts, ys, T)

#         for i in range(len(self.ts)):
#             self.ts[i] = np.array(self.ts[i]).ravel()
#             self.ys[i] = torch.from_numpy(np.array(self.ys[i]).ravel()).float()

#         self.Phis = self._compute_matrices()

#     def _compute_matrices(self):
#         bspline = BSplineBasis(self.config.n_basis, (0,self.T))
#         Phis = [torch.from_numpy(Phi).float() for Phi in bspline.get_all_matrices(self.ts)]
#         return Phis

#     def __getitem__(self, idx):
#         return self.X[idx,:], self.Phis[idx], self.ys[idx]

#     def get_collate_fn(self):
#         def iterative_collate_fn(batch):
#             Xs = []
#             ts = []
#             ys = []
#             for b in batch:
#                 Xs.append(b[0])
#                 ts.append(b[1])
#                 ys.append(b[2])
#             return torch.stack(Xs, dim=0), ts, ys
#         return iterative_collate_fn

# class _TensorDataset(torch.utils.data.TensorDataset, TTSDataset):

#     def __init__(self,config,X,ts,ys,T):
#         """
#         This function is helpful when you have irregularly sampled trajectories.
#         Args:
#             config: an instance of the Config class
#             X: numpy array of shape (D,M) where D is the number of samples and M is the number of static features
#             ts: a list of D 1D numpy arrays of shape (N_i,) where T_i is the number of time steps for the i-th sample
#             ys: a list of D 1D numpy arrays of shape (N_i,) where T_i is the number of time steps for the i-th sample
#             T: a real number that is the time horizon
#         Returns:
#             train_loader: a torch.utils.data.DataLoader object
#             val_loader: a torch.utils.data.DataLoader object
#             test_loader: a torch.utils.data.DataLoader object
#         """

#         TTSDataset.__init__(self,config,X,ts,ys,T)

#         bspline = BSplineBasis(config.n_basis, (0,self.T))
#         Phis = [torch.from_numpy(_pad_to_shape(Phi,(self.N_max,config.n_basis))).float() for Phi in bspline.get_all_matrices(self.ts)]

#         for i in range(len(self.ys)):
#             self.ys[i] = torch.from_numpy(_pad_to_shape(self.ys[i],(self.N_max,))).float()

#         torch.utils.data.TensorDataset.__init__(self.X,torch.stack(Phis,dim=0),torch.stack(self.ys,dim=0),torch.tensor(self.Ns))


#     def __init__(self,config,df_static,df_trajectories,T):
#         """
#         This function is helpful when you have a dataframe of irregularly sampled trajectories.
#         Args:
#             config: an instance of the Config class
#             df_static: a pandas dataframe with columns ['id','x1','x2',...,'xM'] corresponding to the static features
#             df_trajectories: a pandas dataframe with columns ['id','t','y']
#             T: a real number that is the time horizon
#         """
#         ids = df_static['id'].unique()
#         X = []
#         ts = []
#         ys = []
#         for id in ids:
#             df_static_id = df_static[df_static['id'] == id]
#             assert df_static_id.shape[0] == 1 # there should be only one row for each id
#             X.append(df_static_id.values[0,:])
#             df_trajectories_id = df_trajectories[df_trajectories['id'] == id].copy()
#             df_trajectories_id.sort_values(by='t',inplace=True)
#             ts.append(df_trajectories_id['t'].values)
#             ys.append(df_trajectories_id['y'].values)

#         X = torch.from_numpy(np.concatenate(X,axis=0)).float()
#         Ns = [ts[i].shape[0] for i in range(len(ts))]
#         N_max = max(Ns)

#         bspline = BSplineBasis(config.n_basis, (0,T))
#         Phis = [torch.from_numpy(_pad_to_shape(Phi,(N_max,config.n_basis))).float() for Phi in bspline.get_all_matrices(ts)]

#         for i in range(len(ys)):
#             ys[i] = torch.from_numpy(_pad_to_shape(ys[i],(N_max,))).float()

#         super().__init__(X,torch.stack(Phis,dim=0),torch.stack(ys,dim=0),torch.tensor(Ns))

#     def __init__(self,config,df,T):
#         """
#         This function is helpful when you have a dataframe of irregularly sampled trajectories.
#         Args:
#             config: an instance of the Config class
#             df a pandas dataframe with columns ['id','x1','x2',...,'xM','t','y'] where the first M columns are the static features and the last two columns are the time and the observation
#             T: a real number that is the time horizon
#         """
#         ids = df['id'].unique()
#         X = []
#         ts = []
#         ys = []
#         for id in ids:
#             df_id = df[df['id'] == id].copy()
#             X.append(df_id.values[0,:])
#             df_id.sort_values(by='t',inplace=True)
#             ts.append(df_id['t'].values)
#             ys.append(df_id['y'].values)

#         X = torch.from_numpy(np.concatenate(X,axis=0)).float()
#         Ns = [ts[i].shape[0] for i in range(len(ts))]
#         N_max = max(Ns)

#         bspline = BSplineBasis(config.n_basis, (0,T))
#         Phis = [torch.from_numpy(_pad_to_shape(Phi,(N_max,config.n_basis))).float() for Phi in bspline.get_all_matrices(ts)]

#         for i in range(len(ys)):
#             ys[i] = torch.from_numpy(_pad_to_shape(ys[i],(N_max,))).float()

#         super().__init__(X,torch.stack(Phis,dim=0),torch.stack(ys,dim=0),torch.tensor(Ns))

def create_dataloader(config, dataset, indices=None, shuffle=True):
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

    if not isinstance(config, Config):
        raise ValueError("config must be an instance of the Config class")
    if not isinstance(dataset, TTSDataset):
        raise ValueError("dataset must be an instance of the TTSDataset class")
    if indices is not None:
        if not isinstance(indices, list):
            raise ValueError("indices must be a list")

    if not isinstance(shuffle, bool):
        raise ValueError("shuffle must be a boolean")

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


def create_train_val_test_dataloaders(config, dataset):
    """
    This function creates the train, validation, and test dataloaders.
    Args:
        config: an instance of the Config class
        dataset: an instance of the TTSDataset class
        seed: an integer that is the random seed
    Returns:
        train_loader: a torch.utils.data.DataLoader object
        val_loader: a torch.utils.data.DataLoader object
        test_loader: a torch.utils.data.DataLoader object
    """

    # slight change to allow this assert to pass when tuning
    if not (isinstance(config, Config) or isinstance(config, TuningConfig)):
        raise ValueError("config must be an instance of the Config class")
    if not isinstance(dataset, TTSDataset):
        raise ValueError("dataset must be an instance of the TTSDataset class")

    # this seeds the dataloaders so that each mini-batch is reproducible
    def seed_worker(worker_id):
        worker_seed=torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_size=int(config.dataset_split.train * len(dataset))
    val_size=int(config.dataset_split.val * len(dataset))
    test_size=len(dataset) - train_size - val_size

    gen=torch.Generator()
    seed=config.seed
    gen.manual_seed(seed)

    train_dataset, val_dataset, test_dataset=torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=gen)

    batch_size=config.training.batch_size

    collate_fn=dataset.get_collate_fn()

    train_dataloader=torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=gen, worker_init_fn=seed_worker, collate_fn=collate_fn)
    val_dataloader=torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, generator=gen, worker_init_fn=seed_worker, collate_fn=collate_fn)
    test_dataloader=torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, generator=gen, worker_init_fn=seed_worker, collate_fn=collate_fn)

    return train_dataloader, val_dataloader, test_dataloader





# def create_dataloaders(config,X,ts,ys,T,seed=0):
#     """
#     This function is helpful when you have irregularly sampled trajectories.
#     Args:
#         config: an instance of the Config class
#         X: numpy array of shape (D,M) where D is the number of samples and M is the number of static features
#         ts: a list of D 1D numpy arrays of shape (N_i,) where T_i is the number of time steps for the i-th sample
#         ys: a list of D 1D numpy arrays of shape (N_i,) where T_i is the number of time steps for the i-th sample
#         T: a real number that is the time horizon
#     Returns:
#         train_loader: a torch.utils.data.DataLoader object
#         val_loader: a torch.utils.data.DataLoader object
#         test_loader: a torch.utils.data.DataLoader object
#     """
