from timeview.data import BaseDataset
from abc import ABC, abstractmethod
import json
import os
import numpy as np
import pandas as pd
import glob
from scipy.stats import beta

def save_dataset(dataset_name, dataset_builder, dataset_dictionary, notes="", dataset_description_path="dataset_descriptions"):
    # Check if a dataset description directory exists. If not, create it.
    if not os.path.exists(dataset_description_path):
        os.makedirs(dataset_description_path)
    
    # Check if a dataset description file already exists. If so, raise an error.
    path = os.path.join(dataset_description_path, dataset_name + ".json")
    if os.path.exists(path):
        raise ValueError(f"A dataset description file with this name already exists at {path}.")

    dataset_description = {
        'dataset_name': dataset_name,
        'dataset_builder': dataset_builder,
        'dataset_dictionary': dataset_dictionary,
        'notes': notes
    }
    with open(path, 'w') as f:
        json.dump(dataset_description, f, indent=4)

def load_dataset_description(dataset_name, dataset_description_path="dataset_descriptions"):
    path = os.path.join(dataset_description_path, dataset_name + ".json")
    # Check if a dataset description file exists. If not, raise an error.
    if not os.path.exists(path):
        raise ValueError(f"A dataset description file with this name does not exist at {path}.")

    with open(path, 'r') as f:
        dataset_description = json.load(f)
    return dataset_description


def load_dataset(dataset_name, dataset_description_path="dataset_descriptions", data_folder=None):
    dataset_description = load_dataset_description(dataset_name, dataset_description_path=dataset_description_path)
    dataset_builder = dataset_description['dataset_builder']
    dataset_dictionary = dataset_description['dataset_dictionary']
    if data_folder is not None:
        dataset_dictionary['data_folder'] = data_folder
    dataset = get_class_by_name(dataset_builder)(**dataset_dictionary)
    return dataset


def get_class_by_name(class_name):
    """
    This function takes a class name as an argument and returns a python class with this name that is implemented in this module.
    """
    return globals()[class_name]

class SimpleLinearDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        n_points = 10
        n_samples = 100
        self.X = pd.DataFrame({'x':np.linspace(0,1,n_samples)})
        coeffs = np.random.uniform(-1, 1, size=n_points)
        coeffs[0] = 1
        self.ts = [np.linspace(0,1,n_points) for i in range(n_samples)]
        y0s = [np.random.uniform(-1, 1) for i in range(n_samples)]
        self.ys = [coeffs*y0 for y0 in y0s]

    def get_X_ts_ys(self):
        return self.X, self.ts, self.ys
    
    def __len__(self):
        return len(self.X)
    
    def get_feature_names(self):
        return ['x']
    
    def get_feature_ranges(self):
        return {
            'x': (0, 1)
        } 

class ExponentialDataset(BaseDataset):

    def __init__(self, log_t=False):
        super().__init__(log_t=log_t)
        self.X = pd.DataFrame({'x':np.linspace(0,1,100)})
        self.ts = [np.linspace(0,1,20) for i in range(100)]
        self.ys = [np.exp(t*x) for t, x in zip(self.ts, self.X['x'])]
    
    def get_X_ts_ys(self):
        return self.X, self.ts, self.ys
    
    def __len__(self):
        return len(self.X)
    
    def get_feature_names(self):
        return ['x']
    
    def get_feature_ranges(self):
        return {
            'x': (0, 1)
        }
    
class BetaDataset(BaseDataset):

    def __init__(self, n_samples=100, n_timesteps=20):
        super().__init__(n_samples=n_samples, n_timesteps=n_timesteps)
        n_samples_per_dim = int(np.sqrt(n_samples))
        n_samples = n_samples_per_dim**2
        alphas = np.linspace(1.0,4.0,n_samples_per_dim)
        betas = np.linspace(1.0,4.0,n_samples_per_dim)

        grid = np.meshgrid(alphas, betas)
    
        # stack along the last axis and then reshape into 2 columns
        cart_prod = np.stack(grid, axis=-1).reshape(-1, 2)

        self.X = pd.DataFrame({'alpha':cart_prod[:,0], 'beta':cart_prod[:,1]})
        self.ts = [np.linspace(0,1,n_timesteps) for i in range(len(self.X))]
        self.ys = [np.array([beta.pdf(t,alpha, betap) for t in np.linspace(0,1,n_timesteps)]) for alpha, betap in zip(self.X['alpha'], self.X['beta'])]
    
    def get_X_ts_ys(self):
        return self.X, self.ts, self.ys
    
    def __len__(self):
        return len(self.X)
    
    def get_feature_names(self):
        return ['alpha', 'beta']
    
    def get_feature_ranges(self):
        return {
            'alpha': (1.0, 4.0),
            'beta': (1.0, 4.0)
        }
    
class Exponential2Dataset(BaseDataset):

    def __init__(self, n_samples=100, n_timesteps=20):
        super().__init__()
        self.X = pd.DataFrame({'x':np.linspace(-1,1,n_samples)})
        self.ts = [np.linspace(0,1,n_timesteps) for i in range(n_samples)]
        self.ys = [np.exp((t-1)*x) for t, x in zip(self.ts, self.X['x'])]
    
    def get_X_ts_ys(self):
        return self.X, self.ts, self.ys
    
    def __len__(self):
        return len(self.X)
    
    def get_feature_names(self):
        return ['x']
    
    def get_feature_ranges(self):
        return {
            'x': (-1, 1)
        }

class SineDataset(BaseDataset):

    def __init__(self, n_samples=100, n_timesteps=20):
        super().__init__(n_samples=n_samples, n_timesteps=n_timesteps)
        self.X = pd.DataFrame({'x':np.linspace(-1,1,n_samples)})
        self.ts = [np.linspace(0,1,n_timesteps) for i in range(n_samples)]
        self.ys = [np.sin(t*x*np.pi) for t, x in zip(self.ts, self.X['x'])]
    
    def get_X_ts_ys(self):
        return self.X, self.ts, self.ys
    
    def __len__(self):
        return len(self.X)
    
    def get_feature_names(self):
        return ['x']
    
    def get_feature_ranges(self):
        return {
            'x': (-1, 1)
        }
    
class SineTransDataset(BaseDataset):

    def __init__(self, n_samples=100, n_timesteps=20):
        super().__init__(n_samples=n_samples, n_timesteps=n_timesteps)
        self.X = pd.DataFrame({'x':np.linspace(1.0,3.0,n_samples)})
        self.ts = [np.linspace(0,1,n_timesteps) for i in range(n_samples)]
        self.ys = [np.sin(2*t*np.pi/x) for t, x in zip(self.ts, self.X['x'])]
    
    def get_X_ts_ys(self):
        return self.X, self.ts, self.ys
    
    def __len__(self):
        return len(self.X)
    
    def get_feature_names(self):
        return ['x']
    
    def get_feature_ranges(self):
        return {
            'x': (1, 2.5)
        }

class AirfoilDataset(BaseDataset):

    def __init__(self, log_t=False):
        super().__init__(log_t=log_t)
        df = pd.read_csv('data/airfoil/airfoil_self_noise.dat', sep='\t', header=None)
        df.columns = ['t', 'angle', 'chord', 'velocity', 'thickness', 'y']

        # We need to assign ids
        df['id'] = 0
        prev = 10000000
        curr_id = 0
        for index, row in df.iterrows():
            if row['t'] < prev:
                curr_id += 1
            prev = row['t']
            df.at[index, 'id'] = curr_id
        
        df = df[['id', 'angle', 'chord', 'velocity', 'thickness', 't', 'y']]

        df['t'] = df['t'] / 200 # scale to min 1

        if log_t:
            df['t'] = np.log(df['t']) # now it will be in range 0-4.7
        else:
            df['t'] = df['t'] / 100 # scale to range 0-1

        self.X, self.ts, self.ys = self._extract_data_from_one_dataframe(df)
    
    def get_X_ts_ys(self):
        return self.X, self.ts, self.ys
    
    def __len__(self):
        return len(self.X)
    
    def get_feature_names(self):
        return ['angle', 'chord', 'velocity', 'thickness']
    
    def get_feature_ranges(self):
        return {
            'angle': (0, 22),
            'chord': (0.025, 0.30),
            'velocity': (31, 71),
            'thickness': (0.0004, 0.05)
        }


class CelgeneDataset(BaseDataset):

    def __init__(self):
        super().__init__()
        df = pd.read_csv(os.path.join("data", "celgene", "celgene.csv"))
        df = df[df['t'] <= 365]
        df['t'] = df['t'] / 365 # scale

        df['y'] = df['y'] / 100 # scale
        # Filter out patients with fewer than 3 observations
        df = df.groupby('id').filter(lambda x: len(x) > 4)
        # Filter out duplicate observation times (keep the first one)
        df = df.groupby(['id', 't']).first().reset_index()
        df = df[['id'] + self.get_feature_names() + ['t','y']]


        self.X, self.ts, self.ys = self._extract_data_from_one_dataframe(df)
    
    def get_X_ts_ys(self):
        return self.X, self.ts, self.ys
    
    def __len__(self):
        return len(self.X)
    
    def get_feature_names(self):
        return ['age', 'race', 'ECOG', 'location', 'bmi', 'sysbp', 'diabp']
    
    def get_feature_ranges(self):
        return {
            'age': (50, 85),
            'race': ['White', 'Other', 'Black or African American'],
            'ECOG': [0, 1, 2],
            'location': ['lymph nodes', 'organ or soft tissue'],
            'bmi': (18, 50),
            'sysbp': (85, 180),
            'diabp': (50, 115),
        }

class FLChainDataset(BaseDataset):

    def __init__(self, subset='all'):
        super().__init__(subset=subset)
        df = pd.read_csv(os.path.join("data", "flchain", "flchain.csv"))
        df = df[['id'] + self.get_feature_names() + ['t','y']]
        df['t'] = df['t'] / 5000 # scale
        if subset != 'all':
            all_ids = df['id'].unique()
            # Randomly select a subset of patients
            gen = np.random.default_rng(0)
            ids = gen.choice(all_ids, size=subset, replace=False)
            df = df[df['id'].isin(ids)]

        self.X, self.ts, self.ys = self._extract_data_from_one_dataframe(df)
    
    def get_X_ts_ys(self):
        return self.X, self.ts, self.ys
    
    def __len__(self):
        return len(self.X)
    
    def get_feature_names(self):
        return ['age', 'sex', 'creatinine', 'kappa', 'lambda', 'flc.grp', 'mgus']
    
    def get_feature_ranges(self):
        return {
            'age': (50, 100),
            'sex': ['M', 'F'],
            'creatinine': (0.4, 2.0),
            'kappa': (0.01, 5.0),
            'lambda': (0.04, 5.0),
            'flc.grp': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'mgus': ['no', 'yes']
        }


class StressStrainDataset(BaseDataset):

    def __init__(self, lot='all', include_lot_as_feature=False, more_samples=0, downsample=True, specimen='all', max_strain=0.3):
        super().__init__(lot=lot, include_lot_as_feature=include_lot_as_feature, more_samples=more_samples, downsample=downsample, specimen=specimen, max_strain=max_strain)
        if lot == 'all':
            path = os.path.join("data", "stress-strain-curves", "T*.csv")
        else:
            path = os.path.join("data", "stress-strain-curves", f"T*{lot}*.csv")
        filenames = glob.glob(path)

        dfs = []
        for filename in filenames:
            df = pd.read_csv(filename)
            dataset_name = filename.split('T_')[1].split('.csv')[0]
            parts = dataset_name.split('_')
            temp = parts[0]
            lot = parts[1]
            specimen = parts[2]
            if (self.args.specimen != 'all') and (int(specimen) != self.args.specimen):
                continue
            df.columns = ['t', 'y']
            df.drop(df.tail(1).index,inplace=True) # drop last row because it's an outlier
            if downsample:
                df = df.iloc[::3,:] # downsample
            df['temp'] = float(temp)
            if include_lot_as_feature:
                df['lot'] = lot
            df['id'] = lot + '_' + specimen + '_' + temp
            df.reset_index(inplace=True)
            if more_samples > 0:
                for ind in range(more_samples):
                    df.loc[ind::more_samples, 'id'] = df.loc[ind::more_samples, 'id'] + '_' + str(ind)
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        if include_lot_as_feature:
            df = df[['id', 'temp', 'lot', 't', 'y']]
        else:
            df = df[['id', 'temp', 't', 'y']]
        df.drop(index=df[df['t'] < 0].index, inplace=True) # drop rows where t is < 0
        df.drop(index=df[df['y'] < 0].index, inplace=True) # drop rows where y is < 0
        df.drop(index=df[df['t'] > max_strain].index, inplace=True) # drop rows where t is > max_strain
        df['y'] = df['y'] / 300 # scale

        self.X, self.ts, self.ys = self._extract_data_from_one_dataframe(df)

    def get_X_ts_ys(self):
        return self.X, self.ts, self.ys
    
    def __len__(self):
        return len(self.X)

    def get_feature_names(self):
        if self.args.include_lot_as_feature:
            return ['temp', 'lot']
        else:
            return ['temp']
    
    def get_feature_ranges(self):
        if self.args.include_lot_as_feature:
            return {'temp': (20, 300), 'lot': ['A','B','C','D','E','F','G','H', 'I']}
        else:
            return {'temp': (20, 300)}


class TacrolimusDataset(BaseDataset):

    def __init__(self, granularity, normalize=False, max_t=25, data_folder='data'):
        super().__init__(granularity=granularity, normalize=normalize)
        if granularity == 'visit':

            df = pd.read_csv(os.path.join(data_folder, "tacrolimus", "tac_pccp_mr4_250423.csv"))
            dosage_rows = df[df['DOSE'] != 0]
            assert dosage_rows['visit_id'].is_unique
            df.drop(columns=['DOSE', 'EVID','II', 'AGE'], inplace=True) # we drop age because many missing values. the other columns are not needed
            df.drop(index=dosage_rows.index, inplace=True) # drop dosage rows
            # Merge df with dosage rows on visit_id
            df = df.merge(dosage_rows[['visit_id', 'DOSE']], on='visit_id', how='left') # add dosage as a feature
            df.loc[df['TIME'] >= 168, 'TIME'] -= 168 # subtract 168 from time to get time since last dosage
            missing_24h = df[(df['TIME'] == 0) & (df['DV'] == 0)].index
            df.drop(index=missing_24h, inplace=True) # drop rows where DV is 0 and time is 0 - they correspond to missing 24h measurements

            dv_0 = df[df['TIME'] == 0][['visit_id', 'DV']]
            assert dv_0['visit_id'].is_unique
            df = df.merge(dv_0, on='visit_id', how='left', suffixes=('', '_0')) # add DV_0 as a feature

            more_than_t = df[df['TIME'] > max_t].index
            df.drop(index=more_than_t, inplace=True) # drop rows where time is greater than max_t

            df.dropna(inplace=True) # drop rows with missing values

            df = df[['visit_id'] + ['DOSE', 'DV_0', 'SEXE', 'POIDS', 'HT', 'HB', 'CREAT', 'CYP', 'FORMULATION'] + ['TIME', 'DV']]

            df.columns = ['id'] + self.get_feature_names() + ['t', 'y']

            X, ts, ys = self._extract_data_from_one_dataframe(df)

            if normalize:
                # Make each column of X between 0 and 1
                X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
            
            self.X, self.ts, self.ys = X, ts, ys
        else:
            raise NotImplementedError("Only visit granularity is implemented for this dataset.")
        

    def get_X_ts_ys(self):
        return self.X, self.ts, self.ys
    
    def __len__(self):
        return len(self.X)
    
    def get_feature_names(self):
        return ['DOSE', 'DV_0', 'SEX', 'WEIGHT', 'HT', 'HB', 'CREAT', 'CYP', 'FORM']
    
    def get_feature_ranges(self):
        if self.args.normalize:
            return {
                'DOSE': (0, 1),
                'DV_0': (0, 1),
                'SEX': [0,1],
                'WEIGHT': (0, 1),
                'HT': (0, 1),
                'HB': (0, 1),
                'CREAT': (0, 1),
                'CYP': [0, 1],
                'FORM': [0, 1]
            }
        else:
            return {
                'DOSE': (0, 10),
                'DV_0': (0, 20),
                'SEX': [0, 1],
                'WEIGHT': (45, 110),
                'HT': (20, 47),
                'HB': (6, 16),
                'CREAT': (60, 830),
                'CYP': [0, 1],
                'FORM': [0, 1]
            }





class WindDataset(BaseDataset):

    def __init__(self, company, granularity='daily', rolling=False):
        super().__init__(company=company, granularity=granularity, rolling=rolling)

        files = {
            "50Hertz": "50Hertz.csv",
            "Amprion": "Amprion.csv",
            "TenneTTSO": "TenneTTSO.csv",
            "TransnetBW": "TransnetBW.csv"
        }

        def load_company(name):
            file_path = os.path.join("data", "wind_data", files[name])
            df = pd.read_csv(file_path,index_col=0,parse_dates=True,dayfirst=True)

            if self.args.rolling == True:
                df = df.rolling(window=7).mean()
            elif type(self.args.rolling) == int:
                df = df.rolling(window=self.args.rolling, center=True).mean()

            df = df.loc['2019-09-01':'2020-08-31',:].copy()
            df['id'] = [f"{name}{id}" for id in list(range(len(df)))]
            df['day_number'] = list(range(len(df)))
            df['month'] = df.index.month
            df = df.melt(id_vars=['id','day_number','month'],var_name='time', value_name='y')
            df['t'] = pd.to_timedelta(df['time']).dt.total_seconds() / 3600
            df.drop(columns=['time'],inplace=True)

            # Standardize the data between 0 and 1
            # df['y'] = (df['y'] - df['y'].min()) / (df['y'].max() - df['y'].min())
            # Normalize the data so that the mean is 0 and the standard deviation is 1
            df['y'] = (df['y'] - df['y'].mean()) / df['y'].std()
            df['day_number'] = df['day_number'] / 365

            return df


        if company == 'all':
            df = pd.concat([load_company(name) for name in files.keys()])
        else:
            df = load_company(company)

        if self.args.granularity == 'daily':
            df = df[['id','day_number','t','y']]
        elif self.args.granularity == 'monthly':
            df = df[['id','month','t','y']]

        X, ts, ys = self._extract_data_from_one_dataframe(df)
        self.X, self.ts, self.ys = X, ts, ys

    def get_X_ts_ys(self):
        return self.X, self.ts, self.ys
    
    def __len__(self):
        return len(self.X)
    
    def get_feature_names(self):
        if self.args.granularity == 'daily':
            return ['day_number']
        elif self.args.granularity == 'monthly':
            return ['month']

    def get_feature_ranges(self):
        if self.args.granularity == 'daily':
            return {
                'day_number': (0, 1)
            }
        elif self.args.granularity == 'monthly':
            return {
                'month': (1, 12)
            }

class MIMICDataset(BaseDataset):

    def __init__(self, subset=0.1, seed=0):
        super().__init__(subset=subset,seed=seed)
        df = pd.read_csv(os.path.join("data", "mimic", "processed_sepsis3_tts.csv"))

        selected_cols = [
                'traj',
                'o:gender', 'o:mechvent', 'o:re_admission', 'o:age',
                'o:Weight_kg', 'o:GCS', 'o:HR', 'o:SysBP', 'o:MeanBP', 'o:DiaBP',
                'o:RR', 'o:Temp_C', 'o:FiO2_1', 'o:Potassium', 'o:Sodium', 'o:Chloride',
                'o:Glucose', 'o:Magnesium', 'o:Calcium', 'o:Hb', 'o:WBC_count',
                'o:Platelets_count', 'o:PTT', 'o:PT', 'o:Arterial_pH', 'o:paO2',
                'o:paCO2', 'o:Arterial_BE', 'o:HCO3', 'o:Arterial_lactate', 'o:SOFA',
                'o:SIRS', 'o:Shock_Index', 'o:PaO2_FiO2', 'o:cumulated_balance',
                'o:SpO2', 'o:BUN', 'o:Creatinine', 'o:SGOT', 'o:SGPT', 'o:Total_bili',
                'o:INR', 'a:action',
                'step', 'true_score'] 

        df = df[selected_cols]

        df.columns = ['id'] + df.columns[1:-2].tolist() + ['t','y']

        # Filter out patients with less than 5 observations
        df = df.groupby('id').filter(lambda x: len(x) > 4)

        X, ts, ys = self._extract_data_from_one_dataframe(df)

        subset = self.args.subset
        seed = self.args.seed

        n = len(X)

        gen = np.random.default_rng(seed)
        subset_indices = gen.choice(n, int(n*subset), replace=False)
        subset_indices = [i.item() for i in subset_indices]

        X = X.iloc[subset_indices, :]
        ts = [ts[i] for i in subset_indices]
        ys = [ys[i] for i in subset_indices]

        self.X = X
        self.ts = ts
        self.ys = ys

    def get_X_ts_ys(self):
        return self.X, self.ts, self.ys

    def __len__(self):
        return len(self.X)
    
    def get_feature_ranges(self):
        return {
            'o:gender': (0,1),
            'o:mechvent': (0,1), 
            'o:re_admission': (0,1), 
            'o:age': (0,1),
            'o:Weight_kg': (0,1), 
            'o:GCS': (0,1), 
            'o:HR': (0,1), 
            'o:SysBP': (0,1), 
            'o:MeanBP': (0,1), 
            'o:DiaBP': (0,1),
            'o:RR': (0,1), 
            'o:Temp_C': (0,1), 
            'o:FiO2_1': (0,1), 
            'o:Potassium': (0,1), 
            'o:Sodium': (0,1), 
            'o:Chloride': (0,1),
            'o:Glucose': (0,1), 
            'o:Magnesium': (0,1), 
            'o:Calcium': (0,1), 
            'o:Hb': (0,1), 
            'o:WBC_count': (0,1),
            'o:Platelets_count': (0,1), 
            'o:PTT': (0,1), 
            'o:PT': (0,1), 
            'o:Arterial_pH': (0,1), 
            'o:paO2': (0,1),
            'o:paCO2': (0,1), 
            'o:Arterial_BE': (0,1), 
            'o:HCO3': (0,1), 
            'o:Arterial_lactate': (0,1), 
            'o:SOFA': (0,1),
            'o:SIRS': (0,1), 
            'o:Shock_Index': (0,1), 
            'o:PaO2_FiO2': (0,1), 
            'o:cumulated_balance': (0,1),
            'o:SpO2': (0,1), 
            'o:BUN': (0,1), 
            'o:Creatinine': (0,1), 
            'o:SGOT': (0,1), 
            'o:SGPT': (0,1), 
            'o:Total_bili': (0,1),
            'o:INR': (0,1), 
            'a:action': (0,1),
        }

    def get_feature_names(self):
        return [
            'o:gender',
            'o:mechvent', 
            'o:re_admission', 
            'o:age',
            'o:Weight_kg', 
            'o:GCS', 
            'o:HR', 
            'o:SysBP', 
            'o:MeanBP', 
            'o:DiaBP',
            'o:RR', 
            'o:Temp_C', 
            'o:FiO2_1', 
            'o:Potassium', 
            'o:Sodium', 
            'o:Chloride',
            'o:Glucose', 
            'o:Magnesium', 
            'o:Calcium', 
            'o:Hb', 
            'o:WBC_count',
            'o:Platelets_count', 
            'o:PTT', 
            'o:PT', 
            'o:Arterial_pH', 
            'o:paO2',
            'o:paCO2', 
            'o:Arterial_BE', 
            'o:HCO3', 
            'o:Arterial_lactate', 
            'o:SOFA',
            'o:SIRS', 
            'o:Shock_Index', 
            'o:PaO2_FiO2', 
            'o:cumulated_balance',
            'o:SpO2', 
            'o:BUN', 
            'o:Creatinine', 
            'o:SGOT', 
            'o:SGPT', 
            'o:Total_bili',
            'o:INR', 
            'a:action'
        ]




class TumorDataset(BaseDataset):

    FILE_LIST = [
        "input celgene09.csv",
        "input centoco06.csv",
        "input cougar06.csv",
        "input novacea06.csv",
        "input pfizer08.csv",
        "input sanfi00.csv",
        "input sanofi79.csv",
        "inputS83OFF.csv",
        "inputS83ON.csv",
    ]

    def __init__(self, **args):
        super().__init__(**args)
        df_list = list()
        for f in TumorDataset.FILE_LIST:
            df = pd.read_csv(os.path.join("data",'tumor',f))
            df["name"] = df["name"].astype(str) + f
            df_list.append(df)

        df = pd.concat(df_list)

        # Filter out duplicate observation times (keep the first one)
        df = df.groupby(['name', 'date']).first().reset_index()

        # Take the log transform of the tumor volume
        def protected_log(x):
            return np.log(x + 1e-6)

        df['size'] = protected_log(df['size'])

        first_time = df.groupby('name')[['date']].min()
        first_measurements = pd.merge(first_time, df[['name','date','size']], on=['name', 'date'])
        df = pd.merge(df, first_measurements, on='name', suffixes=('', '_first'))

        df['t'] = df['date'] - df['date_first']

        # Filter only to date 365 (1 year)
        df = df[df['t'] <= 365.0]

        # Filter only to patients with at least 10 time steps
        df = df.groupby('name').filter(lambda x: len(x) >= 10)

        df['t'] = df['t'] / 365.0


        df = df[['name','size_first','t','size']]
        df.columns = ['id','y0','t','y']

        self.X, self.ts, self.ys = self._extract_data_from_one_dataframe(df)


    def get_X_ts_ys(self):
        return self.X, self.ts, self.ys

    def __len__(self):
        return len(self.X)
    
    def get_feature_ranges(self):
        return {
            'y0': (-3, 8),
        }
    
    def get_feature_names(self):
        return ['y0']


class SyntheticTumorDataset(BaseDataset):

    def __init__(self, **args):
        super().__init__(**args)
        X, ts, ys = SyntheticTumorDataset.synthetic_tumor_data(
                        n_samples = self.args.n_samples,
                        n_time_steps = self.args.n_time_steps,
                        time_horizon = self.args.time_horizon,
                        noise_std = self.args.noise_std,
                        seed = self.args.seed,
                        equation = self.args.equation)
        if self.args.equation == "wilkerson":
            self.X = pd.DataFrame(X, columns=["age", "weight", "initial_tumor_volume", "dosage"])
        elif self.args.equation == "geng":
            self.X = pd.DataFrame(X, columns=["age", "weight", "initial_tumor_volume", "start_time", "dosage"])

        self.ts = ts
        self.ys = ys
    
    def get_X_ts_ys(self):
        return self.X, self.ts, self.ys

    def __len__(self):
        return len(self.X)
    
    def get_feature_ranges(self):
        if self.args.equation == "wilkerson":
            return {
                "age": (20, 80),
                "weight": (40, 100),
                "initial_tumor_volume": (0.1, 0.5),
                "dosage": (0.0, 1.0)
                }
        elif self.args.equation == "geng":
            return {
                "age": (20, 80),
                "weight": (40, 100),
                "initial_tumor_volume": (0.1, 0.5),
                "start_time": (0.0, 1.0),
                "dosage": (0.0, 1.0)
                }

    def get_feature_names(self):
        if self.args.equation == "wilkerson":
            return ["age", "weight", "initial_tumor_volume", "dosage"]
        elif self.args.equation == "geng":
            return ["age", "weight", "initial_tumor_volume", "start_time", "dosage"]



    def _tumor_volume(t, age, weight, initial_tumor_volume, start_time, dosage):
        """
        Computes the tumor volume at times t based on the tumor model under chemotherapy described in the paper.

        Args:
            t: numpy array of real numbers that are the times at which to compute the tumor volume
            age: a real number that is the age
            weight: a real number that is the weight
            initial_tumor_volume: a real number that is the initial tumor volume
            start_time: a real number that is the start time of chemotherapy
            dosage: a real number that is the chemotherapy dosage
        Returns:
            Vs: numpy array of real numbers that are the tumor volumes at times t
        """

        RHO_0=2.0

        K_0=1.0
        K_1=0.01

        BETA_0=50.0

        GAMMA_0=5.0

        V_min=0.001

        # Set the parameters of the tumor model
        rho=RHO_0 * (age / 20.0) ** 0.5
        K=K_0 + K_1 * (weight)
        beta=BETA_0 * (age/20.0) ** (-0.2)

        # Create chemotherapy function
        def C(t):
            return np.where(t < start_time, 0.0, dosage * np.exp(- GAMMA_0 * (t - start_time)))

        def dVdt(V, t):
            """
            This is the tumor model under chemotherapy.
            Args:
                V: a real number that is the tumor volume
                t: a real number that is the time
            Returns:
                dVdt: a real number that is the rate of change of the tumor volume
            """

            dVdt=rho * (V-V_min) * V * np.log(K / V) - beta * V * C(t)

            return dVdt

        # Integrate the tumor model
        V=odeint(dVdt, initial_tumor_volume, t)[:, 0]
        return V


    def _tumor_volume_2(t, age, weight, initial_tumor_volume, dosage):
        """
        Computes the tumor volume at times t based on the tumor model under chemotherapy described in the paper.

        Args:
            t: numpy array of real numbers that are the times at which to compute the tumor volume
            age: a real number that is the age
            weight: a real number that is the weight
            initial_tumor_volume: a real number that is the initial tumor volume
            start_time: a real number that is the start time of chemotherapy
            dosage: a real number that is the chemotherapy dosage
        Returns:
            Vs: numpy array of real numbers that are the tumor volumes at times t
        """

        G_0=2.0
        D_0=180.0
        PHI_0=10

        # Set the parameters of the tumor model
        # rho = RHO_0 * (age / 20.0) ** 0.5
        # K = K_0 + K_1 * (weight)
        # beta = BETA_0 * (age/20.0) ** (-0.2)

        g=G_0 * (age / 20.0) ** 0.5
        d=D_0 * dosage/weight
        # sigmoid function
        phi=1 / (1 + np.exp(-dosage*PHI_0))

        return initial_tumor_volume * (phi*np.exp(-d * t) + (1-phi)*np.exp(g * t))


    def _get_tumor_feature_ranges(*feautures):
        """
        Gets the ranges of the tumor features.

        Args:
            feautures: a list of strings that are the tumor features
        Returns:
            ranges: a dictionary that maps the tumor features to their ranges
        """

        ranges={}
        for feature in feautures:
            if feature in TUMOR_DATA_FEATURE_RANGES:
                ranges[feature]=TUMOR_DATA_FEATURE_RANGES[feature]
            else:
                raise ValueError(f"Invalid tumor feature: {feature}")
        return ranges



    def synthetic_tumor_data(n_samples,  n_time_steps, time_horizon=1.0, noise_std=0.0, seed=0, equation="wilkerson"):
        """
        Creates synthetic tumor data based on the tumor model under chemotherapy described in the paper.

        We have five static features:
            1. age
            2. weight
            3. initial tumor volume
            4. start time of chemotherapy (only for Geng et al. model)
            5. chemotherapy dosage

        Args:
            n_samples: an integer that is the number of samples
            noise_std: a real number that is the standard deviation of the noise
            seed: an integer that is the random seed
        Returns:
            X: a numpy array of shape (n_samples, 4)
            ts: a list of n_samples 1D numpy arrays of shape (n_time_steps,)
            ys: a list of n_samples 1D numpy arrays of shape (n_time_steps,)
        """
        TUMOR_DATA_FEATURE_RANGES={
            "age": (20, 80),
            "weight": (40, 100),
            "initial_tumor_volume": (0.1, 0.5),
            "start_time": (0.0, 1.0),
            "dosage": (0.0, 1.0)
        }

        # Create the random number generator
        gen=np.random.default_rng(seed)

        # Sample age
        age=gen.uniform(
            TUMOR_DATA_FEATURE_RANGES['age'][0], TUMOR_DATA_FEATURE_RANGES['age'][1], size=n_samples)
        # Sample weight
        weight=gen.uniform(
            TUMOR_DATA_FEATURE_RANGES['weight'][0], TUMOR_DATA_FEATURE_RANGES['weight'][1], size=n_samples)
        # Sample initial tumor volume
        tumor_volume=gen.uniform(TUMOR_DATA_FEATURE_RANGES['initial_tumor_volume']
                                [0], TUMOR_DATA_FEATURE_RANGES['initial_tumor_volume'][1], size=n_samples)
        # Sample start time of chemotherapy
        start_time=gen.uniform(
            TUMOR_DATA_FEATURE_RANGES['start_time'][0], TUMOR_DATA_FEATURE_RANGES['start_time'][1], size=n_samples)
        # Sample chemotherapy dosage
        dosage=gen.uniform(
            TUMOR_DATA_FEATURE_RANGES['dosage'][0], TUMOR_DATA_FEATURE_RANGES['dosage'][1], size=n_samples)

        # Combine the static features into a single array
        if equation == "wilkerson":
            X=np.stack((age, weight, tumor_volume, dosage), axis=1)
        elif equation == "geng":
            X=np.stack((age, weight, tumor_volume, start_time, dosage), axis=1)

        # Create the time points
        ts=[np.linspace(0.0, time_horizon, n_time_steps)
            for i in range(n_samples)]

        # Create the tumor volumes
        ys=[]

        for i in range(n_samples):

            # Unpack the static features
            if equation == "wilkerson":
                age, weight, tumor_volume, dosage=X[i, :]
            elif equation == "geng":
                age, weight, tumor_volume, start_time, dosage=X[i, :]

            if equation == "wilkerson":
                ys.append(SyntheticTumorDataset._tumor_volume_2(
                    ts[i], age, weight, tumor_volume, dosage))
            elif equation == "geng":
                ys.append(SyntheticTumorDataset._tumor_volume(ts[i], age, weight,
                        tumor_volume, start_time, dosage))

            # Add noise to the tumor volumes
            ys[i] += gen.normal(0.0, noise_std, size=n_time_steps)

        return X, ts, ys


