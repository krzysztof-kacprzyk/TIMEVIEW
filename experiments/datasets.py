from tts.data import BaseDataset
from abc import ABC, abstractmethod
import json
import os
import numpy as np
import pandas as pd

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


def load_dataset(dataset_name, dataset_description_path="dataset_descriptions"):
    dataset_description = load_dataset_description(dataset_name, dataset_description_path=dataset_description_path)
    dataset_builder = dataset_description['dataset_builder']
    dataset_dictionary = dataset_description['dataset_dictionary']
    dataset = get_class_by_name(dataset_builder)(**dataset_dictionary)
    return dataset


def get_class_by_name(class_name):
    """
    This function takes a class name as an argument and returns a python class with this name that is implemented in this module.
    """
    return globals()[class_name]


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

        X, ts, ys = self._extract_data_from_one_dataframe(df)

        subset = self.args.subset
        seed = self.args.seed

        n = len(X)

        gen = np.random.default_rng(seed)
        subset_indices = gen.choice(n, int(n*subset), replace=False)
        subset_indices = [i.item() for i in subset_indices]

        X = X[subset_indices, :]
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

        # Filter only to date 365 (1 year)
        df = df[df['date'] <= 365.0]

        # Filter only to patients with at least 10 time steps
        df = df.groupby('name').filter(lambda x: len(x) >= 10)

        # Take the log transform of the tumor volume
        def protected_log(x):
            return np.log(x + 1e-6)

        df['size'] = protected_log(df['size'])

        # Create a new column with the first time step for each patient
        first_time = df.groupby('name')[['date']].min()

        # Merge with df to extract size at first time step
        first_measurements = first_time.merge(df,left_on=['name','date'],right_on=['name','date'])
        first_measurements.columns = ['name','date_first','size_first']

        # Merge with the original df
        df = df.merge(first_measurements,on='name',how='left')

        df = df[['name','date_first','size_first','date','size']]
        df.columns = ['id','t0','y0','t','y']

        self.X, self.ts, self.ys = self._extract_data_from_one_dataframe(df)


    def get_X_ts_ys(self):
        return self.X, self.ts, self.ys

    def __len__(self):
        return len(self.X)
    
    def get_feature_ranges(self):
        return {
            't0': (1, 150),
            'y0': (-3, 9),
        }
    
    def get_feature_names(self):
        return ['t0','y0']


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
        self.X = X
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


