import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider
import numpy as np

def simple_interactive_plot(f, time_horizon, trajectory_range, feature_ranges, n_points=1000, figsize=(8, 3)):
    """
    f: function of time t and features x that returns a trajectory y, signature f(t, **x)
    time_horizon: the time horizon of the trajectory
    trajectory_range: the range of the trajectory, tuple (y_min, y_max)
    feature_ranges: a dictionary of feature names and their ranges, e.g. {'a': (0, 1), 'b': (0, 2)}
    n_points: number of points to plot
    """
    t = np.linspace(0, time_horizon, n_points)

    # Set up the figure and axes
    fig = plt.figure(figsize=figsize)
    line, = plt.plot([], [], lw=2) # initialize the line with empty data

    plt.title("y = tts(t)")
    plt.xlim(0, time_horizon)
    plt.ylim(trajectory_range[0], trajectory_range[1])
    plt.xlabel('t')
    plt.ylabel('y')
    plt.close()
    
    def plot_f(**x):
        y = f(t, **x)
        line.set_data(t, y)
        display(fig)
    
    ## Generate our user interface.
    # Create a dictionary of sliders, one for each feature.
    sliders = {}
    for k, v in feature_ranges.items():
        sliders[k] = FloatSlider(min=v[0], max=v[1], step=0.01, value=v[0])

    interact(plot_f, **sliders);


def simple_tts_plot(litmodel, dataset, trajectory_range, n_points=100, figsize=(8, 3)):
    """
    litmodel: a LitModel object
    dataset: a BaseDataset object
    trajectory_range: the range of the trajectory, tuple (y_min, y_max)
    n_points: number of points to plot
    """
    # Get the time horizon and trajectory range
    time_horizon = litmodel.config.T
    feature_names = dataset.get_feature_names()
    feature_ranges = dataset.get_feature_ranges()

    def trajectory(t, **x):
        features = np.array([x[feature_name] for feature_name in feature_names])
        return litmodel.model.forecast_trajectory(features,t)
    
    simple_interactive_plot(trajectory, time_horizon, trajectory_range, feature_ranges, n_points, figsize=figsize)


def advanced_tts_plot(litmodel, dataset, trajectory_range, n_points=100):
    pass


def template_from_coeffs(coeffs):
    """
    coeffs: a list of coefficients, e.g. [1, 2, 3]
    """
    pass
    