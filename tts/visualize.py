import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider
import numpy as np

def simple_interactive_plot(f, time_horizon, trajectory_range, feature_ranges, n_points=1000):
    """
    f: function of time t and features x that returns a trajectory y, signature f(t, **x)
    time_horizon: the time horizon of the trajectory
    trajectory_range: the range of the trajectory, tuple (y_min, y_max)
    feature_ranges: a dictionary of feature names and their ranges, e.g. {'a': (0, 1), 'b': (0, 2)}
    n_points: number of points to plot
    """
    t = np.linspace(0, time_horizon, n_points)

    # Set up the figure and axes
    fig = plt.figure(figsize=(8, 3))
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