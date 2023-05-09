from scipy.interpolate import UnivariateSpline
import numpy as np
from sklearn.cluster import KMeans


def get_knots_for_single_trajectory(t, y, n_internal_knots, s_guess=None, verbose=False):

    tol_n_knots = 0
    tol_s = 1e-3
    s_lower_bound = 0.0
    s_upper_bound = None

    if s_guess is None:
        s = len(t)
    else:
        s = s_guess

    # First check what is the maximum number of knots that can be found
    found_knots = UnivariateSpline(t, y, s=0).get_knots()
    if n_internal_knots >= len(found_knots):
        return found_knots, 0.0

    for i in range(20):
        if verbose:
            print(f"Trying s={s}")
        found_knots = UnivariateSpline(t, y, s=s).get_knots()
        if verbose:
            print(f"Found {len(found_knots)} knots")
        if np.abs(len(found_knots) - n_internal_knots) <= tol_n_knots:
            return found_knots, s
        elif len(found_knots) < n_internal_knots:
            s_upper_bound = s
            s = (s + s_lower_bound) / 2
        elif len(found_knots) > n_internal_knots:
            s_lower_bound = s
            if s_upper_bound is None:
                s = 2 * s
            else:
                s = (s + s_upper_bound) / 2
        if s_upper_bound is not None:
            if s_upper_bound - s_lower_bound < tol_s:
                tol_n_knots += 1
    
    return found_knots, s


def get_knots_for_trajectories(ts, ys, n_internal_knots, verbose=False):

    found_knots = []
    s = []
    
    for i, (t, y) in enumerate(zip(ts, ys)):
        if verbose:
            print(f"Finding knots for {i+1}-th trajectory")
        if len(s) == 0:
            s_guess = None
        else:
            s_guess = np.mean(s)
        knots, s_ = get_knots_for_single_trajectory(t, y, n_internal_knots, s_guess=s_guess, verbose=verbose)
        found_knots.append(knots)
        s.append(s_)
    return found_knots


def calculate_knot_placement(ts, ys, n_internal_knots, T, seed=0, verbose=False):
    """
    Given a list of trajectories, this function calculates the optimal knot
    placement for all trajectories.
    """
    found_placements = get_knots_for_trajectories(ts, ys, n_internal_knots, verbose=verbose)
    all_knots = np.concatenate(found_placements).reshape(-1,1)
    kmeans = KMeans(n_clusters=n_internal_knots,random_state=seed).fit(all_knots)
    clusters = np.sort(kmeans.cluster_centers_.ravel())
    clusters[0] = 0.0
    clusters[-1] = T
    return clusters

