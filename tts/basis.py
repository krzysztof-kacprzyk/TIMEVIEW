from scipy.interpolate import BSpline
import numpy as np

class BSplineBasis():

    def __init__(self,n_basis,t_range):
        self.n_basis = n_basis
        self.t_range = t_range
        self.t_min = t_range[0]
        self.t_max = t_range[1]
        self.k = 3 # degree of the spline
        self.knots = self.get_knots()

    def get_knots(self):
        n_internal_knots = self.n_basis - self.k - 1
        internal_knots = np.linspace(self.t_min,self.t_max,n_internal_knots+2)[1:-1]

        return np.r_[[self.t_min]*(self.k+1),internal_knots,[self.t_max]*(self.k+1)]

    def get_basis(self,index):
        assert index >= 0 and index < self.n_basis

        # Create a vector of length n_basis with 1 at index and 0 elsewhere
        coeffs = np.zeros(self.n_basis)
        coeffs[index] = 1

        # Create a BSpline object with the basis vector as the coefficients
        return BSpline(self.knots,coeffs,self.k)
    

    def get_matrix(self,t):
        t = np.array(t)
        assert t.ndim == 1
        assert t.min() >= self.t_min and t.max() <= self.t_max
        N = len(t)
        B = np.zeros((N,self.n_basis))
        for b in range(self.n_basis):
            basis = self.get_basis(b)   
            B[:,b] = basis(t)

        return B
    
    def get_all_matrices(self,ts):
        for t in ts:
            yield self.get_matrix(t)

    def get_spline_with_coeffs(self,coeffs):
        assert len(coeffs) == self.n_basis
        return BSpline(self.knots,coeffs,self.k)