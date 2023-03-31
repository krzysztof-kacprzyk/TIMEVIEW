from scipy.interpolate import BSpline
import numpy as np

class BSplineBasis():

    def __init__(self,n_basis,t_range):
        self.n_basis = n_basis
        self.t_range = t_range
        self.t_min = t_range[0]
        self.t_max = t_range[1]
        self.k = 3 # degree of the spline
        self.knots = self._calculate_knots()
        self.B_monomial = self._calculate_monomial_basis()

    def _calculate_monomial_basis(self):
        """
        Each bspline basis function is a piecewise polynomial od degree k.
        Thus for each segment (t_j,t_{j+1}) there is an associated polynomial
        a^j_0 + a^j_1 x + a^j_2 x^2 + ... + a^j_k x^k.
        
        This function calculates a matrix for each degree 0,...,k and each admissible index.
        The matrix is of shape ((k+1),len(internal_knots-1)).
        The (i,j) entry of the matrix is the coefficient a^j_i
        """
        N = len(self.knots)

        def get_w_1(i,k):
            """
            Returns two number a, b that correspond to 
            the value of the coefficient w_{i,k} of the form
            ax + b
            """
            if self.knots[i] == self.knots[i+k]:
                return 0, 0
            else:
                denom = self.knots[i+k] - self.knots[i]
                return 1/(denom), -self.knots[i]/denom
            
        def get_w_2(i,k):
            """
            Returns two number a, b that correspond to 
            the value of the coefficient 1 - w_{i,k} of the form
            ax + b
            """
            if self.knots[i] == self.knots[i+k]:
                return 0, 1
            else:
                denom = self.knots[i+k] - self.knots[i]
                return -1/(denom), 1+self.knots[i]/denom
            
        def multiply(values,a,b):
            """
            Returns a matrix of a B-Spline multiplied by ax+b
            """
            k = values.shape[0]
            new_values = np.zeros((k+1,N-1)) # it has to have a bigger 1st dimension becuase we multiply by ax+b
            new_values[0] = values[0] * b
            for j in range(1,k):
                new_values[j] = values[j] * b + values[j-1] * a
            new_values[k] =  values[k-1] * a
            return new_values

        Bs = [] # Bs[k][l] is going to be the l-th B-Spline of degree k

        # We add the 0-degree B-Splines
        B0 = []
        for i in range(N-1):
            coeffs = np.zeros((1,N-1))
            coeffs[0,i] = 1.0
            B0.append(coeffs)
        Bs.append(B0)

        for k in range(1,self.k+1):
            Bk = []
            for i in range(N-1-k):
                coeff = multiply(Bs[k-1][i],*get_w_1(i,k)) + multiply(Bs[k-1][i+1],*get_w_2(i+1,k))
                Bk.append(coeff)
            Bs.append(Bk)

        return Bs


    def _calculate_knots(self):
        n_internal_knots = self.n_basis - self.k + 1
        internal_knots = np.linspace(self.t_min,self.t_max,n_internal_knots)
        self.internal_knots = internal_knots

        return np.r_[[self.t_min]*(self.k),internal_knots,[self.t_max]*(self.k)]


    def get_knots(self):
        return self.knots
    
    def get_internal_knots(self):
        return self.internal_knots

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
    

    def _monomial_basis(self,index, degree=3):
        """
        Each bspline basis function is a piecewise polynomial od degree k.
        Thus for each segment (t_j,t_{j+1}) there is an associated polynomial
        a^j_0 + a^j_1 x + a^j_2 x^2 + ... + a^j_k x^k.
        
        This function returns a matrix of shape ((k+1),len(internal_knots-1)).
        The (i,j) entry of the matrix is the coefficient a^j_i
        """
        return self.B_monomial[degree][index][:,self.k:self.k+len(self.internal_knots)-1]
        
    def get_spline_with_coeffs_monomial(self,coeffs):
        assert len(coeffs) == self.n_basis
        result = np.zeros((self.k+1,len(self.internal_knots)-1))
        for i in range(self.n_basis):
            result += coeffs[i] * self._monomial_basis(i, degree=self.k)
        return result

     
        