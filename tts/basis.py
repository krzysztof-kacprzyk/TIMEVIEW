from scipy.interpolate import BSpline
import numpy as np

class BSplineBasis():

    def __init__(self,n_basis,t_range, internal_knots=None):
        self.n_basis = n_basis
        self.t_range = t_range
        self.t_min = t_range[0]
        self.t_max = t_range[1]
        self.k = 3 # degree of the spline

        if internal_knots is None:
            self.internal_knots = self._calculate_internal_knots()
        else:
            assert len(internal_knots) == self.n_basis - self.k + 1
            self.internal_knots = internal_knots
        
        self.knots = self._add_boundary_knots(self.internal_knots)
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

    def _add_boundary_knots(self,internal_knots):
        """
        Given a list of internal knots, this function adds the boundary knots
        """
        return np.r_[[self.t_min]*(self.k),internal_knots,[self.t_max]*(self.k)]

    def _calculate_internal_knots(self):
        n_internal_knots = self.n_basis - self.k + 1
        internal_knots = np.linspace(self.t_min,self.t_max,n_internal_knots)
        return internal_knots


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

    def get_template_from_coeffs(self,coeffs):
        monomial_basis_matrix = self.get_spline_with_coeffs_monomial(coeffs)
        full_template = []
        all_transition_points = []
        for i in range(len(self.internal_knots)-1):
            template, transition_points = self.get_template_from_cubic(monomial_basis_matrix[:,i],(self.internal_knots[i],self.internal_knots[i+1]))
            full_template += template
            all_transition_points += transition_points[:-1]
        all_transition_points.append(self.t_max)

        short_template = []
        short_transition_points = []

        short_template.append(full_template[0])
        short_transition_points.append(all_transition_points[0])

        for state, point in zip(full_template[1:], all_transition_points[1:-1]):
            if state != short_template[-1]:
                short_template.append(state)
                short_transition_points.append(point)
        short_transition_points.append(all_transition_points[-1])

        return short_template, short_transition_points
            
    STATES = {
        'line_increasing': 0,
        'line_decreasing': 1,
        'line_constant': 2,
        'convex_increasing': 3,
        'concave_increasing': 4,
        'convex_decreasing': 5,
        'concave_decreasing': 6,
    }


    def get_template_from_cubic(self,coeffs, interval):
        """
        Args:
            coeffs: a vector of length 4. The first entry is the constant term, the second is the linear term, etc.
            interval: a tuple (t_min,t_max) specifying the interval on which the cubic is defined
        """
        assert len(coeffs) == 4

        t_0 = interval[0]
        t_1 = interval[1]

        if coeffs[3] == 0 and coeffs[2] == 0:
            # This is a straight line
            if coeffs[1] > 0:
                return [self.STATES['line_increasing']], [t_0,t_1]
            elif coeffs[1] < 0:
                return [self.STATES['line_decreasing']], [t_0,t_1]
            else:
                return [self.STATES['line_constant']], [t_0,t_1]
        elif coeffs[3] == 0:
            # This is a parabola
            
            # Calculate the vertex
            t_vertex = -coeffs[1]/(2*coeffs[2])

            if t_vertex <= t_0 or t_vertex >= t_1:
                # The vertex is not in the interval

                if coeffs[2] > 0:
                    # The parabola is concave up (convex)
                    if t_vertex <= t_0:
                        # The parabola is increasing
                        return [self.STATES['convex_increasing']], [t_0,t_1]
                    elif t_vertex >= t_1:
                        # The parabola is decreasing
                        return [self.STATES['convex_decreasing']], [t_0,t_1]
                
                elif coeffs[2] < 0:
                    # The parabola is concave down (concave)
                    if t_vertex <= t_0:
                        # The parabola is decreasing
                        return [self.STATES['concave_decreasing']], [t_0,t_1]
                    elif t_vertex >= t_1:
                        # The parabola is increasing
                        return [self.STATES['concave_increasing']], [t_0,t_1]
            
            else:
                # The vertex is in the interval
                if coeffs[2] > 0:
                    # The parabola is concave up (convex)
                    return [self.STATES['convex_decreasing'], self.STATES['convex_increasing']], [t_0,t_vertex,t_1]
                elif coeffs[2] < 0:
                    # The parabola is concave down (concave)
                    return [self.STATES['concave_increasing'], self.STATES['concave_decreasing']], [t_0,t_vertex,t_1]
        else:
            # This is a cubic

             # Calculate the coefficients of the first derivative
            coeffs_dt = np.zeros(4)
            for i in range(1,4):
                coeffs_dt[i-1] = i * coeffs[i]
            
            # Calculate the coefficients of the second derivative
            coeffs_dt2 = np.zeros(4)
            for i in range(1,4):
                coeffs_dt2[i-1] = i * coeffs_dt[i]

            # Find zeroes of the first derivative, it is a quadratic with a non-zero leading coefficient (because it's a cubic)
            dt_zeros = []
            Delta = coeffs_dt[1]**2 - 4 * coeffs_dt[2] * coeffs_dt[0]
            if Delta < 0:
                pass
            elif Delta == 0:
                dt_zeros.append(-coeffs_dt[1]/(2*coeffs_dt[2]))
            else:
                dt_zeros.append((-coeffs_dt[1] + np.sqrt(Delta))/(2*coeffs_dt[2]))
                dt_zeros.append((-coeffs_dt[1] - np.sqrt(Delta))/(2*coeffs_dt[2]))

            # Find zero of the second derivative, it is a linear function with a non-zero slope (because it's a cubic)
            dt2_zero = - coeffs_dt2[0]/coeffs_dt2[1] # this is the inlfection point

            if dt2_zero <= t_0 or dt2_zero >= t_1:
                # The inflection point is not in the interval so we it has a constant convexity

                zeros_in_interval = [z for z in dt_zeros if (t_0 < z) and (z < t_1)]

                if (dt2_zero <= t_0 and coeffs_dt2[1] > 0) or (dt2_zero >= t_1 and coeffs_dt2[1] < 0):
                    # The cubic is concave up (convex)
                    # Now we need to check zeros of the first derivative

                    if len(zeros_in_interval) == 0:
                        # The cubic does not change direction
                        # Evaluate the first derivative at the center of the interval
                        t_center = (t_0 + t_1)/2
                        dt_center = coeffs_dt[0] + coeffs_dt[1] * t_center + coeffs_dt[2] * t_center**2
                        if dt_center > 0:
                            # The cubic is increasing
                            return [self.STATES['convex_increasing']], [t_0,t_1]
                        elif dt_center < 0:
                            # The cubic is decreasing
                            return [self.STATES['convex_decreasing']], [t_0,t_1]
                        else:
                            # Cannot happen because there are no zeros in the interval
                            pass
                    elif len(zeros_in_interval) == 1:
                        # The cubic changes direction once
                        return [self.STATES['convex_decreasing'], self.STATES['convex_increasing']], [t_0,zeros_in_interval[0],t_1]
                    else:
                        # Cannot happen because there is not inflection point in the interval
                        pass

                elif (dt2_zero <= t_0 and coeffs_dt2[1] < 0) or (dt2_zero >= t_1 and coeffs_dt2[1] > 0):
                    # The cubic is concave down (concave)
                    # Now we need to check zeros of the first derivative

                    if len(zeros_in_interval) == 0:
                        # The cubic does not change direction
                        # Evaluate the first derivative at the center of the interval
                        t_center = (t_0 + t_1)/2
                        dt_center = coeffs_dt[0] + coeffs_dt[1] * t_center + coeffs_dt[2] * t_center**2
                        if dt_center < 0:
                            # The cubic is decreasing
                            return [self.STATES['concave_decreasing']], [t_0,t_1]
                        elif dt_center > 0:
                            # The cubic is increasing
                            return [self.STATES['concave_increasing']], [t_0,t_1]
                        else:
                            # Cannot happen because there are no zeros in the interval
                            pass
                    elif len(zeros_in_interval) == 1:
                        # The cubic changes direction once
                        return [self.STATES['concave_increasing'], self.STATES['concave_decreasing']], [t_0,zeros_in_interval[0],t_1]
                    else:
                        # Cannot happen because there is not inflection point in the interval
                        pass
                
                else:
                    # Cannot happen
                    pass

            else:
                # The inflection point is in the interval so it has a non-constant convexity
                zeros_in_interval = sorted([z for z in dt_zeros if (t_0 < z) and (z < t_1)])

                if len(zeros_in_interval) == 0:

                    # Calculate the value of the first derivative at the inflection point
                    dt_inflection = coeffs_dt[0] + coeffs_dt[1] * dt2_zero + coeffs_dt[2] * dt2_zero**2

                    if coeffs_dt2[1] > 0:
                        if dt_inflection < 0:
                            return [self.STATES['concave_decreasing'], self.STATES['convex_decreasing']], [t_0,dt2_zero,t_1]
                        elif dt_inflection > 0:
                            return [self.STATES['concave_increasing'], self.STATES['convex_increasing']], [t_0,dt2_zero,t_1]
                        else:
                            # The first derivative is tangent to the x-axis at the inflection point
                            if coeffs_dt[2] > 0:
                                return [self.STATES['concave_increasing'], self.STATES['convex_increasing']], [t_0,dt2_zero,t_1]
                            elif coeffs_dt[2] < 0:
                                return [self.STATES['concave_decreasing'], self.STATES['convex_decreasing']], [t_0,dt2_zero,t_1]
                    elif coeffs_dt2[1] < 0:
                        if dt_inflection < 0:
                            return [self.STATES['convex_decreasing'], self.STATES['concave_decreasing']], [t_0,dt2_zero,t_1]
                        elif dt_inflection > 0:
                            return [self.STATES['convex_increasing'], self.STATES['concave_increasing']], [t_0,dt2_zero,t_1]
                        else:
                            # The first derivative is tangent to the x-axis at the inflection point
                            if coeffs_dt[2] > 0:
                                return [self.STATES['convex_increasing'], self.STATES['concave_increasing']], [t_0,dt2_zero,t_1]
                            elif coeffs_dt[2] < 0:
                                return [self.STATES['convex_decreasing'], self.STATES['concave_decreasing']], [t_0,dt2_zero,t_1]
                        
                elif len(zeros_in_interval) == 1:

                    if zeros_in_interval[0] < dt2_zero:

                        if coeffs_dt2[1] > 0:
                            return [self.STATES['concave_increasing'], self.STATES['concave_decreasing'], self.STATES['convex_decreasing']], [t_0,zeros_in_interval[0],dt2_zero,t_1]
                        elif coeffs_dt2[1] < 0:
                            return [self.STATES['convex_decreasing'], self.STATES['convex_increasing'], self.STATES['concave_increasing']], [t_0,zeros_in_interval[0],dt2_zero,t_1]

                    elif zeros_in_interval[0] > dt2_zero:

                        if coeffs_dt2[1] > 0:
                            return [self.STATES['concave_decreasing'], self.STATES['convex_decreasing'], self.STATES['convex_increasing']], [t_0,dt2_zero,zeros_in_interval[0],t_1]
                        elif coeffs_dt2[1] < 0:
                            return [self.STATES['convex_increasing'], self.STATES['concave_increasing'], self.STATES['concave_decreasing']], [t_0,dt2_zero,zeros_in_interval[0],t_1]

                    elif zeros_in_interval[0] == dt2_zero:
                            
                            if coeffs_dt2[1] > 0:
                                return [self.STATES['concave_increasing'], self.STATES['convex_increasing']], [t_0,zeros_in_interval[0],t_1]
                            elif coeffs_dt2[1] < 0:
                                return [self.STATES['convex_decreasing'], self.STATES['concave_decreasing']], [t_0,zeros_in_interval[0],t_1]
                
                elif len(zeros_in_interval) == 2:

                    if coeffs_dt2[1] > 0:
                        return [self.STATES['concave_increasing'], self.STATES['concave_decreasing'], self.STATES['convex_decreasing'], self.STATES['convex_increasing']], [t_0,zeros_in_interval[0],dt2_zero,zeros_in_interval[1],t_1]
                    elif coeffs_dt2[1] < 0:
                        return [self.STATES['convex_decreasing'], self.STATES['convex_increasing'], self.STATES['concave_increasing'], self.STATES['concave_decreasing']], [t_0,zeros_in_interval[0],dt2_zero,zeros_in_interval[1],t_1]