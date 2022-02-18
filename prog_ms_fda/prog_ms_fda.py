import numpy as np
from scipy.stats import gaussian_kde
from scipy import linalg as la


class ProgressiveFDA():

    '''
    Progressive Version of Functional Directional Outlyingness.
    Non-incremental version is introduced in "Multivariate Functional Data
    Visualization and Outlier Detection" [Dai and Genton, 2018].
    Parameters
    ----------
    None
    Attributes
    ----------
    t: integer
        number of time points
    MO: ndarray, shape(n_dimensions, 1)
        mean directional outlyingness. n_dimensions is p in Dai and Genton, 2018].
    VO: ndarray, shape(n_dimensions, 1)
        variation ofdirectional outlyingness
    FO: ndarray, shape(n_dimensions, 1)
        functional directional outlyingness

       ''' 
   
    def __init__(self, inc_fdo = None, X = None, threshold = 10):

        if inc_fdo == None:
            self.X = None
            self.t = 0
            self.MO = None
            self.VO = None
            self.FO = None
            self._l = None
            self.threshold = threshold
            self.Z = None
            self.mad = None

        else:

            self.X = X
            self.t = X.shape[1]
            self.MO = inc_fdo.MO
            self.VO = inc_fdo.VO
            self.FO = inc_fdo.FO
            self._l = inc_fdo._l
            self.threshold = 10
            self.Z = np.median(X, axis=0)
            self.mad = np.median(np.abs(X - self.Z), axis=0) * 1.4826
        
        
    def initial_fit(self, X):

        '''
        Initial fitting to generate initial t, MO, FO, VO.
        Parameters
        ----------
        X: array-like, shape(n_dimensions, n_time_points, 1)
            Multidimensional function data discretized with even intervals.
        Returns
        -------
        self
        '''
       
        self.X = X
        self.t = X.shape[1]
        Z = np.median(X, axis=0)
        self.Z = Z
        X_Z = X - Z
        # Median absolute deviation
        mad = np.median(np.abs(X_Z), axis=0) * 1.4826
        self.mad = mad
        
        # stagel_donoho_outlyingness
        E = np.abs(X_Z) / mad

        self._l = (la.norm(X_Z, axis=1) + np.finfo(float).eps)[..., np.newaxis]
        V = X_Z / self._l

        O = E * V

        self.MO = np.sum(O, axis=1) / self.t
        self.FO = np.sum(O * O, axis=1) / self.t
        self.VO = self.FO - self.MO * self.MO

        return self
    
    def check_fit(self, X_new):
        
        Z_new = np.median(X_new, axis=0)
        X_Z_new = X_new - self.Z
        
        mad_new = np.median(np.abs(X_Z_new), axis=0) * 1.4826
        
        #try:
            
        p = gaussian_kde(mad_new)(mad_new)
        q = gaussian_kde(self.mad)(self.mad) + np.finfo(float).eps


        if self.kulbackLeibler_div_symmetric(p, q) > self.threshold:

            self.initial_fit(X_new)
            return True

        else: return False
        
            

    def partial_fit(self, X1, approximate=True):

        '''
        Partial fitting to update t, MO, FO, VO. Before using this method first
        time, applying initial_fit is required.
        Parameters
        ----------
        x: array-like, shape(n_dimensions, 1)
            A new time series data that discretized
            with even intervals.
        approximate: perform approximation check or not. If approximation check is true then
        the results are approximated as long as they lie within the error 'threshold' specified during
        initialization.
        Returns
        -------
        self
        '''
                
        X_new = np.concatenate((self.X, X1[np.newaxis,...]), axis=0)

        if approximate:
            if self.check_fit(X_new):
                return self
        
        x_Z = X1 - self.Z
        _l_p1 = (la.norm(x_Z) + np.finfo(float).eps)[..., np.newaxis]
        
        self._l = np.concatenate((self._l, _l_p1[np.newaxis,...]), axis=0)
        
        O_p1 = (np.abs(x_Z)/self.mad) * (x_Z/_l_p1)
        M_p1 = np.sum(O_p1)/self.t
        
        self.MO = np.concatenate((self.MO, M_p1[np.newaxis,...]))
        self.FO = np.concatenate((self.FO, (M_p1 * M_p1)[np.newaxis,...]))
        self.VO = self.FO - self.MO * self.MO
        
        # update X
        self.X = X_new
        return self
    
    def kulbackLeibler_div_symmetric(self, p,q):
        return np.sum(np.where(p !=0, 0.5 * (p-q) * np.log(p/q), 0))
        