import numpy as np
from scipy import linalg as la


class IncFDO():
    '''
    Incremental Version of Functional Directional Outlyingness.
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
    ----------
    Examples
    --------
    >>> import numpy as np
    >>> import skfda
    >>> import matplotlib.pyplot as plt
    >>> from inc_fdo import IncFDO

    >>> ### 0. generate synthetic data
    >>> random_state = np.random.RandomState(0)
    >>> n_samples = 200
    >>> fd = skfda.datasets.make_gaussian_process(
    ...     n_samples=n_samples,
    ...     n_features=100,
    ...     cov=skfda.misc.covariances.Exponential(),
    ...     mean=lambda t: 4 * t,
    ...     random_state=random_state)
    >>> magnitude_outlier = skfda.datasets.make_gaussian_process(
    ...     n_samples=1,
    ...     n_features=100,
    ...     cov=skfda.misc.covariances.Exponential(),
    ...     mean=lambda t: 4 * t + 20,
    ...     random_state=random_state)
    >>> shape_outlier_shift = skfda.datasets.make_gaussian_process(
    ...     n_samples=1,
    ...     n_features=100,
    ...     cov=skfda.misc.covariances.Exponential(),
    ...     mean=lambda t: 4 * t + 10 * (t > 0.4),
    ...     random_state=random_state)
    >>> shape_outlier_peak = skfda.datasets.make_gaussian_process(
    ...     n_samples=1,
    ...     n_features=100,
    ...     cov=skfda.misc.covariances.Exponential(),
    ...     mean=lambda t: 4 * t - 10 * ((0.25 < t) & (t < 0.3)),
    ...     random_state=random_state)
    >>> shape_outlier_sin = skfda.datasets.make_gaussian_process(
    ...     n_samples=1,
    ...     n_features=100,
    ...     cov=skfda.misc.covariances.Exponential(),
    ...     mean=lambda t: 4 * t + 2 * np.sin(18 * t),
    ...     random_state=random_state)
    >>> shape_outlier_slope = skfda.datasets.make_gaussian_process(
    ...     n_samples=1,
    ...     n_features=100,
    ...     cov=skfda.misc.covariances.Exponential(),
    ...     mean=lambda t: 10 * t,
    ...     random_state=random_state)
    >>> magnitude_shape_outlier = skfda.datasets.make_gaussian_process(
    ...     n_samples=1,
    ...     n_features=100,
    ...     cov=skfda.misc.covariances.Exponential(),
    ...     mean=lambda t: 4 * t + 2 * np.sin(18 * t) - 20,
    ...     random_state=random_state)
    >>> fd = fd.concatenate(magnitude_outlier, shape_outlier_shift, shape_outlier_peak,
    ...                     shape_outlier_sin, shape_outlier_slope,
    ...                     magnitude_shape_outlier)

    >>> ### 1. ordinary MS plot
    >>> X = fd.data_matrix
    >>> inc_fdo = IncFDO()
    >>> inc_fdo.initial_fit(X)

    >>> plt.figure()
    >>> plt.scatter(inc_fdo.MO[:, 0], inc_fdo.VO[:, 0])
    >>> plt.title('all time points')
    >>> plt.show()

    >>> ### 2. incremental MS plot
    >>> # make an initial result with 10 time points
    >>> X_10 = fd.data_matrix[:, 0:10, :]
    >>> inc_fdo.initial_fit(X_10)

    >>> plt.figure(figsize=(14, 6))
    >>> plt.subplot(2, 5, 1)
    >>> plt.scatter(inc_fdo.MO[:, 0], inc_fdo.VO[:, 0], s=10)
    >>> plt.title('first 10 points')

    >>> # incremental update
    >>> for i in range(10, 100):
    ...     x_new = fd.data_matrix[:, i, :]
    ...     inc_fdo.partial_fit(x_new)
    ...     if (i + 1) % 10 == 0:
    ...         plt.subplot(2, 5, (i + 1) // 10)
    ...         plt.scatter(inc_fdo.MO[:, 0], inc_fdo.VO[:, 0], s=10)
    ...         plt.title(f'first {i+1} points')

    >>> plt.show()
    '''
    def __init__(self, prg_obj = None):
        if prg_obj == None:
            self.t = 0
            self.MO = None
            self.VO = None
            self.FO = None
            self._l = None
        else:
            self.t = prg_obj.t
            self.MO = prg_obj.MO
            self.VO = prg_obj.VO
            self.FO = prg_obj.FO
            self._l = prg_obj._l
        

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
        self.t = X.shape[1]
        Z = np.median(X, axis=0)
        X_Z = X - Z
        
        mad = (np.median(np.abs(X_Z), axis=0) * 1.4826 ) + np.finfo(float).eps
        O = X_Z / mad

        self.MO = np.sum(O, axis=1) / self.t
        self.FO = np.sum(O * O, axis=1) / self.t
        self.VO = self.FO - self.MO * self.MO

        return self

    def partial_fit(self, x, type='add'):
        '''
        Partial fitting to update t, MO, FO, VO. Before using this method first
        time, applying initial_fit is required.
        Parameters
        ----------
        x: array-like, shape(n_dimensions, 1)
            A new time point of multidimensional function data that discretized
            with even intervals.
        Returns
        -------
        self
        '''
        z = np.median(x, axis=0)
        x_z = (x - z)[..., np.newaxis]
        # Median absolute deviation
        mad = np.median(np.abs(x_z), axis=0) * 1.4826

        t_new = self.t + 1

        A = (x_z / mad) [..., 0] 
        coeff1 = self.t

        self.MO = (self.MO * coeff1 + A) / t_new
        self.FO = (self.FO * coeff1 + A**2) / t_new
        self.VO = self.FO - self.MO * self.MO
        self.t = t_new

        return self

    def partial_delete(self, x):
        '''
        Partial fitting to downdate t, MO, FO, VO. Before using this method
        first time, applying initial_fit is required.
        Parameters
        ----------
        x: array-like, shape(n_dimensions, 1)
            A deleting time point of multidimensional function data that
            discretized with even intervals.
        Returns
        -------
        self
        '''
        z = np.median(x, axis=0)
        x_z = (x - z)[..., np.newaxis]
        # Median absolute deviation
        mad = np.median(np.abs(x_z), axis=0) * 1.4826
        t_new = self.t - 1

        A = (x_z / mad) [..., 0]
        coeff1 = self.t

        self.MO = (self.MO * coeff1 - A) / t_new
        self.FO = (self.FO * coeff2 - A**2) / t_new
        self.VO = self.FO - self.MO * self.MO
        self.t = t_new
        self._l = l_new

        return self
