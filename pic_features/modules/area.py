import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew


class Area:
    image = np.array([])
    h_ch = np.array([])
    s_ch = np.array([])
    v_ch = np.array([])
    r_ch = np.array([])
    g_ch = np.array([])
    b_ch = np.array([])
    mean = 0
    disp = 0
    kurtosis = 0
    assym = 0

    def __init__(self):
        pass

    def set_param(self, param, val):
        pass
        


