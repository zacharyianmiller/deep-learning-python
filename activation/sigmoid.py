# sigmoid.py
import numpy as np

def calc(x):
    return 1 / (1 + np.exp(-x))

def calc_dydx(x):
    return calc(x) * (1 - calc(x))