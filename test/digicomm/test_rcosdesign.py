import numpy as np
import matplotlib.pyplot as plt
import digicomm
from scipy.io import loadmat

span = 5
alpha = 0.5
Ts = 1
Fs = 20
h,t=digicomm.rcosdesign(alpha, span, Fs, shape='sqrt')

data = loadmat('test/pulse_sqrt.mat')
h_matlab = data['h'][0]
e = h_matlab - h
print("Inner product (should be small):", e @ e)

span = 5
alpha = 0.5
Ts = 1
Fs = 20
h,t=digicomm.rcosdesign(alpha, span, Fs, shape='normal')

data = loadmat('test/pulse_normal.mat')
h_matlab = data['h'][0]
e = h_matlab - h
print("Inner product (should be small):", e @ e)