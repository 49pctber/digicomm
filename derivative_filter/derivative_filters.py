#
# Derivative filter demo
#
# Author: redd
#

import numpy as np
from matplotlib import pyplot as plt
from digicomm import derivativeFilter

Tsamp = .01
Tstart = -10
Tstop = 10
t = np.arange(Tstart,Tstop,step=Tsamp)
N = 4*12-1
x = np.sinc(t) # np.log(t) # np.abs(t) # np.exp(t) # np.cos(t)
y = derivativeFilter(x,N=N,Tsamp=Tsamp,zero_edge=True)

plt.plot(t,x)
plt.plot(t,y)
plt.grid(True)
plt.legend([
    "Original Signal",
    "Derivative"
])
plt.show()

