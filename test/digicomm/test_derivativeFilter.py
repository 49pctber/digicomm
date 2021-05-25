from digicomm import *
import numpy as np
import matplotlib.pyplot as plt

# x,T = np.linspace(-2,2,num=101,retstep=True)
# y = np.exp(x) + 0.001 * np.random.normal(size=x.shape)
x,T = np.linspace(-np.pi,np.pi,num=101,retstep=True)
y = np.sin(x) + 0.001 * np.random.normal(size=x.shape)

N = 4*10-1
yd1 = derivativeFilter(y,N=N,Tsamp=T)
yd2 = derivativeFilter2(y,N=N,Tsamp=T)

plt.figure()
plt.subplot(2,1,1)
plt.plot(x,y)
plt.plot(x,yd1)
plt.grid()
plt.subplot(2,1,2)
plt.plot(x,y)
plt.plot(x,yd2)
plt.grid()
plt.show()