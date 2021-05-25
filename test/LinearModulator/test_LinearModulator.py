from LinearModulator import *
import matplotlib.pyplot as plt
import numpy as np

# Simulate BER
# lm = LinearModulator(ctype='16apsk')
# print(lm.simulateBER(0))

# Produce Signal
lm = LinearModulator(ctype='qpsk')
bits = np.array([
    0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,
    1,0,1,1,0,1,0,1,0,1,1,1,1,0,0,1,0,1,0,0,1,1,0,1,
    1,0,1,0,1,0,1,1,0,1,1,0,0,1,1,1,0,1,0,1,0,0,1,1,
],dtype='int')
rx = lm.modulate(bits, SNR=3, phase=np.pi/2)
bit_hats = lm.demodulate(rx, phase=-np.pi/2)

t = np.arange(0,len(rx)) * lm.Tsamp

be = np.count_nonzero(bit_hats - bits)
ber = be / len(bits)
print(f"There were {be} bit errors.")
print(f"That is a BER of {ber}.")

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(np.real(rx),np.imag(rx),'k',linewidth=.1)
plt.grid()
plt.axis('equal')
plt.title("Phase Trajectory")
plt.subplot(2,2,2)
plt.plot(t,np.real(rx),'k',linewidth=.6)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.title("Real Part of Received Signal")
plt.subplot(2,2,4)
plt.plot(t,np.imag(rx),'k',linewidth=.6)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.title("Imaginary Part of Received Signal")
plt.show()
