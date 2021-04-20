#
# An example of BER simulations for various constellations.
#
# Author: redd
#

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from digicomm import * # import my helper functions

# Parameters
c_types = constellations.keys()
ber_data = {}
nsyms = 2**10
be_threshold = 10
minbits = nsyms * 16 # simulate at least this many symbols per SNR (helps at low SNRs)

for c_type in c_types: # for each constellation
    c = getConstellation(type=c_type) # get constellation to use as LUT
    M = len(c) # number of constellation points
    bpsym = int(np.log2(M)) # bits per symbol

    SNR = np.linspace(0,10,11) # SNRs to simulate
    BER = np.zeros(len(SNR))

    for i, snr in enumerate(SNR): # for each SNR
        be = 0
        nbits = 0

        while be < be_threshold or nbits < minbits: # simulate until you find at least a certain number of errors
            # produce random bits
            nbits += bpsym * nsyms
            bits = np.random.randint(0,2,size=(bpsym * nsyms,)) # choose random bits
            syms = bitsToSymbols(bits,M) # map bits to symbols

            # transmission
            tx = c[syms]
            rx = addNoise(tx, SNR=snr, Eb=1/np.log2(len(c)))

            # make decisions
            rx_sym = makeDecisions(rx, c) # make decisions on noisy received symbols
            rx_bits = symbolsToBits(rx_sym, M)
            be += np.count_nonzero(rx_bits - bits) # count bit errors

        BER[i] = be / nbits # calculate BER

    ber_data[c_type] = BER

# Plots
for c in ber_data.keys():
    plt.semilogy(SNR, ber_data[c], '*--')
plt.legend(ber_data.keys())
plt.grid(True)
plt.xlim([0,10])
plt.xlabel("SNR (Eb/N0)")
plt.ylabel("BER")
plt.title("Constellation BER in AWGN Channel")
plt.show()