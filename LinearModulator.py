import digicomm
import numpy as np

class LinearModulator:
    def __init__(self, ctype='qpsk', **kwargs):
        self.constellation = digicomm.getConstellation(type=ctype)
        self.M = len(self.constellation)
        
        if 'shape' in kwargs.keys():
            if kwargs['shape'] == 'srrc':
                self.alpha = kwargs['alpha']
                self.span = kwargs['span']
                self.N = kwargs['N']
                self.T = kwargs['T']
                self.p, self.t = digicomm.rcosdesign(self.alpha, self.span, self.N, Ts=self.T, shape='sqrt')
            else:
                raise Exception("Pulse shape not supported.")
        else:
            print("Using default pulse shape.")
            self.alpha = 0.5
            self.span = 6
            self.N = 10 # samples per symbol
            self.T = 1
            self.p, self.t = digicomm.rcosdesign(self.alpha, self.span, self.N, Ts=self.T, shape='sqrt')

    def simulateBER(self, SNR, L=10000):
        # constellation parameters
        self.n_bits = L * int(np.log2(self.M))

        # modulation
        bit_seq = digicomm.generateRandomBits(self.n_bits)
        sym_seq = digicomm.bitsToSymbols(bit_seq, self.M) # indexes of symbols to transmit
        z = digicomm.symbolsToIq(sym_seq, self.constellation)

        # generate signal
        tx = digicomm.upsample(z,self.p,self.N) # signal to transmit

        # channel model
        rx = digicomm.addNoise(tx,SNR=SNR,Eb=1/np.log2(self.M))
        # rx = digicomm.addNoise(tx,SNR=SNR,Eb=1)

        # demodulation
        mf = digicomm.matchedFilter(rx,self.p)
        sampinsts = np.arange(0,len(mf),self.N)
        mfo = mf[sampinsts]
        mfo = mfo[self.span:len(mfo)-self.span]
        decisions = digicomm.makeDecisions(mfo,self.constellation)

        # calculate BER
        bit_est = digicomm.symbolsToBits(decisions,self.M)
        ber = digicomm.calculateBer(bit_seq, bit_est)
        return ber

if __name__ == "__main__":
    
    # BER Plot Example
    import matplotlib.pyplot as plt
    import numpy as np

    snrs = np.arange(0,11)
    bers = np.zeros(snrs.shape)

    plt.figure()

    ctypes = ['qpsk','8psk','16psk']
    for c in ctypes:
        print(f"Calculating BER curve for {c}...")
        lm = LinearModulator(ctype=c, shape='srrc', alpha=0.5, span=6, N=10, T=1)
        for i,snr in enumerate(snrs):
            bers[i] = lm.simulateBER(snr, L=int(2e4))
        plt.semilogy(snrs,bers,'o--')

    print("Plotting BER curves...")
    plt.grid()
    plt.ylim([1e-5,1e0])
    plt.xlim([snrs[0],snrs[-1]])
    plt.ylabel("BER")
    plt.xlabel("SNR (Eb/N0)")
    plt.title("BER Curve Comparison")
    plt.legend(ctypes)
    plt.show()

    print("Done!")