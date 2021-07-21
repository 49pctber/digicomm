#
# This needs a bit more work.
#
# Check out dev_pll.ipynb
#
# Author: redd
#

import scipy.signal as signal
import numpy as np

class LoopFilter:

    def __init__(self, wn, K, zeta=1/np.sqrt(2), Ts=1, xi=0):
        self.wn = wn # corner frequency [rad]
        self.K = K # gain [rad/V]
        self.zeta = zeta # damping factor

        self.tau1 = K / wn**2
        self.tau2 = 2*zeta/wn - 1/K

        self.b,self.a = signal.bilinear(
            np.array([self.tau2,1]),
            np.array([self.tau1,1]),
            fs=1/Ts
        )

        self.zi = xi * signal.lfilter_zi(self.b,self.a)
    
    def update(self, xn):

        if type(xn) != np.ndarray:
            xn = np.array([xn])

        z, self.zi = signal.lfilter(self.b,self.a,xn,zi=self.zi)
        return z


class Integrator:
    """
    Models an integrator with
    gain K,
    sample time Ts,
    initial value phi,
    an optional limit that contrains the outputs to [-limit, limit]

    integrate using integrate(). Input can be a scalar or numpy array.
    """

    def __init__(self, K, Ts=1, phi=0, limit=None):
        self.phi = 0 # most recent integrator value
        self.Ts = Ts # sample time [s]
        self.K = K # integrator gain
        self.limit = limit # maximum absolute value for integrator, use None if n/a

    def _checkLimits(self, phi):
        """
        For checking limits on a numpy array
        """
        phi[phi > self.limit] = self.limit
        phi[phi < -self.limit] = -self.limit
        return phi

    def _checkLimit(self, phi):
        """
        For checking limits on a scalar
        """
        phi = self.limit if phi > self.limit else phi
        phi = -self.limit if phi < -self.limit else phi
        return phi

    def integrate(self, v):
        if type(v) == np.ndarray:
            phi = self.phi + self.K * self.Ts * np.cumsum(v)
            phi = phi if self.limit == None else self._checkLimits(phi)
            self.phi = phi[-1]
        else:
            phi = self.phi + self.K * self.Ts * v
            phi = phi if self.limit == None else self._checkLimit(phi)
            self.phi = phi
        return phi


class Dds:

    def __init__(self, dds_freq, K0, Ts=1):
        self.K0 = K0
        self.i = Integrator(1,Ts=Ts)
        self.C = 2*np.pi*dds_freq # [rad/samp]

    def update(self, v):
        z = self.i.integrate(self.K0 * v + self.C)
        return np.exp(1j*z)


class Pll:

    def __init__(self, dds_freq, wn, K, K0, zeta=1/np.sqrt(2), Ts=1):
        self.lf = LoopFilter(wn, K, zeta=zeta, Ts=Ts) # TODO this works, but it's not great
        self.dds = Dds(dds_freq, K0, Ts=Ts)
        self.p = 0j

    def update(self, x, reterror=False):
        e = np.zeros(x.shape)
        y = np.zeros(x.shape, dtype='complex')
        
        for i in range(0,len(y)):
            e[i] = np.angle(x[i] * np.conj(self.p))
            v = self.lf.update(e[i])
            y[i] = complex(self.dds.update(v))
            self.p = y[i]
        
        if reterror:
            return y, e
        else:
            return y