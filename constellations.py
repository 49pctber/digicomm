#
# Constellations. These are not necessarily unit energy.
#
# Author: redd
#

import numpy as np


r_g = 2.75 # radius gamma used for 16-APSK


constellations = {
    'bpsk' : np.array([-1, 1], 'complex128'),
    'bpam' : np.array([-1, 1], 'complex128'),
    '4pam' : np.array([-3, -1, 3, 1], 'complex128'),
    '8pam' : np.array([-7, -5, -1, -3, 7, 5, 1, 3], 'complex128'),
    'yqam' : np.array([0, np.exp(1j*np.pi/6), np.exp(1j*np.pi*3/2), np.exp(1j*5*np.pi/6)]),
    '8qam' : np.array([-3+1j,-3-1j,-1+1j,-1-1j,3+1j,3-1j,1+1j,1-1j]),
    'qpsk' : np.array([1+1j,1-1j,-1+1j,-1-1j]),
    '8psk' : np.array([
            np.exp(1j*0*np.pi/4),
            np.exp(1j*1*np.pi/4),
            np.exp(1j*3*np.pi/4),
            np.exp(1j*2*np.pi/4),
            np.exp(1j*6*np.pi/4),
            np.exp(1j*7*np.pi/4),
            np.exp(1j*5*np.pi/4),
            np.exp(1j*4*np.pi/4)
        ]),
    '16qam' : np.array([-3+3j,-3+1j,-3-3j,-3-1j,-1+3j,-1+1j,-1-3j,-1-1j,3+3j,3+1j,3-3j,3-1j,1+3j,1+1j,1-3j,1-1j]),
    '16psk' : np.array([
            np.exp(1j*0*np.pi/8),
            np.exp(1j*1*np.pi/8),
            np.exp(1j*3*np.pi/8),
            np.exp(1j*2*np.pi/8),
            np.exp(1j*6*np.pi/8),
            np.exp(1j*7*np.pi/8),
            np.exp(1j*15*np.pi/8),
            np.exp(1j*14*np.pi/8),
            np.exp(1j*10*np.pi/8),
            np.exp(1j*11*np.pi/8),
            np.exp(1j*9*np.pi/8),
            np.exp(1j*8*np.pi/8),
            np.exp(1j*12*np.pi/8),
            np.exp(1j*13*np.pi/8),
            np.exp(1j*5*np.pi/8),
            np.exp(1j*4*np.pi/8)
        ]),
    '16apsk' : np.array([
        r_g*np.exp(1j*3*np.pi/12),
        r_g*np.exp(1j*-3*np.pi/12),
        r_g*np.exp(1j*9*np.pi/12),
        r_g*np.exp(1j*-9*np.pi/12),
        r_g*np.exp(1j*1*np.pi/12),
        r_g*np.exp(1j*-1*np.pi/12),
        r_g*np.exp(1j*11*np.pi/12),
        r_g*np.exp(1j*-11*np.pi/12),
        r_g*np.exp(1j*5*np.pi/12),
        r_g*np.exp(1j*-5*np.pi/12),
        r_g*np.exp(1j*7*np.pi/12),
        r_g*np.exp(1j*-7*np.pi/12),
        np.exp(1j*3*np.pi/12),
        np.exp(1j*-3*np.pi/12),
        np.exp(1j*9*np.pi/12),
        np.exp(1j*-9*np.pi/12),
    ]),
} # not necessarily unit energy!