from digicomm import bitsToSymbols
import numpy as np

def test_bitsToSymbols():
    bits = np.array([0,0,0,1,1,0,1,1])
    assert (bitsToSymbols(bits,2) == np.array([0,0,0,1,1,0,1,1])).all()
    assert (bitsToSymbols(bits,4) == np.array([0,1,2,3])).all()
    assert (bitsToSymbols(bits,16) == np.array([1,11])).all()