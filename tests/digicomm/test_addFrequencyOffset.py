from digicomm import addFrequencyOffset
import numpy as np

threshold = 1e-15 # precision for floating point errors

def test_addFrequencyOffset():
    iqs = np.ones((5,), dtype='complex')
    z = threshold * np.ones((5,))

    assert (np.abs(addFrequencyOffset(iqs,nuT=0.0) - iqs) < z).all()
    assert (np.abs(addFrequencyOffset(iqs,nuT=1.0) - iqs) < z).all()
    assert (np.abs(addFrequencyOffset(iqs,nuT=-1.0) - iqs) < z).all()

    assert (np.abs(addFrequencyOffset(iqs,nuT=0.5) - np.array([1.0,-1.0,1.0,-1.0,1.0])) < z).all()
    assert (np.abs(addFrequencyOffset(iqs,nuT=-0.5) - np.array([1.0,-1.0,1.0,-1.0,1.0])) < z).all()

    assert (np.abs(addFrequencyOffset(iqs,nuT=0.25) - np.array([1.0,1j,-1.0,-1j,1.0])) < z).all()
    assert (np.abs(addFrequencyOffset(iqs,nuT=-0.25) - np.array([1.0,-1j,-1.0,1j,1.0])) < z).all()