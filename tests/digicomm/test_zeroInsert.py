from digicomm import zeroInsert
import numpy as np

def test_zeroInsert():
    x = np.array([1,2,3,4],dtype='complex')
    assert (zeroInsert(x,2) == np.array([1,0,2,0,3,0,4],dtype='complex')).all()
    assert (zeroInsert(x,3) == np.array([1,0,0,2,0,0,3,0,0,4],dtype='complex')).all()
    assert (zeroInsert(x,4) == np.array([1,0,0,0,2,0,0,0,3,0,0,0,4],dtype='complex')).all()