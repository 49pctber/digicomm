from digicomm import calculateBer
import numpy as np

def test_calculateBer():
    b0 = np.array([0,0,0,1,1,0,1,1])
    b1 = np.array([0,0,0,1,0,1,0,0])
    b2 = np.array([0,1,0,1,0,1,0,1])
    b3 = np.array([0,1,0,1,0,1,0,0])

    assert calculateBer(b0,b0) == 0/8
    assert calculateBer(b0,b1) == 4/8
    assert calculateBer(b0,b2) == 4/8
    assert calculateBer(b0,b3) == 5/8

    assert calculateBer(b1,b1) == 0/8
    assert calculateBer(b1,b2) == 2/8
    assert calculateBer(b1,b3) == 1/8

    assert calculateBer(b2,b2) == 0/8  
    assert calculateBer(b2,b3) == 1/8

    assert calculateBer(b3,b3) == 0/8