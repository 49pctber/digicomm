from digicomm import wrap
import numpy as np
import pytest

def test_wrap():
    assert wrap(0.5,1) == 0.5
    assert wrap(-0.5,1) == -0.5
    assert wrap(0,0.1) == 0
    assert wrap(2,1) == 0
    assert wrap(2,3) == 2
    assert wrap(4,3) == -2
    assert wrap(3,2) == -1
    assert wrap(4.1,2) == pytest.approx(.1)
    assert wrap(5,2) == 1

def test_wrap_equal():
    assert wrap(2,2) == -2 # notice this weird case.

def test_wrap_numpy():
    assert wrap(np.array([3,4.1,5]),2) == pytest.approx(np.array([-1,.1,1]))
    assert wrap(np.array([3,4.1,5]),4) == pytest.approx(np.array([3,-3.9,-3]))
    assert wrap(np.array([2,-3,15]),2) == pytest.approx(np.array([-2,1,-1]))
