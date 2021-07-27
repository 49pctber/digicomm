from digicomm import valleyFill
import numpy as np

def test_valleyFill():
    assert (valleyFill(np.array([1,2,3,4])) == np.array([4,4,4,4])).all()
    assert (valleyFill(np.array([1,5,3,4])) == np.array([5,5,4,4])).all()
    assert (valleyFill(np.array([10,2,3,4])) == np.array([10,4,4,4])).all()
    assert (valleyFill(np.array([4,3,2,1])) == np.array([4,3,2,1])).all()
    assert (valleyFill(np.array([4.53,1.23,9.86,5.96])) == np.array([9.86,9.86,9.86,5.96])).all()

    assert (valleyFill(np.array([1,2,3,4]), flip=True) == np.array([1,2,3,4])).all()
    assert (valleyFill(np.array([1,5,3,4]), flip=True) == np.array([1,5,5,5])).all()
    assert (valleyFill(np.array([10,2,3,4]), flip=True) == np.array([10,10,10,10])).all()
    assert (valleyFill(np.array([4,3,2,1]), flip=True) == np.array([4,4,4,4])).all()
    assert (valleyFill(np.array([4.53,1.23,9.86,5.96]), flip=True) == np.array([4.53,4.53,9.86,9.86])).all()