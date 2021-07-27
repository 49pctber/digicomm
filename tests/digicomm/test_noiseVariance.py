from digicomm import noiseVariance

def test_calculateBer0():
    assert noiseVariance(0,1) == 1
    assert noiseVariance(10,1) == 0.1
    assert noiseVariance(-10,1) == 10

def test_calculateBer1():
    assert noiseVariance(0,2) == 2
    assert noiseVariance(10,.5) == 0.05
    assert noiseVariance(-10,10) == 100