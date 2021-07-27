# `tests\other`

These tests were used during development, but they aren't used by pytest. Maybe I'll get around to converting them some day.

## test_rcosdesign

I created pulse_normal and pulse_sqrt using the following MATLAB code:

```
alpha = 0.5;
span = 5;
Fs = 20;

h = rcosdesign(alpha, span, Fs, 'sqrt');
save('pulse_sqrt.mat', 'h');

h = rcosdesign(alpha, span, Fs, 'normal');
save('pulse_normal.mat', 'h');
```

``rcosdesign_test.py`` calculates the inner product of the MATLAB generated impulse response and the Python generated impulse response. The results should be very close to zero. If not, the Python version isn't producing the same values at MATLAB.