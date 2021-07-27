# digicomm

Author: redd

Tools for digital communications simulations.

Install using `pip install digicomm`

## Organization

### `digicomm`
Most of the useful functions can be accessed using `import digicomm`. These include functions that can add noise for a given SNR, symbol decision functions, filters, etc.

### `digicomm.constellations`
`digicomm.constellations` contains common constellations used for linear modulations.

### `digicomm.LinearModulator`
`digicomm.LinearModulator` contains a class that can be used to simulate BER performance for various modulation schemes.

### `digicomm.Pll`
`digicomm.Pll` contains a PLL for timing and phase error tracking.
