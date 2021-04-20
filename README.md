# digicomm

Tools for digital communications simulations.

## Organization

``digicomm.py`` is the main workhorse that contains the relevant functions. This script can be imported to have access to
- functions that can add noise, phase offsets, frequency offsets to complex values representing symbols
- symbol decision functions
- perform frequency offset estimation
- derivative filters
- a MATLAB-style rcosdesign
- ...and more to come.

Folders contain example applications of the ``digicomm.py`` script, like bit error rate simulations, derivative filters, symbol synchronization, etc. The ``test\`` contains code that I used to develop functionality for a given function.