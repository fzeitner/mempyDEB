# mempyDEB

mempyDEB is a Python package that provides the basics for doing DEB-TKTD modelling in Python. 
It mostly defines a baseline model, default parameters and functions to run the model.

## Installation

To install `mempyDEB`, use the command

`pip install git+https://github.com/simonhansul/mempyDEB.git`

(e.g. in Anaconda prompt with the desired environment activated).

## Getting started

The examples directory contains a notebook which demonstrates the basic functionality of this package. <br>
In short, you can run a default simulation using

```Python
from mempyDEB.DEBODE.simulators import * # imports functions to run models
from mempyDEB.DEBODE.defaultparams import * # imports default parameters
sim = simulate_DEBBase(defaultparams_DEBBase) # runs the DEBBase model (a variant of DEBkiss) with default parameters
```

Generally, `mempyDEB` is a fairly slim package, designed as a low-barrier entry point to DEB-TKTD modelling. <br>
There are not many functions, but the code can be adapted to be used for more extensive (research) tasks.

## Info & Acknowledgements

This pacakge was developed for the course "Mechanistic Effect Modelling" at Osnabrück University, as well as the postgraduate course "Dynamic Modelling of Ecotoxicological Effects" organized at University of Copenhagen. 
