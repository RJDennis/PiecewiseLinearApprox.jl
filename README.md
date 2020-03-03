# PiecewiseLinearApprox

Installation
------------

To install this package simply type in the REPL:

```
using Pkg
Pkg.add("PiecewiseLinearApprox")
```

Use
---

The package performs piecewise linear interpolation over an arbitrary number of dimensions.  There are only two functions that need mentioning.  The first function computes a set of nodes spaced uniformly over a specified domain for a variable.

```
nodes = piecewise_linear_nodes(n,domain)
```

where `n` is an integer representing the desired number of nodes and `domain` is a 1D array containing the upper and lower values for the domain.  If no domain is specified, then it defaults to [1.0,-1.0].

To evaluate the piecewise linear approximation at an arbitrary point in the domain we use the command

```
y_hat = piecewise_linear_evaluate(y,nodes,point)
```

where `y` is a multidimensional array, `nodes` is a tuple of 1D-arrays or an array of 1D-arrays, and `point` is a 1D array.

If `point` resides outside the domain in any dimension, then linear extrapolation in that dimension is performed.
