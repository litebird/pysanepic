# API reference

## High-level interface

```{eval-rst}
.. autofunction:: pysanepic.make_maps

.. autoclass:: pysanepic.DetectorData
   :no-members:

.. autoclass:: pysanepic.Pointing
   :members: pixels_and_psi

.. autoclass:: pysanepic.GLSParameters
   :no-members:

.. autoclass:: pysanepic.MapResult
   :no-members:
```

## Low-level interface

```{eval-rst}
.. autoclass:: pysanepic.GLS
   :members: solve, apply_A, rhs, to_healpix

.. autofunction:: pysanepic.one_over_f_weights

.. autofunction:: pysanepic.two_angle

.. autofunction:: pysanepic.resolve_nthreads
```
