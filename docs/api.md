# API reference

## High-level interface

```{eval-rst}
.. autofunction:: pysanepic.make_maps

.. autoclass:: pysanepic.DetectorData
   :no-members:

.. autoclass:: pysanepic.Pointing
   :members: pixels_and_psi

.. autoclass:: pysanepic.OneOverFNoise
   :members: weights

.. autoclass:: pysanepic.TabulatedNoise
   :members: weights

.. autoclass:: pysanepic.DetectorGroup
   :no-members:

.. autoclass:: pysanepic.CommonMode
   :no-members:

.. autofunction:: pysanepic.estimate_noise

.. autoclass:: pysanepic.GLSParameters
   :no-members:

.. autoclass:: pysanepic.CalibrationParameters
   :no-members:

.. autoclass:: pysanepic.Template
   :no-members:

.. autofunction:: pysanepic.pol_angle_template

.. autofunction:: pysanepic.fwhm_template

.. autofunction:: pysanepic.beam_templates

.. autofunction:: pysanepic.beam_tods

.. autofunction:: pysanepic.beam_parameters

.. autofunction:: pysanepic.blm_templates

.. autofunction:: pysanepic.blm_tods

.. autoclass:: pysanepic.MapResult
   :no-members:
```

## Low-level interface

```{eval-rst}
.. autoclass:: pysanepic.GLS
   :members: solve, shared_covariance, apply_A, rhs, set_templates, binned, full, to_healpix

.. autofunction:: pysanepic.one_over_f_weights
.. autofunction:: pysanepic.psd_weights
.. autofunction:: pysanepic.fast_fft_length

.. autofunction:: pysanepic.two_angle

.. autofunction:: pysanepic.resolve_nthreads
```
