Bias plugin
===============

This plugin stacks bias frames into a single image which can be used to correct Lightspeed data. The result is saved as a fits file which you can use for the :code:`--bias` flag of the other plugins. It is never necessary to run the bias plugin because you can always provide the raw data to the other plugins' :code:`--bias`` flags. They will do the stacking automatically. However, using the bias plugin can save time, because it means that the stacking is run only once.

Noise modeling
^^^^^^^^^^^^^^

If you will later use the weight method for your analysis, you will need to fit models to the noise distribution of each pixel, using data from the stack of bias frames. The `bias` plugin will do this if you provide the :code:`--map-noise` command line flag. If you'd also like to visualize the noise distribution, you can provide the :code:`--dbg-noise` argument too.

API documentation
^^^^^^^^^^^^^^^^^

.. automodule:: lightspeedpy.bias
    :members:
    :undoc-members:
    :show-inheritance: