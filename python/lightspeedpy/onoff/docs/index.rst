Onoff plugin
===============

The `onoff` plugin is designed for pulsar analysis. It assigns phases to each frame, makes a stacked image for phases in the "on" range, and subtracts the image for phases in the "off" range. The on and off ranges are provided as the :code:`--range` command line argument as two comma-separated values: :code:`--range ON_RANGE, OFF_RANGE`. Note that a .par ephemeris file is necessary, which is used to assign phases using the `PINT` software.

The format of a single :code:`RANGE` should be formatted as follows. It should be a list :code:`RANGE1|RANGE2|...` where the range triggers on an event whose phase falls inside any of the :code:`RANGE`s. Each :code:`RANGE` element is formatted as :code:`START:STOP`. If START > STOP, the phase window is assumed to wrap around through 1. So e.g. :code:`0.2:0.3|0.9:0.1` triggers on every event whose phase $\phi$ satisfies $0.2 < \phi < 0.3$ or $0.9 < \phi$ or $\phi < 0.1$.

API documentation
^^^^^^^^^^^^^^^^^

.. automodule:: lightspeedpy.onoff
    :members:
    :undoc-members:
    :show-inheritance: