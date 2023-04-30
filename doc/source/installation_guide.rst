Installation guide
------------------

1. Default method (install the most up-to-date changes)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    pip install plot-utils

2. Install a specific release
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    pip install git+https://github.com/jsh9/python-plot-utils@v0.6.13

Note
^^^^

If you run into the following issue on Mac OS X (or macOS) when importing ``plot_utils``:

.. code-block:: bash

    RuntimeError: Python is not installed as a framework.
    The Mac OS X backend will not be able to function correctly if Python
    is not installed as a framework.

Then please follow this solution to fix the issue: https://stackoverflow.com/a/21789908/8892243.
