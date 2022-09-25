Installation
============

Installing **QSynthesis** is rather straightforward, you just need to:

* Use the provided `setup.py` to install QSynthesis
* (Optional) copy the `ida_plugin/qsynthesis_plugin.py` in the IDA Pro plugin folder
* (Optional) install additionnal dependencies to generate tables or to serve tables as server

Thus it can be done with:

.. code-block:: bash

    $ git clone https://github.com/quarkslab/qsynthesis.git
    $ cd qsynthesis
    $ pip3 install .

You're ready to go !

Optional dependencies
^^^^^^^^^^^^^^^^^^^^^

Not to install too many third-party libraries the default installation
only install the strict minimum. Various features requires installing
additional packages.

* **Table generation**: requires `dragonffi <https://github.com/aguinet/dragonffi>`_ and `sympy <https://www.sympy.org/en/index.html>`_ ``pip install .[generator]``
* **Reassemby**: requires `arybo <https://github.com/quarkslab/arybo>`_ and `llvmlite <https://github.com/numba/llvmlite>`_ ``pip install .[reassembly]``
* **Server**: requires `FastAPI <https://fastapi.tiangolo.com>`_ ``pip install .[server]``

You can install all dependencies all at once with ``pip install .[all]``.

Installing the IDA Plugin
-------------------------

Then to install the plugin just copy the file `ida_plugin/qsynthesis_plugin.py` in your IDA Pro plugin directory.

.. warning:: The plugin is solely working on IDA with Python3, and latest versions of IDA.
