Installation
============

Installing **QSynthesis** is rather straightforward, you just need to:

* Install Triton on your system (cf. `Documentation <https://triton.quarkslab.com/documentation/doxygen/index.html#install_sec>`_)
* Use the provided `setup.py` to install QSynthesis
* (Optional) Install Qtrace-DB to enable synthesizing from execution trace
* (Optional) copy the `ida_plugin/qsynthesis_plugin.py` in the IDA Pro plugin folder
* (Optional) install additionnal dependencies to generate tables or to serve tables as server


Installing QSynthesis
---------------------

After installation of Triton, just run the following command:

.. code-block:: bash

    $ git clone gitlab@gitlab.qb:synthesis/qsynthesis.git
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

Installing Qtrace-DB
--------------------

QSynthesis takes a Triton AST as input. Thus its origin does not significantly matter.
However it provides some utilities to extract expressions coming from Qtrace-DB traces.

The whole installation of Qtrace-DB is documented here: `Qtrace-DB Documentation <https://qtrace.doc.qb/qtrace-db/>`_


Installing the IDA Plugin
-------------------------

The IDA plugin is made to work equally whether Qtrace-IDA is installed or not.
If you want to run it through Qtrace-IDA follow installation at: https://gitlab.qb/qtrace/qtrace-ida

Then to install the plugin just copy the file `ida_plugin/qsynthesis_plugin.py` in your IDA Pro plugin directory.

.. warning:: The plugin is solely working on IDA with Python3, and latest versions of IDA.
