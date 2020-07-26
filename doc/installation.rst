Installation
============

Installing **QSynthesis** is rather straightforward, you just need to:

1. Install Triton on your system (cf. `Documentation <https://triton.quarkslab.com/documentation/doxygen/index.html#install_sec>`_)
2. Use the provided `setup.py` to install QSynthesis
3. (Optional) Install Qtrace-DB to enable synthesizing from execution trace
4. (Optional) copy the `ida_plugin/qsynthesis_plugin.py` in the IDA Pro plugin folder
5. (Optional) install additionnal dependencies to generate tables or to serve tables as server


Installing QSynthesis
---------------------

After installation of Triton, just run the following command:

.. code-block:: bash

    $ git clone gitlab@gitlab.qb:synthesis/qsynthesis.git
    $ cd qsynthesis
    $ pip3 install .

You're ready to go !


Installing Qtrace-DB
--------------------

QSynthesis takes a Triton AST as input. Thus its origin does not significantly matter.
However it provides some utilities to extract expressions coming from Qtrace-DB traces.

The whole installation of Qtrace-DB is documented here: `Qtrace-DB Documentation <https://qtrace.doc.qb/qtrace-db/>`_


Installing the IDA Plugin
-------------------------

The IDA plugin is made to work equally whether Qtrace-IDA is installed or not.

If you want to run it through Qtrace-IDA follow installation at: https://gitlab.qb/qtrace/qtrace-ida

To install it just copy the file `ida_plugin/qsynthesis_plugin.py` in your IDA Pro plugin directory.

.. note:: The plugin is solely working on IDA with Python3


Installing optional dependencies
--------------------------------

**Tables**: Table generation require two additional packages:

* `dragonffi <https://github.com/aguinet/dragonffi>`_ for JITing the evaluation of expressions
* `sympy <https://www.sympy.org/en/index.html>`_ for linearization of expressions

They can be installed with:

.. code-block:: bash

    $ pip3 install .[generator]

**Table server as REST**: Qsynthesis enables serving lookup table as a REST API. It relies
on `FastAPI <https://fastapi.tiangolo.com/>`_  for that. It can also be installed with:

.. code-block:: bash

    $ pip3 install .[server]