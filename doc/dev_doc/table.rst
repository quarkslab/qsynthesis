Oracle Management
=================

qsynthesis-table-manager utility
--------------------------------

The tedious task about using QSynthesis is having good tables (aka oracles)
for efficient synthesis. QSynthesis provides `qsynthesis-table-manager`
which provide high-level function to manage tables. Its help file is:

.. highlight:: none

::

    Usage: qsynthesis-table-manager [OPTIONS] COMMAND [ARGS]...

    Options:
      --version   Show the version and exit.
      -h, --help  Show this message and exit.

    Commands:
      check     Checking the equivalence of hashes against evaluation of...
      compare   Compare two tables
      dump      Dump the content of the table on stdout
      generate  Table generation utility
      info      Getting information of a given database
      merge     Merge entries of the first database in the second



The first interesting command is ``info`` that provides information about the
database, like grammar used, inputs, used, number of entries etc.

The ``check`` command re-evaluate all expressions against the inputs of table
to ensure the hash is valid. **This is a very slow process** and should only
be used for debugging purposes.

The ``compare`` command takes two tables and compare them to know how many entries
they have in common thus not in common.

The ``merge`` command merge two tables as long as they are using the same set of inputs.


Table Generation
----------------

The ``generate`` command of the utility, enables generating new tables. It provides various parameters.
The help message is the following:

::

    Usage: qsynthesis-table-manager generate [OPTIONS] OUTPUT_FILE

      Table generation utility

    Options:
      -bs, --bitsize bitsize          Bit size of expressions
      --var-num INTEGER               Number of variables
      --input-num INTEGER             Number of inputs
      --random-level INTEGER          Randomness level of inputs 0 means higlhly
                                      biased to use corner-case values (0,1,-1)

      --op-num INTEGER                Operator number
      -v, --verbosity                 increase output verbosity
      --ops TEXT                      specifying operators to uses
      --inputs TEXT                   specifying input vector to use
      --watchdog FLOAT                Activate RAM watchdog (percentage of load
                                      when to stop)

      -c, --cst TEXT                  Constant to add in the generation process
      --linearization                 If set activate linearization of expressions
      -h, --help                      Show this message and exit.


The parameters are the following:

* ``bitsize``: size of expression to generate. At the moment tables cannot contain expression
  of different sizes. *(before looking a table the synthesize will ensure expressions are of same size)*
* ``var-num``: number of variables to include in the grammar
* ``input-num``: Size of the Input vector, on which each expression are evaluated. Too small
  vectors might induce false positives, while too long vectors might uselessly be too computation intensive (15 is good)
* ``random-level``: Number of random value that will be used as inputs. Low value favor values like (0, 1, -1)
* ``op-num``: Unless provided this argument select randomly X operators (among 8 at the moment)
* ``ops``: takes comma separated operators to use for generation (if not provided they are selected randomly)
* ``inputs``: Should be given as a1,b1,c1,a2,b2,c2 where a,b,c are the variable. (a1,b1,c1) is the first input etc.
  If not provided inputs are selected randomly using the ``random-level``.
* ``watchdog``: Generation is highly RAM consuming *(exponential algorithm)*. That enable a watchdog and
  the value is the percentage of RAM above which the generation should be stopped.
* ``cst``: Additional constants to introduce in the generation process.
* ``linearization``: enable linearization of expression using sympy


A very detailed example where all the inputs, ops are defined in advance is the following line:

::

    qsynthesis-table-manager generate tmp.bin -t bin -bs 64 --var-num 3 --max-depth 5 -k 1 \
    --ops SUB,NEG,ADD,XOR,OR --hash-mode MD5 --watchdog 85 --input-num 15 --random-level 7


Once generated. If the generation was performed in 'bin' (thus :py:class:`LookupTableRaw`)
files can be imported in a Level-db database with ``qsynthesis-table-manager import table.bin new_table_ldb``.

.. _label_rest_api:

Oracle REST server
------------------

For large oracles it is convenient to serve them on an API instead of having them locally.
The synthesis is slower but more flexible. Qsynthesis provides ``qsynthesis-table-server``
allowing to serve Level-DB database via a REST API.

The command is straightforward and takes the table in parameter *(and optionally a port)* e.g:

::

    qsynthesis-table-server my_table_leveldb -p 8080
