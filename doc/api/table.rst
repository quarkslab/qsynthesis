Grammar
=======

Oracles contain pre-computed expression on a given grammar. A grammar is composed
of variables, and operators (unary/binary) that can be applied on these
variables. An oracle object is instanciated with such grammar class and an input vector.

.. autoclass:: qsynthesis.grammar.grammar.TritonGrammar
   :members:


Oracles
=======

QSynthesis support any types of oracles as long as they implement :py:class:`InputOutputOracle` base class
which interface is given below. At the moment two children classes implement this interface:

* :py:class:`InputOutputOracleLevelDB`: Main table object where entries are stored in a key-value database
  made by called `Level-DB <https://github.com/google/leveldb>`_ made by Google, that guarantee logarithmic
  read in database.
* :py:class:`LookupTableREST`: Class interfacing a ``InputOutputOracleLevelDB`` being served on a REST API
  (cf. :ref:`label_rest_api`). This table avoids having a whole table locally.


.. autoclass:: qsynthesis.tables.base.InputOutputOracle
   :members:
