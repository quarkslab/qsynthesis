Grammar
=======

Lookup tables pre-compute expression on a given grammar. A grammar is composed
of free variables, and operators (unary/binary) that can be applied on these
variables. A lookup table object is instanciated with such grammar class and
an input vector.

.. autoclass:: qsynthesis.grammar.grammar.TritonGrammar
   :members:

Operators used by a grammar instance are defined in the :py:mod:`qsynthesis.grammar.ops` module.

.. automodule:: qsynthesis.grammar.ops
   :members:



Lookup Tables
=============

QSynthesis support any types of tables as long as they implement :py:class:`LookupTable` base class
which interface is given below. At the moment three children clases implement this interface:

* :py:class:`LookupTableLevelDB`: Main table object where entries are stored in a key-value database
  made by called `Level-DB <https://github.com/google/leveldb>`_ made by Google, that guarantee logarithmic
  read in database.
* :py:class:`LookupTableRaw`: Raw tables serialized as binary in custom format file. This tables are
  used for very fast generation and are not meant to be used in practice
* :py:class:`LookupTableREST`: Class interfacing a table being served on a REST API (cf. :ref:`label_rest_api`).
  This table avoids having a whole table locally


.. autoclass:: qsynthesis.tables.base.LookupTable
   :members:


.. autoclass:: qsynthesis.tables.base.HashType
   :members: