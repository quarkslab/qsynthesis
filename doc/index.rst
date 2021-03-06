QSynthesis documentation
========================

**QSynthesis** is a Python API to perform Greybox synthesis of bitvector
expressions. The base synthesis algorithm is based on
both a blackbox I/O based approach comparing solely input-to-output
pairs to perform the synthesis, but also a whitebox AST search
to synthesize sub-expressions *(if the root expression cannot be synthesized
all at once)*. Also, instead of deriving possible candidates on each expressions
to synthesize, QSynthesis relies on pre-computated expressions stored in a
database. The database keeps the mapping of an expression and its associated output
vector representing its behavior on a set of inputs. Given a bitvector expression
to synthesize, finding a candidate can thus be performed in
near O(1) as the lookup is the lookup cost in database. With the structure
used (Google Level-DB) the cost is thus Log(N) with N the number of entries
in the database.

.. toctree::
   :caption: Getting started
   :maxdepth: 2

    Installation <installation>
    API usage <api_usage>

.. toctree::
    :caption: Plugin Usage
    :maxdepth: 2

    Plugin Introduction <plugin/plugin_usage>

.. toctree::
    :caption: Python API
    :maxdepth: 3

    TritonAst <api/tritonast>
    Synthesis algorithms <api/synthesis>
    Oracles <api/table>
    Symbolic Execution utilities <api/sym_exec>
    Types <api/types>

.. toctree::
    :caption: Advanced Usage
    :maxdepth: 2

    Lookup table Management <dev_doc/table>

