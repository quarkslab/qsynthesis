.. _qsynthesis_api_usage:

API Usage
=========

Use-case: Synthesis in pure symbolic with Triton
---------------------------------------------------

While working on an execution trace is very powerful thanks to concretization that can be made
we often can't executed the program or the execution does not reach the intended location. As such,
this use-case shows a very similar one where instruction are provided by the use directly.

First lets consider the following bytes representing the x86_64 computation of an obfuscated
expression.

.. code-block:: python

    INSTRUCTIONS = [b'U', b'H\x89\xe5', b'H\x89}\xf8', b'H\x89u\xf0', b'H\x89U\xe8', b'H\x89M\xe0', b'L\x89E\xd8',
                    b'H\x8bE\xf0', b'H#E\xe0', b'H\x89\xc2', b'H\x8bE\xf0', b'H\x0bE\xe0', b'H\x0f\xaf\xd0', b'H\x8bE\xe0',
                    b'H\xf7\xd0', b'H#E\xf0', b'H\x89\xc1', b'H\x8bE\xf0', b'H\xf7\xd0', b'H#E\xe0', b'H\x0f\xaf\xc1',
                    b'H\x01\xc2', b'H\x8bE\xe0', b'H\x0f\xaf\xc0', b'H\x89\xd6', b'H!\xc6', b'H\x8bE\xf0', b'H#E\xe0',
                    b'H\x89\xc2', b'H\x8bE\xf0', b'H\x0bE\xe0', b'H\x0f\xaf\xd0', b'H\x8bE\xe0', b'H\xf7\xd0', b'H#E\xf0',
                    b'H\x89\xc1', b'H\x8bE\xf0', b'H\xf7\xd0', b'H#E\xe0', b'H\x0f\xaf\xc1', b'H\x01\xc2', b'H\x8bE\xe0',
                    b'H\x0f\xaf\xc0', b'H\t\xd0', b'H)\xc6', b'H\x89\xf0', b'H\x83\xe8\x01', b'H3E\xf0', b'H\x89\xc2',
                    b'H\x8bE\xf0', b'H#E\xe0', b'H\x89\xc1', b'H\x8bE\xf0', b'H\x0bE\xe0', b'H\x0f\xaf\xc8', b'H\x8bE\xe0',
                    b'H\xf7\xd0', b'H#E\xf0', b'H\x89\xc6', b'H\x8bE\xf0', b'H\xf7\xd0', b'H#E\xe0', b'H\x0f\xaf\xc6',
                    b'H\x01\xc1', b'H\x8bE\xe0', b'H\x0f\xaf\xc0', b'H1\xc8', b'H#E\xf0', b'H\x01\xc0', b'H)\xc2',
                    b'H\x89\xd0', b']', b'\xc3']

The first thing to do is to executed these instruction with triton. For that Qsynthesis
provides an utility class :py:class:`SimpleSymExec` (cf: :ref:`label_simplesymexec`) facilitating
various tasks. It takes the architecture in parameter. We arbritrarily initialize ``rip`` and
``rsp`` to arbitrary addresses and feed all instructions to that wrapper.

.. code-block:: python

    from qsynthesis.utils.symexec import SimpleSymExec

    symexec = SimpleSymExec("x86_64")              # Initialize it with the intended architecture
    symexec.initialize_register('rip', 0x40B160)   # arbitrary address
    symexec.initialize_register('rsp', 0x800000)   # arbitrary address
    for opcode in INSTRUCTIONS:
        symexec.execute(opcode)                    # Execute the given opcode
    rax = symexec.get_register_ast("rax")          # Retrieve rax AST after executing instructions


As of now the ``rax`` expression can be synthesized by instanciating the synthesizer with
an oracle table.

.. code-block:: python

    from qsynthesis import TopDownSynthesizer, InputOutputOracleLevelDB

    table = InputOutputOracleLevelDB.load("my_leveldb_table/")  # Load the lookup table database

    synthesizer = TopDownSynthesizer(table)       # Instanciate the synthesize of the table
    synt_rax, simp_bool = synthesizer.synthesize(rax)  # Trigger synthesize of the rax expression

    # Print synthesis results
    print(f"simplified: {simp_bool}")
    print(f"synthesized expression: {synt_rax.pp_str}")
    print(f"size: {rax.node_count} -> {synt_rax.node_count}")
