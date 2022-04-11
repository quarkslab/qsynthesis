# Qsynthesis

QSynthesis is a Python3 API to perform I/O based program synthesis
of bitvector expressions. It aims at facilitating code deobfuscation.
The algorithm is greybox approach combining both a blackbox I/O based
synthesis and a whitebox AST search to synthesize sub-expressions *(if
the root node cannot be synthesized)*. 

This algorithm as originaly been described at the BAR academic workshop:

* [QSynth: A Program Synthesis based Approach for Binary Code Deobfuscation](https://archive.bar/pdfs/bar2020-preprint9.pdf)
  (benchmark used are available: [here](https://github.com/werew/qsynth-artifacts))

The code has been release as part of the following Black Hat talk:

* [Greybox Program Synthesis: A New Approach to Attack Dataflow Obfuscation](https://www.blackhat.com/us-21/briefings/schedule/index.html#greybox-program-synthesis-a-new-approach-to-attack-dataflow-obfuscation-22930)

**Disclaimer: This framework is experimental, and shall only be used for experimentation purposes.
It mainly aims at stimulating research in this area.**


## Documentation

The installation, examples, and API documentation is available on the dedicated documentation: [Documentation](https://quarkslab.github.io/qsynthesis)


## Functionalities

The core synthesis is based on [Triton](https://triton.quarkslab.com) symbolic engine on which is built
the whole framework. It provides the following functionalities:

* synthesis of bitvector expressions
* ability to check through SMT the semantic equivalence of synthesized expressions
* ability to synthesize constants *(if the expression encode a constant)*
* ability to improve oracles (pre-computed tables) overtime through a learning mechanism
* ability to reassemble synthesized expression back to assembly
* ability to serve oracles through a REST API to facilitate the synthesis usage  
* an IDA plugin providing an integration of the synthesis


## Quick start

### Installation

In order to work Triton first has to be installed: [install documentation](https://triton.quarkslab.com/documentation/doxygen/index.html#install_sec).
Triton does not automatically install itself in a virtualenv, copy it in your venv or use --system-site-packages when configuring your venv.

Then:

    $ git clone https://github.com/quarkslab/qsynthesis.git
    $ cd qsynthesis
    $ pip3 install '.[all]'

The ``[all]`` will installed all dependencies *(see the documentation for a light install)*.

### Table generation

The synthesis algorithm requires generating oracle tables derived from a grammar *(a
set of variables and operators)*. Qsynthesis installation provides the utility ``qsynthesis-table-manager``
enabling manipulating tables. The following command generate a table with 3 variables of 64 bits,
5 operators using a vector of 16 inputs. We limit the generation to 5 million entries.

    $ qsynthesis-table-manager generate -bs 64 --var-num 3 --input-num 16 --random-level 5 --ops AND,NEG,MUL,XOR,NOT --watchdog 80 --limit 5000000 my_oracle_table
    Generate Table
    Watchdog value: 80.0
    Depth 2 (size:3) (Time:0m0.23120s)
    Depth 3 (size:21) (Time:0m0.23198s)
    Depth 4 (size:574) (Time:0m0.26068s)
    Depth 5 (size:400858) (Time:0m21.23231s)
    Threshold reached, generation interrupted
    Stop required
    Depth 5 (size:5000002) (Time:4m52.56009s) [RAM:9.52Gb]



Note: The generation process is RAM consuming the ``--watchdog`` enables setting a
percentage of the RAM above which the generation is interrupted.

### Synthesizing a bitvector expression

We then can try simplifying a seemingly obfuscated expression with:

```python
from qsynthesis import SimpleSymExec, TopDownSynthesizer, InputOutputOracleLevelDB

blob = b'UH\x89\xe5H\x89}\xf8H\x89u\xf0H\x89U\xe8H\x89M\xe0L\x89E\xd8H\x8bE' \
       b'\xe0H\xf7\xd0H\x0bE\xf8H\x89\xc2H\x8bE\xe0H\x01\xd0H\x8dH\x01H\x8b' \
       b'E\xf8H+E\xe8H\x8bU\xe8H\xf7\xd2H\x0bU\xf8H\x01\xd2H)\xd0H\x83\xe8' \
       b'\x02H!\xc1H\x8bE\xe0H\xf7\xd0H\x0bE\xf8H\x89\xc2H\x8bE\xe0H\x01\xd0' \
       b'H\x8dp\x01H\x8bE\xf8H+E\xe8H\x8bU\xe8H\xf7\xd2H\x0bU\xf8H\x01\xd2' \
       b'H)\xd0H\x83\xe8\x02H\t\xf0H)\xc1H\x89\xc8H\x83\xe8\x01]\xc3'

# Perform symbolic execution of the instructions
symexec = SimpleSymExec("x86_64")
symexec.initialize_register('rip', 0x40B160)  # arbitrary address
symexec.initialize_register('rsp', 0x800000)  # arbitrary stack
symexec.execute_blob(blob, 0x40B160)
rax = symexec.get_register_ast("rax")  # retrieve rax register expressions

# Load lookup tables
ltm = InputOutputOracleLevelDB.load("my_oracle_table")

# Perform Synthesis of the expression
synthesizer = TopDownSynthesizer(ltm)
synt_rax, simp = synthesizer.synthesize(rax)

print(f"expression: {rax.pp_str}")
print(f"synthesized expression: {synt_rax.pp_str} [{simp}]")
```

## Limitations

* synthesis accuracy limited by pre-computed tables exhaustivness
* table generation limited by RAM consumption
* reassembly cannot involve memory variable, destination is necessarily a register and
  architecture depends on llvmlite *(thus mostly x86_64)*
* the code references trace-based synthesis which is disabled *(as the underlying
  framework is not yet open-source)*  

## Authors

* Robin David (@RobinDavid), Quarkslab

## Contributors

Huge thanks to contributors to this research:

* Luigi Coniglio
* Jonathan Salwan
