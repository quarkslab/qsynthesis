# qsynthesis

Main qsynthesis python module

# Installation

## Module installation

QSynthesis works on **Triton** ASTs thus it has to be installed first. It requires version `>= 0.8`.
Triton can be found here: https://github.com/JonathanSalwan/Triton

Then Qsynthesis can simply be installed with:

```bash
python3 setup.py install
```

## IDA plugin installation

QSynthesis is integrated in IDA and can works in two manners:

* As a standalone plugin working on static data. In this mode synthesis is bound to basic blocks
* In collaboration with **Qtrace-IDA** to synthesize expressions along a trace. This mode allows synthesizing
  expressions accross basic blocks and functions. As long as expressions are kept 

The python module should be installed first. The plugin can then simply be installed with:

```bash
cp ida_plugin/qsynthesis_plugin.py $IDA_HOME/plugins
```

## Usage

TODO


# TODO

