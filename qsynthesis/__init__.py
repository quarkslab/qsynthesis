from qsynthesis.algorithms import TopDownSynthesizer, PlaceHolderSynthesizer
from qsynthesis.grammar import TritonGrammar, BvOp
from qsynthesis.tables import InputOutputOracleREST, InputOutputOracleLevelDB, HashType
from qsynthesis.utils.symexec import SimpleSymExec
from qsynthesis.tritonast import TritonAst

import logging

__version__ = "0.1.2"

# Simple object used to retrieve the plugin in IDA
qsynthesis_plugin = None


def enable_logging(level: int = 0):
    logger = logging.getLogger("qsynthesis")
    if level:
        logger.setLevel(level)
    logger.disabled = False


def disable_logging():
    logger = logging.getLogger("qsynthesis")
    logger.disabled = True
