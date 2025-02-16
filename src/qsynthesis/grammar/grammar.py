# Standard modules
from __future__ import annotations
import random

# Qsynthesis types
from qsynthesis.types import BitSize, Char, Input, Dict, List, Tuple
from qsynthesis.tritonast import TritonAst
from qsynthesis.grammar.ops import BvOp, Operator, OPERATORS


class TritonGrammar(object):
    """
    Triton Grammar class. It represent a set of operators, and variables
    of a given size (only 64 bits at the moment).
    """

    def __init__(self, vars: List[Tuple[Char, BitSize]], ops: List[BvOp]):
        """
        Constructor taking a set of variables (name and size) and a set of operators.

        :param vars: list of tuple of (name ,size)
        :type vars: List[Tuple[:py:obj:`qsynthesis.types.Char`, :py:obj:`qsynthesis.types.BitSize`]]
        :param ops: list of BvOp representing operators
        :type ops: List[BvOp]
        """
        self.ops = ops
        self.vars_dict = {x[0]: x[1] for x in vars}  # Dict of str->size
        self.vars = list(self.vars_dict.keys())

        self.size = self.vars_dict[self.vars[0]]  # take size of the first var as they all have the same size

    @property
    def non_terminal_operators(self) -> List[Operator]:
        """
        Return the list of non-terminal operators. All unary and
        binary operators are non terminal as they can be derived.

        :return: list of operators namedtuples
        """
        return [OPERATORS[x] for x in self.ops]

    def gen_test_inputs(self, n: int) -> List[Input]:
        """
        Generate a list of ``n`` input. Thus it generate a random
        valuation for each variables of the grammar and that n times.

        :param n: Number of Input to generate (size of the list)
        :type n: int
        :returns: list of inputs
        :rtype: List[:py:obj:`qsynthesis.types.Input`]
        """
        return [{var: random.getrandbits(self.vars_dict[var]) for var in self.vars} for _ in range(n)]

    def str_to_expr(self, s: str, *args) -> TritonAst:
        """
        Convert a string in the format of the grammar into a TritonAst.
        In practice an args[0] should be a TritonAst from which to spawn
        a new TritonAst. That is required to get the same mapping of normalized
        variables than the one used by expr.

        :param s: expression string to convert to TritonAst
        :return: the TritonAst representing the expressions string
        :raises: NameError, TypeError
        """
        expr = args[0]
        return expr.normalized_str_to_ast(s)

    def to_dict(self) -> Dict:
        """
        Return a dictionnary representation of the grammar.
        This is used for serialization in database etc.
        """
        return dict(
            vars=[(n, sz) for n, sz in self.vars_dict.items()],
            operators=[x.name for x in self.ops]
        )

    @staticmethod
    def from_dict(g_dict: Dict) -> 'TritonGrammar':
        """
        Static method instanciating a TritonGrammar from its representation as
        a dictionnary.

        :param g_dict: dictionarry representation of the grammar
        :returns: TritonGrammar object
        """
        return TritonGrammar(g_dict['vars'], [BvOp[x] for x in g_dict['operators']])
