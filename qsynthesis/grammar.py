from triton import AST_NODE
from enum import IntEnum
from typing import Dict, List, Tuple
import random


class BoolOp(IntEnum):
    AND = AST_NODE.LAND
    NOT = AST_NODE.LNOT
    LOR = AST_NODE.LOR
    IFF = AST_NODE.IFF
    EQUAL = AST_NODE.EQUAL
    DISTINCT = AST_NODE.DISTINCT


class BvOp(IntEnum):
    #Basic ops
    NOT = AST_NODE.BVNOT
    AND = AST_NODE.BVAND
    OR  = AST_NODE.BVOR
    XNOR = AST_NODE.BVXNOR
    NOR = AST_NODE.BVNOR
    NAND = AST_NODE.BVNAND
    XOR = AST_NODE.BVXOR
    NEG = AST_NODE.BVNEG
    ADD = AST_NODE.BVADD
    MUL = AST_NODE.BVMUL
    SUB = AST_NODE.BVSUB
    #Extended
    SHL = AST_NODE.BVSHL
    LSHR = AST_NODE.BVLSHR
    ROL = AST_NODE.BVROL
    ROR = AST_NODE.BVROR
    UDIV = AST_NODE.BVUDIV
    UREM = AST_NODE.BVUREM
    ASHR = AST_NODE.BVASHR
    #Extended extended
    SDIV = AST_NODE.BVSDIV
    SREM = AST_NODE.BVSREM
    SMOD = AST_NODE.BVSMOD
    ZEXT = AST_NODE.ZX
    SEXT = AST_NODE.SX
    # Change bit size
    CONCAT = AST_NODE.CONCAT
    EXTRACT = AST_NODE.EXTRACT
    #Other
    ITE = AST_NODE.ITE
    EQ = AST_NODE.EQUAL
    #Boolean ops
    UGE = AST_NODE.BVUGE
    UGT = AST_NODE.BVUGT
    ULE = AST_NODE.BVULE
    ULT = AST_NODE.BVULT
    SLE = AST_NODE.BVSLE
    SLT = AST_NODE.BVSLT
    SGE = AST_NODE.BVSGE
    SGT = AST_NODE.BVSGT


class TritonGrammar(object):
    """
    Triton Grammar
    """

    def __init__(self, vars: List[Tuple[str, int]], ops: List[BvOp]):
        self.ops = ops
        self.vars_dict = {x[0]: x[1] for x in vars}  # Dict of str->size
        self.vars = list(self.vars_dict.keys())

        self.size = self.vars_dict[self.vars[0]]  # take size of the first var as they all have the same size

    @property
    def non_terminal_tokens(self) -> List:
        return self.ops

    def gen_test_inputs(self, n: int) -> List[Dict[str, int]]:
        return [{var: random.getrandbits(self.vars_dict[var]) for var in self.vars} for _ in range(n)]

    def str_to_expr(self, s, *args):
        expr = args[0]
        return expr.normalized_str_to_ast(s)

    def to_dict(self) -> Dict:
        return {
            'vars': [(n, sz) for n, sz in self.vars_dict.items()],
            'operators': [x.name for x in self.ops]
        }

    @staticmethod
    def from_dict(g_dict: Dict) -> 'TritonGrammar':
        return TritonGrammar(g_dict['vars'], [BvOp[x] for x in g_dict['operators']])

    @staticmethod
    def dump_inputs(inputs):
        return [{(k, 0): v for k, v in inp.items()} for inp in inputs]

    @staticmethod
    def load_inputs(inp_l):
        return [{x[0]: v for x, v in inp.items()} for inp in inp_l]
