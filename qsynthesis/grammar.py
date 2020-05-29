from triton import AST_NODE
from enum import IntEnum
from typing import Dict, List, Tuple
import random
import operator
from collections import namedtuple

import pydffi

SZ = 64  # FIXME: make it variadic
SZ_MASK = 0xffffffffffffffff  # FIXME: make it variadic

# First, declare an FFI context
CODE = '''
#include <stdint.h> 
uint64_t add(uint64_t a, uint64_t b) { return a+b; }
uint64_t sub(uint64_t a, uint64_t b) { return a-b; }
uint64_t mul(uint64_t a, uint64_t b) { return a*b; }
uint64_t udiv(uint64_t a, uint64_t b) { return a/b; }
uint64_t usub(uint64_t a) { return -a; }
uint64_t invert(uint64_t a) { return ~a; }
uint64_t to_uint(int64_t a) { return (uint64_t) a; }
uint64_t urem(uint64_t a, uint64_t b) { return  b==0? a: a % b ; }
uint64_t ashr(int64_t a, uint64_t b) { return  a >> b; }
uint64_t sle(int64_t a, int64_t b) { return a <= b ; }        
uint64_t slt(int64_t a, int64_t b) { return a < b ; }
uint64_t sge(int64_t a, int64_t b) { return a >= b ; }
uint64_t sgt(int64_t a, int64_t b) { return a > b ; }
uint64_t lshift(uint64_t a, uint64_t b) { return a << b; }
uint64_t rshift(uint64_t a, uint64_t b) { return a >> b; }
uint64_t rol(uint8_t i, uint64_t n) { return (n << i) | (n >> (8*sizeof(n) - i)); }
uint64_t ror(uint8_t i, uint64_t n) { return (n >> i)|(n << (8*sizeof(n) - i)); }
uint64_t mod(uint64_t a, uint64_t b) { return a % b; }
'''

ffi_ctx = pydffi.FFI()
CU = ffi_ctx.compile(CODE)

to_uint = CU.funcs.to_uint

#
# def bnd(f):
#     return lambda x, y: SZ_MASK & f(x, y)

#
# def left_rotate(i, n):
#     return (n << i) | (n >> (SZ - i))
#
#
# def right_rotate(i, n):
#     return (n >> i)|(n << (SZ - i))


def sign_ext(sz, ext, v):
    if v < 0:
        return v
    if sz == v.bit_length():  # on integer means it is one
        ext_bits = (pow(2, ext)-1)
        return concat(sz, ext_bits, v)
    else:
        return v


def concat(sz, a, b):
    return (a << sz) | b


def extract(to, frm, v):
    return v >> frm & (pow(2, to-frm+1)-1)


def ite(b_cond, a, b):
    return a if b_cond else b


class BoolOp(IntEnum):
    # Bool x Bool -> Bool
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
    XOR = AST_NODE.BVXOR
    NEG = AST_NODE.BVNEG
    ADD = AST_NODE.BVADD
    MUL = AST_NODE.BVMUL
    SUB = AST_NODE.BVSUB
    SHL = AST_NODE.BVSHL
    LSHR = AST_NODE.BVLSHR
    ROL = AST_NODE.BVROL  # Int x Bv -> Bv
    ROR = AST_NODE.BVROR  # Int x Bv -> Bv
    UDIV = AST_NODE.BVUDIV
    UREM = AST_NODE.BVUREM
    ASHR = AST_NODE.BVASHR
    SDIV = AST_NODE.BVSDIV
    SREM = AST_NODE.BVSREM
    SMOD = AST_NODE.BVSMOD
    # Extended
    XNOR = AST_NODE.BVXNOR
    NOR = AST_NODE.BVNOR
    NAND = AST_NODE.BVNAND
    # Change bit size
    ZEXT = AST_NODE.ZX  # Int x Bv -> Bv
    SEXT = AST_NODE.SX  # Int x Bv -> Bv
    CONCAT = AST_NODE.CONCAT
    EXTRACT = AST_NODE.EXTRACT  # Int x Int x Bv -> Bv
    # Other: Bool x BV x BV -> Bv
    ITE = AST_NODE.ITE
    # Boolean ops: BV x BV -> Bool
    UGE = AST_NODE.BVUGE
    UGT = AST_NODE.BVUGT
    ULE = AST_NODE.BVULE
    ULT = AST_NODE.BVULT
    SLE = AST_NODE.BVSLE
    SLT = AST_NODE.BVSLT
    SGE = AST_NODE.BVSGE
    SGT = AST_NODE.BVSGT


Operator = namedtuple("Operator", "id symbol op_triton eval arity commutative id_eq id_zero is_prefix can_overflow bool_ret")

OPERATORS = {               # ID               strop    Trit op         Py op                arit comm   id_eq  id_zero is_pfx  can_ov bool_ret
    BoolOp.EQUAL:    Operator(BoolOp.EQUAL,    "==",    operator.eq,    operator.eq,         2,   True,  True,  False,  False,  False, True),
    BoolOp.DISTINCT: Operator(BoolOp.DISTINCT, "!=",    operator.ne,    operator.ne,         2,   True,  False, True,   False,  False, True),
    BoolOp.IFF:      Operator(BoolOp.IFF,      "iff",   "iff",          None,                2,   False, False, False,  False,  False, True),
    BoolOp.LOR:      Operator(BoolOp.LOR,      "lor",   "lor",          lambda x,y: x or y,  2,   True,  True,  False,  True,   False, True),
    BoolOp.AND:      Operator(BoolOp.AND,      "land",  "land",         lambda x,y: x and y, 2,   True,  True,  False,  True,   False, True),
    BoolOp.NOT:      Operator(BoolOp.NOT,      "lnot",  "lnot",         lambda x: not x,     1,   False, False, False,  True,   False, True),
    BvOp.NOT:        Operator(BvOp.NOT,        "~",     operator.invert,CU.funcs.invert,     1,   False, False, False,  True,   False, False),
    BvOp.AND:        Operator(BvOp.AND,        "&",     operator.and_,  operator.and_,       2,   True,  True,  False,  False,  False, False),
    BvOp.OR:         Operator(BvOp.OR,         '|',     operator.or_,   operator.or_,        2,   True,  True,  False,  False,  False, False),
    BvOp.XOR:        Operator(BvOp.XOR,        '^',     operator.xor,   operator.xor,        2,   True,  False, True,   False,  False, False),
    BvOp.NEG:        Operator(BvOp.NEG,        '-',     operator.neg,   CU.funcs.usub,       1,   False, False, False,  True,   False, False),
    BvOp.ADD:        Operator(BvOp.ADD,        '+',     operator.add,   CU.funcs.add,        2,   True,  False, False,  False,  True,  False),
    BvOp.MUL:        Operator(BvOp.MUL,        '*',     operator.mul,   CU.funcs.mul,        2,   True,  False, False,  False,  True,  False),
    BvOp.SUB:        Operator(BvOp.SUB,        '-',     operator.sub,   CU.funcs.sub,        2,   False, False, True,   False,  False, False),
    BvOp.SHL:        Operator(BvOp.SHL,        "<<",    operator.lshift,CU.funcs.lshift,     2,   False, False, False,  False,  True,  False),
    BvOp.LSHR:       Operator(BvOp.LSHR,       ">>",    operator.rshift,CU.funcs.rshift,     2,   False, False, False,  True,   False, False),
    BvOp.ROL:        Operator(BvOp.ROL,        "bvrol", "bvrol",        CU.funcs.rol,        2,   False, False, False,  True,   False, False),
    BvOp.ROR:        Operator(BvOp.ROR,        "bvror", "bvror",        CU.funcs.ror,        2,   False, False, False,  True,   False, False),
    BvOp.UDIV:       Operator(BvOp.UDIV,       "/",     operator.truediv,CU.funcs.udiv,      2,   False, False, False,  True,   False, False),
    BvOp.UREM:       Operator(BvOp.UREM,       "%",     operator.mod,   CU.funcs.urem,       2,   False, False, False,  False,  False, False),
    BvOp.ASHR:       Operator(BvOp.ASHR,       ">>",    "bvashr",       CU.funcs.ashr,       2,   False, False, False,  False,  False, False),
    BvOp.SDIV:       Operator(BvOp.SDIV,       "bvsdiv","bvsdiv",       operator.truediv,    2,   False, False, False,  False,  False, False),
    BvOp.SREM:       Operator(BvOp.SREM,       "bvsrem","bvsrem",       None,                2,   False, False, True,   True,   False, False),
    BvOp.SMOD:       Operator(BvOp.SMOD,       "bvsmod","bvsmod",       CU.funcs.mod,        2,   False, False, True,   False,  False, False),
    BvOp.XNOR:       Operator(BvOp.XNOR,       "bvxnor","bvxnor",       lambda x,y:~(x ^ y), 2,   True,  False, False,  True,   False, False),
    BvOp.NOR:        Operator(BvOp.NOR,        "bvnor", "bvnor",        lambda x, y:~(x | y),2,   True,  False, False,  True,   False, False),
    BvOp.NAND:       Operator(BvOp.NAND,       "bvnand","bvnand",       lambda x, y:~(x & y),2,   True,  False, False,  True,   False, False),
    BvOp.ZEXT:       Operator(BvOp.ZEXT,       "zx",    "zx",           lambda y, x: x,      2,   False, True,  False,  True,   False, False),
    BvOp.SEXT:       Operator(BvOp.SEXT,       "sx",    "sx",           sign_ext,            2,   False, False, False,  True,   False, False), # FIXME: operator
    BvOp.CONCAT:     Operator(BvOp.CONCAT,     "concat","concat",       concat,              2,   False, False, False,  True,   False, False), # FIXME: operator
    BvOp.EXTRACT:    Operator(BvOp.EXTRACT,    "extract","extract",     extract,             2,   False, False, False,  True,   False, False), # FIXME: operator
    BvOp.ITE:        Operator(BvOp.ITE,        "If",     "If",          ite,                 3,   False, False, False,  True,   False, False),
    BvOp.UGE:        Operator(BvOp.UGE,        ">=",     operator.ge,   operator.ge,         2,   False, True,  False,  False,  False, True),
    BvOp.UGT:        Operator(BvOp.UGT,        ">",      operator.gt,   operator.gt,         2,   False, False, True,   False,  False, True),
    BvOp.ULE:        Operator(BvOp.ULE,        "<=",     operator.le,   operator.le,         2,   False, True,  False,  False,  False, True),
    BvOp.ULT:        Operator(BvOp.ULT,        "<",      operator.lt,   operator.lt,         2,   False, False, True,   False,  False, True),
    BvOp.SLE:        Operator(BvOp.SLE,        "bvsle",  "bvsle",       CU.funcs.sle,        2,   False, True,  False,  True,   False, True),
    BvOp.SLT:        Operator(BvOp.SLT,        "bvslt",  "bvslt",       CU.funcs.slt,        2,   False, False, True,   True,   False, True),
    BvOp.SGE:        Operator(BvOp.SGE,        "bvsge",  "bvsge",       CU.funcs.sge,        2,   False, True,  False,  True,   False, True),
    BvOp.SGT:        Operator(BvOp.SGT,        "bvsgt",  "bvsgt",       CU.funcs.sgt,        2,   False, False, True,   True,   False, True)
}


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
    def non_terminal_operators(self) -> List[Operator]:
        return [OPERATORS[x] for x in self.ops]

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
