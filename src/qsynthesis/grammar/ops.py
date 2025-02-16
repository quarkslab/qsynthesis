from enum import IntEnum
from triton import AST_NODE
from collections import namedtuple
import operator


class BoolOp(IntEnum):
    """
    Enum of SMT boolean operators using Triton AST_NODE
    enum value
    """
    # Bool x Bool -> Bool
    AND = AST_NODE.LAND
    NOT = AST_NODE.LNOT
    LOR = AST_NODE.LOR
    IFF = AST_NODE.IFF
    EQUAL = AST_NODE.EQUAL
    DISTINCT = AST_NODE.DISTINCT


class BvOp(IntEnum):
    """
    Enum of SMT Bitvector operators as declared by Triton AST_NODE
    """
    # Basic ops
    NOT = AST_NODE.BVNOT
    AND = AST_NODE.BVAND
    OR = AST_NODE.BVOR
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


Operator = namedtuple("Operator", "id symbol eval_trit arity commutative id_eq id_zero is_prefix can_overflow bool_ret")


OPERATORS = {               # ID               strop    Trit op            arit comm   id_eq  id_zero is_pfx  can_ov bool_ret
    # BoolOp.EQUAL:    Operator(BoolOp.EQUAL,    "==",    operator.eq,     2,   True,  True,  False,  False,  False, True),
    # BoolOp.DISTINCT: Operator(BoolOp.DISTINCT, "!=",    operator.ne,     2,   True,  False, True,   False,  False, True),
    # BoolOp.IFF:      Operator(BoolOp.IFF,      "iff",   "iff",           2,   False, False, False,  False,  False, True),
    # BoolOp.LOR:      Operator(BoolOp.LOR,      "lor",   "lor",           2,   True,  True,  False,  True,   False, True),
    # BoolOp.AND:      Operator(BoolOp.AND,      "land",  "land",          2,   True,  True,  False,  True,   False, True),
    # BoolOp.NOT:      Operator(BoolOp.NOT,      "lnot",  "lnot",          1,   False, False, False,  True,   False, True),
    BvOp.NOT:        Operator(BvOp.NOT,        "~",     operator.invert,   1,   False, False, False,  True,   False, False),
    BvOp.AND:        Operator(BvOp.AND,        "&",     operator.and_,     2,   True,  True,  False,  False,  False, False),
    BvOp.OR:         Operator(BvOp.OR,         '|',     operator.or_,      2,   True,  True,  False,  False,  False, False),
    BvOp.XOR:        Operator(BvOp.XOR,        '^',     operator.xor,      2,   True,  False, True,   False,  False, False),
    BvOp.NEG:        Operator(BvOp.NEG,        '-',     operator.neg,      1,   False, False, False,  True,   False, False),
    BvOp.ADD:        Operator(BvOp.ADD,        '+',     operator.add,      2,   True,  False, False,  False,  True,  False),
    BvOp.MUL:        Operator(BvOp.MUL,        '*',     operator.mul,      2,   True,  False, False,  False,  True,  False),
    BvOp.SUB:        Operator(BvOp.SUB,        '-',     operator.sub,      2,   False, False, True,   False,  False, False),
    BvOp.SHL:        Operator(BvOp.SHL,        "<<",    operator.lshift,   2,   False, False, False,  False,  True,  False),
    BvOp.LSHR:       Operator(BvOp.LSHR,       ">>",    operator.rshift,   2,   False, False, False,  False,   False, False),

    BvOp.ROL:        Operator(BvOp.ROL,        "bvrol", "bvrol",           2,   False, False, False,  True,   False, False),
    BvOp.ROR:        Operator(BvOp.ROR,        "bvror", "bvror",           2,   False, False, False,  True,   False, False),
    # BvOp.UDIV:       Operator(BvOp.UDIV,       "/",     operator.truediv,2,   False, False, False,  False,  False, False), # TODO: Fix /0
    # BvOp.UREM:       Operator(BvOp.UREM,       "%",     operator.mod,    2,   False, False, False,  False,  False, False), # TODO: Fix /0
    BvOp.ASHR:       Operator(BvOp.ASHR,       "bvashr", "bvashr",         2,   False, False, False,  True,   False, False),
    # BvOp.SDIV:       Operator(BvOp.SDIV,       "bvsdiv","bvsdiv",        2,   False, False, False,  False,  False, False),
    # BvOp.SREM:       Operator(BvOp.SREM,       "bvsrem","bvsrem",        2,   False, False, True,   True,   False, False),
    # BvOp.SMOD:       Operator(BvOp.SMOD,       "bvsmod","bvsmod",        2,   False, False, True,   False,  False, False),
    # BvOp.XNOR:       Operator(BvOp.XNOR,       "bvxnor","bvxnor",        2,   True,  False, False,  True,   False, False),
    # BvOp.NOR:        Operator(BvOp.NOR,        "bvnor", "bvnor",         2,   True,  False, False,  True,   False, False),
    # BvOp.NAND:       Operator(BvOp.NAND,       "bvnand","bvnand",        2,   True,  False, False,  True,   False, False),
    # BvOp.ZEXT:       Operator(BvOp.ZEXT,       "zx",    "zx",            2,   False, True,  False,  True,   False, False),
    # BvOp.SEXT:       Operator(BvOp.SEXT,       "sx",    "sx",            2,   False, False, False,  True,   False, False), # FIXME: operator
    # BvOp.CONCAT:     Operator(BvOp.CONCAT,     "concat","concat",        2,   False, False, False,  True,   False, False), # FIXME: operator
    # BvOp.EXTRACT:    Operator(BvOp.EXTRACT,    "extract","extract",      2,   False, False, False,  True,   False, False), # FIXME: operator
    # BvOp.ITE:        Operator(BvOp.ITE,        "If",     "If",           3,   False, False, False,  True,   False, False),
    # BvOp.UGE:        Operator(BvOp.UGE,        ">=",     operator.ge,    2,   False, True,  False,  False,  False, True),
    # BvOp.UGT:        Operator(BvOp.UGT,        ">",      operator.gt,    2,   False, False, True,   False,  False, True),
    # BvOp.ULE:        Operator(BvOp.ULE,        "<=",     operator.le,    2,   False, True,  False,  False,  False, True),
    # BvOp.ULT:        Operator(BvOp.ULT,        "<",      operator.lt,    2,   False, False, True,   False,  False, True),
    # BvOp.SLE:        Operator(BvOp.SLE,        "bvsle",  "bvsle",        2,   False, True,  False,  True,   False, True),
    # BvOp.SLT:        Operator(BvOp.SLT,        "bvslt",  "bvslt",        2,   False, False, True,   True,   False, True),
    # BvOp.SGE:        Operator(BvOp.SGE,        "bvsge",  "bvsge",        2,   False, True,  False,  True,   False, True),
    # BvOp.SGT:        Operator(BvOp.SGT,        "bvsgt",  "bvsgt",        2,   False, False, True,   True,   False, True)
}
