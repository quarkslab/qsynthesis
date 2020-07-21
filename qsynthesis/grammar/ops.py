from enum import IntEnum
from triton import AST_NODE
from collections import namedtuple


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


Operator = namedtuple("Operator", "id symbol eval_trit eval eval_a arity commutative id_eq id_zero is_prefix can_overflow bool_ret")
