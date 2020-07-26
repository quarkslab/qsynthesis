from unittest.mock import MagicMock

class TritonContext:
    pass


ARCH = MagicMock()


class MemoryAccess:
    pass


class CALLBACK:
    pass


class MODE:
    pass


class Instruction:
    pass


class AST_REPRESENTATION:
    pass


class OPERAND:
    pass


AST_NODE = MagicMock()
    # LAND
    # LNOT
    # LOR
    # IFF
    # EQUAL
    # DISTINCT
    # BVNOT
    # BVAND
    # BVOR
    # BVXOR
    # BVNEG
    # BVADD
    # BVMUL
    # BVSUB
    # BVSHL
    # BVLSHR
    # BVROL  # Int x Bv -> Bv
    # BVROR  # Int x Bv -> Bv
    # BVUDIV
    # BVUREM
    # BVASHR
    # BVSDIV
    # BVSREM
    # BVSMOD
    # BVXNOR
    # BVNOR
    # BVNAND
    # ZEXT = AST_NODE.ZX  # Int x Bv -> Bv
    # SEXT = AST_NODE.SX  # Int x Bv -> Bv
    # CONCAT = AST_NODE.CONCAT
    # EXTRACT = AST_NODE.EXTRACT  # Int x Int x Bv -> Bv
    # # Other: Bool x BV x BV -> Bv
    # ITE = AST_NODE.ITE
    # # Boolean ops: BV x BV -> Bool
    # UGE = AST_NODE.BVUGE
    # UGT = AST_NODE.BVUGT
    # ULE = AST_NODE.BVULE
    # ULT = AST_NODE.BVULT
    # SLE = AST_NODE.BVSLE
    # SLT = AST_NODE.BVSLT
    # SGE = AST_NODE.BVSGE
    # SGT = AST_NODE.BVSGT


class SYMBOLIC:
    REGISTER_VARIABLE = 1
    MEMORY_VARIABLE = 2

