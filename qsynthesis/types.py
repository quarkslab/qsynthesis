from typing import TypeVar, Union, Tuple

# Triton Types
AstNode = TypeVar("AstNode")                        # Triton AstNode object
Register = TypeVar("Register")                      # Triton Register type
SymbolicExpression = TypeVar("SymbolicExpression")  # Triton SymbolicExpression classes

# TritonAst & Grammar Types
Expr = AstNode       # Expression type in the associated grammar (namely a Triton AstNode)

# Lookup Table types
Hash = Union[bytes, int, Tuple[int]]
