# built-in imports
from typing import Set
from collections import deque, namedtuple

# third-party imports
from triton import AST_NODE

# local imports
from qsynthesis.types import Addr

NodeInfo = namedtuple("NodeInfo", "id expr_off address")


def backslice(root_sym_expr: 'SymbolicExpression') -> Set[Addr]:
    """
    Extract backslice of given symbolic expression as a DGraph
    of instructions ids.

    :param ctx: Triton context to extract the backslice from
    :param root_sym_expr: target symbolic expression
    :return: A DGraph where each node corresponds to an instruction id
    """

    addrs = set()

    # Add root node to graph
    root_node_id, off, addr = [int(x) for x in root_sym_expr.getComment().split('#')]
    addrs.add(addr)

    # Create a queue of (node, parent_id) tuples
    nodes_to_process = deque()
    nodes_to_process.append((root_sym_expr.getAst(), root_node_id))
    handled = set()

    while len(nodes_to_process) > 0:

        node, parent_id = nodes_to_process.popleft()
        skip_node = False

        # If the given AstNode is a reference consume it
        # and then continue using the AstNode of the referenced
        # symbolic expression
        while node.getType() == AST_NODE.REFERENCE:
            # Get referenced symbolic expr
            se = node.getSymbolicExpression()

            # Get expression id
            instr_id, expr_nb, instr_addr = [int(x) for x in se.getComment().split('#')]
            expr_id = (instr_id, expr_nb)

            addrs.add(instr_addr)

            # Do not process this node if it has already been handled
            # NOTE: since we could come from an unseen parent we add
            # an edge before going to the next node
            if expr_id in handled:
                skip_node = True
                break
            else:
                handled.add(expr_id)

            # Move to the referenced AstNode
            parent_id = instr_id
            node = node.getSymbolicExpression().getAst()

        if skip_node:
            continue

        # Process children
        for child in node.getChildren():
            nodes_to_process.append((child, parent_id))

    return addrs
