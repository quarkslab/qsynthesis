# built-in module
from __future__ import annotations
from typing import Tuple

# qsynthesis deps
from qsynthesis.tritonast import TritonAst
from qsynthesis.algorithms.synthesizer_base import SynthesizerBase, logger


class TopDownSynthesizer(SynthesizerBase):
    """
    Synthesize with a Top-Down **only** AST search based on Triton AST.
    The complexity in worst case of the search is then O(n) with
    n the number of nodes in the AST to synthesize.
    """

    def synthesize(self, ioast: TritonAst, check_sem: bool = False) -> Tuple[TritonAst, bool]:
        """
        Perform the Top-Down search on the ioast to try synthesizing it. The algorithm first
        tries to synthesize the root node. If not successful descend recursively in all children
        of the AST to try substituting sub-ASTs.

        :param ioast: TritonAst object to synthesize
        :param check_sem: boolean one whether to check the semantic equivalence of expression
                          before substituting them. That ensure soundness of the synthesis
        :returns: tuple with new TritonAst and whether some replacement took place or not

        .. warning:: Activating the `check_sem` parameter implies a strong overhead
                     on the synthesis as SMT queries are being performed for any candidates
        """
        ioast = ioast.duplicate()  # Make a copy of the AST to modify it with breaking the one given in parameter
        self.expr_cache = {}
        self.eval_cache = {}
        self.call_to_eval = 0
        self.eval_count = 0

        next_expr_to_send = None
        has_been_synthesized = False
        is_first = True
        try:
            g = ioast.visit_replacement(update=True)
            while True:
                cur_ast = g.send(next_expr_to_send)
                next_expr_to_send = None  # reset next_expr_to_send

                if not cur_ast.has_children():  # If don't have children either constant or variable
                    continue

                logger.debug(f"try synthesis lookup: {cur_ast.pp_str} [{cur_ast.var_num}]")

                synt_res = self.try_synthesis_lookup(cur_ast, check_sem)

                if synt_res is not None:
                    has_been_synthesized = True
                    logger.debug(f"Replace: {cur_ast.pp_str} ===> {synt_res.pp_str}")
                    if is_first:  # root node has been synthesized
                        return synt_res, has_been_synthesized
                    else:
                        next_expr_to_send = synt_res

                is_first = False

        except StopIteration:
            return ioast, has_been_synthesized
