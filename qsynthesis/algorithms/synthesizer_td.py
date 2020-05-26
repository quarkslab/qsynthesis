import logging
from typing import List, Optional, Tuple, Set, Dict, Generator, Union
import triton
from enum import IntEnum

from qsynthesis.lookuptable import LookupTable
from qsynthesis.tritonast import TritonAst
from qsynthesis.algorithms.synthesizer_base import SynthesizerBase


class TopDownSynthesizer(SynthesizerBase):
    """
    Synthesize with Top-Down ONLY AST search based on Triton AST.
    This synthesis mechanism always converges
    """

    def synthesize(self, ioast: TritonAst, check_sem: bool = False) -> Tuple[TritonAst, bool]:
        ioast = ioast.duplicate()  # Make a copy of the AST to to modify it
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

                logging.debug(f"try synthesis lookup: {cur_ast.pp_str} [{cur_ast.var_num}]")

                synt_res = self.try_synthesis_lookup(cur_ast, check_sem)

                if synt_res is not None:
                    has_been_synthesized = True
                    logging.debug(f"Replace: {cur_ast.pp_str} ===> {synt_res.pp_str}")
                    if is_first:  # root node has been synthesized
                        return synt_res, has_been_synthesized
                    else:
                        next_expr_to_send = synt_res

                is_first = False

        except StopIteration:
            return ioast, has_been_synthesized
