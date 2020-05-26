import logging
from qsynthesis.lookuptable import LookupTable
from typing import List, Optional, Tuple, Set, Dict, Generator, Union
import triton
from enum import IntEnum
from orderedset import OrderedSet

from qsynthesis.tritonast import TritonAst
from qsynthesis.algorithms.synthesizer_base import SynthesizerBase

AstType = IntEnum("AstNode", {k: v for k, v in triton.AST_NODE.__dict__.items() if isinstance(v, int)})


class YieldT(IntEnum):
    TopBottom = 1
    BottomUp = 2


class TopDownBottomUpSynthesizer(SynthesizerBase):
    """
    Synthesize with Top-Down then Bottom-Up AST search based on
    Triton AST.
    This synthesis mechanism always converges
    """

    def __init__(self, ltms: List[LookupTable], only_first_match: bool = False):
        super(TopDownBottomUpSynthesizer, self).__init__(ltms, only_first_match)
        self.total_repl_td = 0
        self.total_repl_bu = 0

    def synthesize(self, ioast: TritonAst, check_sem: bool = False) -> Tuple[TritonAst, bool]:
        ioast = ioast.duplicate()  # Make a copy of the ast not tamper it
        self.expr_cache = {}
        self.eval_cache = {}
        self.call_to_eval = 0
        self.eval_count = 0

        expr_repl_visitor = self._visit_replacement(ioast)
        new_expr_to_send = None
        expr_modified = False

        while 1:
            #logging.debug(f"Sending: {new_expr_to_send}")
            cur_expr, info = expr_repl_visitor.send(new_expr_to_send)  # Iterate generator
            #logging.debug(f"Receiving: {cur_expr}")
            if isinstance(info, bool):
                final_expr = cur_expr[0]
                #logging.debug("final expression:", final_expr.replace("\n"," "))
                final_expr.update_all()
                return final_expr, expr_modified

            else:  # Try synthesizing expression
                logging.debug(f"try synthesis lookup: {cur_expr.pp_str if cur_expr.node_count < 50 else 'too large'}")
                synt_res = self.try_synthesis_lookup(cur_expr, check_sem)
                if synt_res is not None:
                    expr_modified = True
                new_expr_to_send = synt_res  # Send the result to the generator (thus either new expr or None)

    def _visit_replacement(self, ast: TritonAst, iff_replace=True, update=True) -> Generator[Tuple[TritonAst, Union[YieldT, bool]], TritonAst, None]:
        chs = ast.get_children()
        if not chs:  # If arity is 0 (variable and constant)
            yield ast, False  # Yield the final expr as-is
        else:
            # Normally here in Z3Ast check here if expression is bool
            rep = yield ast, YieldT.TopBottom  # First (Top-Down yield)

            if rep is not None:  # We should replace this expr
                self.total_repl_td += 1
                yield rep, True  # Final (yield)
            else:
                reps = {}  # index -> (expr, bool) the boolean true if expr was replaced
                for i, c in enumerate(chs):  # Iterate (and simplify) all childrens
                    g = self._visit_replacement(c, iff_replace, update)
                    recv = None
                    while 1:
                        it, info = g.send(recv)  # forward the new expression to the sub-generator
                        if isinstance(info, bool):  # is a final yield
                            reps[i] = it, info
                            break
                        else:
                            recv = yield it, info  # re-yield it above (and receive a new expression)

                if sum(x[1] for x in reps.values()) or not iff_replace:  # at least on replacement took place among children
                    for i, (e, b) in reps.items():
                        if b:  # expr has been replaced
                            ast.set_child(i, e)  # replace the child
                    if update:  # if one of the children got replaced and update, update fields
                        ast.update()
                    rep = yield ast, YieldT.BottomUp  # Second (Bottom-Up yield) (if not bool expr)
                    if rep is None:
                        yield ast, True  # Final yield
                    else:
                        self.total_repl_bu += 1
                        yield rep, True  # Final yield
                else:
                    yield ast, False

    def _visit_replacement_bfs(self, orig_ast: TritonAst, iff_replace=True, update=True) -> Generator[Tuple[TritonAst, Union[YieldT, bool]], TritonAst, None]:
        worklist = OrderedSet([orig_ast])
        bottomup_worklist = OrderedSet([orig_ast])
        modified = False
        mode = YieldT.TopBottom

        while worklist:
            ast = worklist.pop(0)
            logging.debug(f"\n[visit_replacement] pop [{ast.ptr_id}]({len(ast.get_children())}) {ast.pp_str}  worklist:[{len(worklist)}] bu_worklist:[{len(bottomup_worklist)}]")

            if not ast.is_leaf():
                rep = yield ast, mode  # Current yield (TopB, or BotUp)
            else:
                rep = None

            if rep is not None:  # We should replace this expr
                modified = True
                self.total_repl_td += 1 if mode == YieldT.TopBottom else 0
                self.total_repl_bu += 1 if mode == YieldT.BottomUp else 0
                if ast.is_root():
                    logging.debug(f"YIeld new root ahead of time: {rep.pp_str}")
                    yield rep, modified  # Final yield (ahead of time)
                else:
                    logging.debug(f"replace_self: {str(ast.expr)} => {rep.pp_str}")
                    ast.replace_self(rep, update_parents=True)
            else:
                if mode == YieldT.TopBottom:
                    logging.debug(f"[visit_replacement] worklist updates before wl:[{len(worklist)}] buwl:[{len(bottomup_worklist)}]")
                    worklist.update(ast.get_children())  # add child in the TopDown only if it has not been replaced
                    bottomup_worklist.add(ast)  # add itself in the BottomUp list !
                    logging.debug(f"[visit_replacement] worklist updates wl:[{len(worklist)}] buwl:[{len(bottomup_worklist)}]")

            if not worklist:
                logging.debug("--------------------------------- Switch Bottom Up -----------------------------------")
                logging.debug(f"AST at that time: {orig_ast.pp_str}")
                mode = YieldT.BottomUp
                worklist = bottomup_worklist[::-1]  # switch the worklist
                bottomup_worklist = []
        yield orig_ast, modified  # Final yield
