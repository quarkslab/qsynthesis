# built-in modules
from typing import List, Tuple, Generator, Union
from enum import IntEnum

# third-party modules
from ordered_set import OrderedSet

# qsynthesis modules
from qsynthesis.tables.base import InputOutputOracle
from qsynthesis.tritonast import TritonAst
from qsynthesis.algorithms.synthesizer_base import SynthesizerBase, logger


class YieldT(IntEnum):
    """
    Enum to identify the yield type in the huge
    generator that performs the AST search and
    substitution.
    """
    TopBottom = 1
    BottomUp = 2


class TopDownBottomUpSynthesizer(SynthesizerBase):
    """
    Synthesize with Top-Down then Bottom-Up AST search based on Triton AST.
    The idea behind this bi-directional search is that a node simplified in
    a sub-AST might enable synthesizing one of its parent by means of simplification
    if for instance we turn a two-variable AST in a one-variable AST.

    .. warning:: This class is not meant to be instanciated directly but rather
                 serving a common base for children class using this search strategy.
    """

    def __init__(self, ltms: Union[InputOutputOracle, List[InputOutputOracle]], only_first_match: bool = False, learning_new_exprs: bool = False):
        """
        Constructor that takes lookup tables as input.

        :param ltms: Single lookup table of a list of them
        :param only_first_match: boolean that stop interating over tables as soon as the lookup is successfull for one
        :param learning_new_exprs: boolean that enables improving the current table if if a synthesized entry appears
                                   to be bigger than the one submitted
        """
        super(TopDownBottomUpSynthesizer, self).__init__(ltms, only_first_match, learning_new_exprs)
        self.total_repl_td = 0
        self.total_repl_bu = 0

    def synthesize(self, ioast: TritonAst, check_sem: bool = False) -> Tuple[TritonAst, bool]:
        """
        Performs the Top-Down and then Bottom-Search to synthesize the AST

        :param ioast: TritonAst object to synthesize
        :param check_sem: boolean one whether to check the semantic equivalence of expression
                          before substituting them. That ensure soundness of the synthesis
        :returns: tuple with new TritonAst and whether some replacement took place or not

        .. warning:: Activating the `check_sem` parameter implies a strong overhead
                     on the synthesis as SMT queries are being performed for any candidates
        """
        ioast = ioast.duplicate()  # Make a copy of the ast not tamper it
        self.expr_cache = {}
        self.eval_cache = {}
        self.call_to_eval = 0
        self.eval_count = 0

        expr_repl_visitor = self._visit_replacement(ioast)
        new_expr_to_send = None
        expr_modified = False

        while 1:
            # logger.debug(f"Sending: {new_expr_to_send}")
            cur_expr, info = expr_repl_visitor.send(new_expr_to_send)  # Iterate generator
            # logger.debug(f"Receiving: {cur_expr}")
            if isinstance(info, bool):
                final_expr = cur_expr
                # logger.debug("final expression:", final_expr.replace("\n"," "))
                final_expr.update_all()
                return final_expr, expr_modified

            else:  # Try synthesizing expression
                logger.debug(f"try synthesis lookup: {cur_expr.pp_str if cur_expr.node_count < 50 else 'too large'}")
                synt_res = self.try_synthesis_lookup(cur_expr, check_sem)
                if synt_res is not None:
                    expr_modified = True
                new_expr_to_send = synt_res  # Send the result to the generator (thus either new expr or None)

    def _visit_replacement(self, ast: TritonAst, iff_replace=True, update=True) -> Generator[Tuple[TritonAst, Union[YieldT, bool]], TritonAst, None]:
        """
        Generator that will iterate over the AST. It will yield each sub-AST and is expecting
        through the send mechanism to receive None (meaning nothing has been found) or a TritonAst
        which means the AST just yielded have to be substituted by this one.

        .. note:: Deprecated to use. Use _visit_replacement_bfs instead
        """
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

    def _visit_replacement_bfs(self, orig_ast: TritonAst) -> Generator[Tuple[TritonAst, Union[YieldT, bool]], TritonAst, None]:
        """
        Generator that will iterate over the AST. It will yield each sub-AST and is expecting
        through the send mechanism to receive None (meaning nothing has been found) or a TritonAst
        which means the AST just yielded have to be substituted by this one. The search is made in
        a Breath-First-Search manner as it provides better results for synthesis.

        :param orig_ast: TritonAst to iterate (and to modify)
        :returns: generator that always yield a TritonAst and either a yield type indicating whether it is during
                  the top-down or bottom-up search or a boolean indicating all substitutions have been performed
                  in the AST. The generate receive via send a new TritonAst or None
        """
        worklist = OrderedSet([orig_ast])
        bottomup_worklist = OrderedSet([orig_ast])
        modified = False
        mode = YieldT.TopBottom

        while worklist:
            ast = worklist.pop(0)
            logger.debug(f"\n[visit_replacement] pop [{ast.ptr_id}]({len(ast.get_children())}) {ast.pp_str}  worklist:[{len(worklist)}] bu_worklist:[{len(bottomup_worklist)}]")

            if not ast.is_leaf():
                rep = yield ast, mode  # Current yield (TopB, or BotUp)
            else:
                rep = None

            if rep is not None:  # We should replace this expr
                modified = True
                self.total_repl_td += 1 if mode == YieldT.TopBottom else 0
                self.total_repl_bu += 1 if mode == YieldT.BottomUp else 0
                if ast.is_root():
                    logger.debug(f"YIeld new root ahead of time: {rep.pp_str}")
                    yield rep, modified  # Final yield (ahead of time)
                else:
                    logger.debug(f"replace_self: {str(ast.expr)} => {rep.pp_str}")
                    ast.replace_self(rep, update_parents=True)
            else:
                if mode == YieldT.TopBottom:
                    logger.debug(f"[visit_replacement] worklist updates before wl:[{len(worklist)}] buwl:[{len(bottomup_worklist)}]")
                    worklist.update(ast.get_children())  # add child in the TopDown only if it has not been replaced
                    bottomup_worklist.add(ast)  # add itself in the BottomUp list !
                    logger.debug(f"[visit_replacement] worklist updates wl:[{len(worklist)}] buwl:[{len(bottomup_worklist)}]")

            if not worklist:
                logger.debug("--------------------------------- Switch Bottom Up -----------------------------------")
                logger.debug(f"AST at that time: {orig_ast.pp_str}")
                mode = YieldT.BottomUp
                worklist = bottomup_worklist[::-1]  # switch the worklist
                bottomup_worklist = []
        yield orig_ast, modified  # Final yield
