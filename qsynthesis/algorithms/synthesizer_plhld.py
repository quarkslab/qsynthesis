# built-in modules
from typing import Tuple, Dict

# qsynthesis modules
from qsynthesis.tritonast import TritonAst
from qsynthesis.algorithms.synthesizer_tdbu import TopDownBottomUpSynthesizer, YieldT, logger


class PlaceHolderSynthesizer(TopDownBottomUpSynthesizer):
    """
    Synthesizer inherited from TopDownBottomUpSynthesizer which thus
    form the search in that manner. The specificity of this synthesizer
    is to temporarily replacing synthesized expressions with a placeholder
    variable that will 'abstract the behavior of the synthesized AST'. The
    intended effect is to recursively synthesized previously synthesized
    sub-AST (with these placeholders). At the end of the search all placeholder
    variables are substituted by their associated synthesized expression.
    """

    def synthesize(self, ioast: TritonAst, check_sem: bool = False) -> Tuple[TritonAst, bool]:
        """
        Performs the placeholder based top-down and then bottom-up search for synthesizing
        the given TritonAst.

        :param ioast: TritonAst object to synthesize
        :param check_sem: boolean one whether to check the semantic equivalence of expression
                          before substituting them. That ensure soundness of the synthesis
        :return: tuple with new TritonAst and whether some replacement took place or not

        .. warning:: Activating the `check_sem` parameter implies a strong overhead
                     on the synthesis as SMT queries are being performed for any candidates
        """
        ioast = ioast.duplicate()  # Make a copy of the AST to to modify it
        self.expr_cache = {}
        self.eval_cache = {}
        self.call_to_eval = 0
        self.eval_count = 0

        expr_repl_visitor = self._visit_replacement_bfs(ioast)
        new_expr_to_send = None  # hold Ast replacement
        expr_modified = False    # True as soon as one sub-ast has been synthesized
        replacements = {}        # AST -> Placeholder AST
        replacements_hashs = {}  # AST hash -> Placeholder
        cur_placeholder = 0

        while 1:
            logger.debug(f"[plhdr] sending: {new_expr_to_send.pp_str if new_expr_to_send is not None else None}")
            cur_ast, info = expr_repl_visitor.send(new_expr_to_send)  # Iterate generator

            if isinstance(info, bool):
                final_ast = cur_ast
                break # Don't do anything of final expr return

            # Check if not already substituted, if so yield directly the placeholder
            if cur_ast.hash in replacements_hashs:
                logger.debug(f"[plhdr] Hash match: {cur_ast.pp_str} ==> {replacements_hashs[cur_ast.hash].pp_str}")
                new_expr_to_send = replacements_hashs[cur_ast.hash]
                continue

            # If don't have children either constant or variable just continue
            if not cur_ast.has_children(): # FIXME: If constant maybe symbolizing it
                continue

            # Try synthesizing expression
            logger.debug(f"[phldr] try synthesis lookup: {cur_ast.pp_str if cur_ast.node_count < 50 else 'too large'} [{cur_ast.node_count}] [{info.name}]")
            synt_res = self.try_synthesis_lookup(cur_ast, check_sem)

            if synt_res is not None:
                expr_modified = True
            else:
                new_expr_to_send = None
                if info == YieldT.TopBottom:
                    continue
                else:  # BottomUp
                    if self._ast_binop_with_cst(cur_ast):  # If binary operation with a constant make a placeholder
                        synt_res = cur_ast
                    else:
                        continue

            # Here necessarily synthesized of BottomUp

            # Create the a placeholder whether or not we synthesized the expression !
            placeholder = ioast.mk_variable(f"plhd_{cur_placeholder}", synt_res.size)
            cur_placeholder += 1
            replacements[placeholder] = synt_res
            replacements_hashs[cur_ast.hash] = placeholder
            new_expr_to_send = placeholder
            logger.debug(f"[plhdr] Create Placeholder plhd_{cur_placeholder-1} for: {synt_res.pp_str} ptr_id: {synt_res.ptr_id}")

        logger.debug(f"[plhdr] AST before replacement: {final_ast.pp_str}")
        for k, v in replacements.items():
            logger.debug(f"Final replace {k.pp_str}  ==> {v.pp_str}")
        self.replace_all(final_ast, replacements, recursive=True)
        final_ast.update_all()
        logger.info(f"Final AST: {final_ast.pp_str}")
        return final_ast, expr_modified

    @staticmethod
    def _ast_binop_with_cst(ast: TritonAst) -> bool:
        """
        Find if either the left or right branch of an AST is a constant.
        That enable replacing them with placeholder variable to help the
        synthesis by temporarily getting rid of constants.

        :param ast: TritonAst
        :return: True if left or right child of the AST is a constant
        """
        return sum(x.is_constant() for x in ast.get_children()) != 0

    def replace_all(self, ast: TritonAst, replacement: Dict[TritonAst, TritonAst], recursive: bool = False) -> None:
        """
        Performs the final replacement of Placeholder variable with their synthesized expression equivalence
        in `ast` provided in parameter.

        :param ast: TritonAst object in which to perform all substitutions
        :param replacement: dictionnary of placeholder variable as TritonAst to synthesized expression (TritonAst)
        :param recursive: whether to also perform the substitution in synthesized expressions given in the dictionnary
        :return: None as the `ast` object is modified in place

        .. warning:: All the expressions to replace should be present only once because TritonAst objects
                     are not deepcopied. Thus duplicating it at various location of the AST would be various dangerous
                     for attributes coherence when calling update
        """
        try:
            new_expr_to_send = None
            hash_mapping = {k.hash: k for k in replacement}

            logger.debug(f"Len replacement: {len(replacement)}  len hash_mapping:{len(hash_mapping)}")

            g = ast.visit_replacement(update=False)

            while True:
                cur_ast = g.send(new_expr_to_send)
                new_expr_to_send = None
                cur_ast_id = cur_ast.hash

                if cur_ast_id in hash_mapping:  # Prefilter with hash to check if current ast might be substituted
                    key = hash_mapping[cur_ast_id]
                    logger.debug(f"cur_ast: {cur_ast.pp_str}  key:{key.pp_str}  equal:{cur_ast.expr.equalTo(key.expr)}")

                    if not cur_ast.expr.equalTo(key.expr):  # Check that they are equal
                        logger.warning("two different expressions !")

                    new_expr_to_send = replacement[key]
                    logger.debug(f"Will replace {cur_ast.pp_str}  ==> {new_expr_to_send.pp_str}")

                    if recursive:
                        self.replace_all(new_expr_to_send, replacement, recursive=recursive)
        except StopIteration:
            pass
