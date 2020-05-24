import logging
from typing import List, Optional, Tuple, Set, Dict, Generator, Union
import triton
from enum import IntEnum

from qsynthesis.tritonast import TritonAst
from qsynthesis.algorithms.synthesizer_tdbu import TritonTDBUSynthesizer, YieldT


AstType = IntEnum("AstNode", {k: v for k, v in triton.AST_NODE.__dict__.items() if isinstance(v, int)})


class TritonPlaceHolderSynthesizer(TritonTDBUSynthesizer):
    """
    Synthesize with Top-Down then Bottom-Up AST search based on
    Triton AST.
    This synthesis mechanism always converges
    """

    def synthesize(self, ioast: TritonAst) -> Tuple[TritonAst, bool]:
        self.expr_cache = {}
        self.eval_cache = {}
        self.call_to_eval = 0
        self.eval_count = 0

        expr_repl_visitor = self._visit_replacement_bfs(ioast, iff_replace=False, update=True)
        new_expr_to_send = None  # hold Ast replacement
        expr_modified = False  # True as soon as one sub-ast has been synthesized
        replacements = {}  # AST -> Placeholder AST
        replacements_hashs = {}  # AST hash -> Placeholder
        cur_placeholder = 0

        while 1:
            logging.debug(f"[plhdr] sending: {new_expr_to_send.pp_str if new_expr_to_send is not None else None}")
            cur_ast, info = expr_repl_visitor.send(new_expr_to_send)  # Iterate generator
            #logging.debug(f"Receiving: {cur_ast}")

            if isinstance(info, bool):
                final_ast = cur_ast
                break # Don't do anything of final expr return

            # Check if not already substituted, if so yield directly the placeholder
            if cur_ast.hash in replacements_hashs:
                logging.debug(f"[plhdr] Hash match: {cur_ast.pp_str} ==> {replacements_hashs[cur_ast.hash].pp_str}")
                new_expr_to_send = replacements_hashs[cur_ast.hash] #.duplicate()
                continue

            # If don't have children either constant or variable just continue
            if not cur_ast.has_children(): # FIXME: If constant maybe symbolizing it
                continue

            # Try synthesizing expression
            logging.debug(f"[phldr] try synthesis lookup: {cur_ast.pp_str if cur_ast.node_count < 50 else 'too large'} [{cur_ast.node_count}] [{info.name}]")
            synt_res = self.try_synthesis_lookup(cur_ast)

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
            logging.debug(f"[plhdr] Create Placeholder plhd_{cur_placeholder-1} for: {synt_res.pp_str} ptr_id: {synt_res.ptr_id}")

        logging.debug(f"[plhdr] AST before replacement: {final_ast.pp_str}")
        for k, v in replacements.items():
            logging.debug(f"Final replace {k.pp_str}  ==> {v.pp_str}")
        self.replace_all(final_ast, replacements, recursive=True)
        final_ast.update_all()
        logging.info(f"Final AST: {final_ast.pp_str}")
        return final_ast, expr_modified

    def _ast_binop_with_cst(self, ast):
        return sum(x.is_constant() for x in ast.get_children()) != 0

    def replace_all(self, ast, replacement: Dict['TritonAst', 'TritonAst'], recursive=False) -> None:
        """
        WARNING: All the expressions to replace should be present only once because TritonIOAST objects
        are not deepcopied. Thus duplicating it at various location of the AST would be various dangerous
        for attributes coherence when calling update
        :param replacement:
        :param recursive:
        :return:
        """
        try:
            new_expr_to_send = None
            hash_mapping = {k.hash: k for k in replacement}

            logging.debug(f"Len replacement: {len(replacement)}  len hash_mapping:{len(hash_mapping)}")

            g = ast.visit_replacement(update=False)

            while True:
                cur_ast = g.send(new_expr_to_send)
                new_expr_to_send = None
                cur_ast_id = cur_ast.hash

                if cur_ast_id in hash_mapping:  # Prefilter with hash to check if current ast might be substituted
                    key = hash_mapping[cur_ast_id]
                    logging.debug(f"cur_ast: {cur_ast.pp_str}  key:{key.pp_str}  equal:{cur_ast.expr.equalTo(key.expr)}")

                    if not cur_ast.expr.equalTo(key.expr):  # Check that they are equal
                        logging.warning("two different expressions !")

                    new_expr_to_send = replacement[key]
                    logging.debug(f"Will replace {cur_ast.pp_str}  ==> {new_expr_to_send.pp_str}")

                    if recursive:
                        self.replace_all(new_expr_to_send, replacement, recursive=recursive)
        except StopIteration:
            pass
