import logging
from typing import List, Optional, Tuple, Set, Dict, Generator, Union
import triton
from enum import IntEnum

from qsynthesis.tables.lookuptable import LookupTable
from qsynthesis.tritonast import TritonAst


class SynthesizerBase:
    """
    Synthesize with Top-Down ONLY AST search based on Triton AST.
    This synthesis mechanism always converges
    """

    def __init__(self, ltms: Union[LookupTable, List[LookupTable]], only_first_match: bool = False):
        """
        Initialize TritonTDBUSynthesizer

        :param ltms: List of lookup tables to use
        """
        self._ltms = [ltms] if isinstance(ltms, LookupTable) else ltms
        self.only_first_match = only_first_match

        # Caches use internally
        self.expr_cache = {}  # Dict expr_str -> synth_expr
        self.eval_cache = {}  # Dict associating a the hash of an AST to a map of inputs -> output

        # Stats
        self.call_to_eval = 0
        self.cache_hit = 0
        self.eval_count = 0

    def synthesize(self, ioast: TritonAst, check_sem: bool = False) -> Tuple[TritonAst, bool]:
        raise NotImplementedError("Should be implemented in children class")

    def try_synthesis_lookup(self, cur_ast: TritonAst, check_sem=False) -> Optional[TritonAst]:
        # Try all lookup-tables and pick the shortest expression
        best_len = 0
        best_expr = None
        vars_e = cur_ast.symvars
        size_e = cur_ast.node_count

        #expr_str = str(cur_ast.expr)
        if cur_ast.hash in self.expr_cache:
            self.cache_hit += 1
            logging.debug("expression cache found !")
            return self.expr_cache[cur_ast.hash] #.duplicate()

        for table in self._ltms:
            # skip tables not having enough variables or the wrong bitsize
            if table.var_number < len(vars_e) or table.bitsize != cur_ast.size:
                continue

            synth_ast = self.run_direct_synthesis(table, cur_ast)
            if synth_ast is None:
                continue

            # Check if we got something better
            e_node_cnt = synth_ast.node_count
            if (best_expr is None and e_node_cnt < size_e) or (e_node_cnt < best_len):
                logging.debug(f"[base] candidate expr accepted: current:{size_e} candidate:{e_node_cnt} (best:{best_len})  => {synth_ast.pp_str}")
                best_len, best_expr = e_node_cnt, synth_ast
                if self.only_first_match:
                    break
            else:
                pass
                #logging.debug(f"candidate expr rejected because too big: current:{size_e} candidate:{e_node_cnt} (best:{best_len})")

            if best_len == 1:  # It cannot go better that this, so we can stop
                break

        if best_expr is not None:
            self.expr_cache[cur_ast.hash] = best_expr
            if check_sem:
                if cur_ast.is_semantically_equal(best_expr):
                    logging.info("Expressions are semantically equal")
                else:
                    logging.error("Expressions are semantically different (return None)!!")
                    return None
            return best_expr

    def run_direct_synthesis(self, ltm: LookupTable, cur_ast: TritonAst) -> Optional['TritonIOAst']:
        # Evaluate node on LTMs inputs
        #outputs = [cur_ast.eval_oracle(i) for i in ltm.inputs]
        outputs = [self.eval_ast(cur_ast, i) for i in ltm.inputs]
        #logging.debug(f"{ltm.name.name}: outputs => {outputs}")

        if len(set(outputs)) == 1:  # If all outputs are equal then we consider this as a constant expression
            logging.debug(f"[base] Found constant expression in {ltm.name}: {cur_ast.pp_str} ===> {outputs[0]}")
            return cur_ast.mk_constant(outputs[0], cur_ast.size)
        else:
            # Lookup expression with same outputs in the LT
            lk_expr = ltm.lookup(outputs, cur_ast, use_cache=False)
            if lk_expr is None:
                return None
            else:
                if lk_expr.node_count > cur_ast.node_count:
                    logging.warning(f"[base] synthesized bigger expression ({lk_expr.pp_str}) than given ({cur_ast.pp_str})")
                #logging.debug(f"found candidate entry {ltm.name}")#}: {cur_ast.expr} ===> {lk_expr}")
                return lk_expr

    def eval_ast(self, ioast, inputs):
        self.call_to_eval += 1
        tup_inputs = tuple(inputs.items())

        ast_hash = ioast.hash
        if ast_hash in self.eval_cache:
            if tup_inputs in self.eval_cache[ast_hash]:
                return self.eval_cache[ast_hash][tup_inputs]
        else:
            self.eval_cache[ast_hash] = {}

        output = ioast.eval_oracle(inputs)
        self.eval_count += 1
        self.eval_cache[ast_hash][tup_inputs] = output
        return output
