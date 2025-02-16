# built-in modules
from __future__ import annotations
import logging

# qsynthesis deps
from qsynthesis.tables.base import InputOutputOracle
from qsynthesis.tritonast import TritonAst
from qsynthesis.types import Input, List, Optional, Tuple, Union, Output

logger = logging.getLogger("qsynthesis")


class SynthesizerBase:
    """
    Base Synthesizer that provides base function for children classes.
    It provides function for a given TritonAst to evaluate it against
    oracle inputs and to perform the lookup in order to know
    if a shorter expression exists.
    """

    def __init__(self, ltms: Union[InputOutputOracle, List[InputOutputOracle]], only_first_match: bool = False, learning_new_exprs: bool = False):
        """
        Constructor that takes one or multiple oracles as input.

        :param ltms: Single oracle or a list
        :param only_first_match: boolean that stop interating over tables as soon as the lookup is successfull for one
        :param learning_new_exprs: boolean that enables improving the current table if if a synthesized entry appears
                                   to be bigger than the one submitted
        """
        self._ltms = [ltms] if isinstance(ltms, InputOutputOracle) else ltms
        self.only_first_match = only_first_match
        self.learning_enabled = learning_new_exprs

        # Caches use internally
        self.expr_cache = {}  # Dict expr_str -> synth_expr
        self.eval_cache = {}  # Dict associating a the hash of an AST to a map of inputs -> output

        # Stats
        self.call_to_eval = 0
        self.cache_hit = 0
        self.eval_count = 0

    def synthesize(self, ioast: TritonAst, check_sem: bool = False) -> Tuple[TritonAst, bool]:
        """
        Abstract function that synthesize the given TritonAst, into a smaller if it exists.
        The implementation of this function is delegated to children classes.

        :param ioast: TritonAst object to synthesize
        :param check_sem: boolean one whether to check the semantic equivalence of expression
                          before substituting them. That ensure soundness of the synthesis
        :returns: tuple with new TritonAst and whether some replacement took place or not

        .. warning:: Activating the `check_sem` parameter implies a strong overhead
                     on the synthesis as SMT queries are being performed for any candidates
        """
        raise NotImplementedError("Should be implemented in children class")

    def try_synthesis_lookup(self, cur_ast: TritonAst, check_sem: bool = False) -> Optional[TritonAst]:
        """
        Performs a direct synthesis lookup. And returns an optional TritonAst if it the
        I/O evaluation has been found in one of the tables. Unlike :meth:`SynthesizerBase.synthesize`
        which can go down the AST to try simplifying sub-AST here only the root node is
        attempted to be synthesized.

        :param cur_ast: TritonAst to synthesize
        :param check_sem: boolean on whether to check the semantic equivalence of expression
                          before performing substituting.
        :returns: optional TritonAst if the the AST has been synthesized
        """

        # Try all lookup-tables and pick the shortest expression
        best_len = 0
        best_expr = None
        vars_e = cur_ast.symvars
        size_e = cur_ast.node_count

        if cur_ast.hash in self.expr_cache:
            self.cache_hit += 1
            logger.debug("expression cache found !")
            return self.expr_cache[cur_ast.hash]

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
                logger.debug(f"[base] candidate expr accepted: current:{size_e} candidate:{e_node_cnt} (best:{best_len})  => {synth_ast.pp_str}")
                best_len, best_expr = e_node_cnt, synth_ast
                if self.only_first_match:
                    break
            else:
                pass

            if best_len == 1:  # It cannot go better that this, so we can stop
                break

        if best_expr is not None:
            self.expr_cache[cur_ast.hash] = best_expr
            if check_sem:
                if cur_ast.is_semantically_equal(best_expr):
                    logger.info("Expressions are semantically equal")
                else:
                    logger.error("Expressions are semantically different (return None)!!")
                    return None
            return best_expr

    def run_direct_synthesis(self, ltm: InputOutputOracle, cur_ast: TritonAst) -> Optional['TritonAst']:
        """
        Evaluate `cur_ast` on inputs provided by `ltm` the oracle which provide an
        output vector then used to perform the lookup in the database. If an entry is found
        it is returned.

        :param ltm: InputOutputOracle object in which to perform the query
        :param cur_ast: TritonAst object to synthesize
        :returns: optional TritonAst if the an entry was found
        """

        # Evaluate node on LTMs inputs
        outputs = [self.eval_ast(cur_ast, i) for i in ltm.inputs]

        if len(set(outputs)) == 1:  # If all outputs are equal then we consider this as a constant expression
            logger.debug(f"[base] Found constant expression in {ltm.name}: {cur_ast.pp_str} ===> {outputs[0]}")
            return cur_ast.mk_constant(outputs[0], cur_ast.size)
        else:
            # Lookup expression with same outputs in the LT
            lk_expr = ltm.lookup(outputs, cur_ast, use_cache=False)
            if lk_expr is None:
                return None
            else:
                if lk_expr.node_count > cur_ast.node_count:
                    logger.debug(f"[base] synthesized bigger expression ({lk_expr.pp_str}) than given ({cur_ast.pp_str})")
                    if ltm.is_writable and self.learning_enabled:
                        h = ltm.hash(outputs)
                        s = cur_ast.to_normalized_str()
                        logger.info(f"[base] expression {s} added to {ltm.name}")
                        ltm.add_entry(h, s)
                return lk_expr

    def eval_ast(self, ioast: TritonAst, input: Input) -> Output:
        """
        Run evaluation of the TritonAst `ioast` on the given Input (valuation for all vars).
        The result is an Output (integer)

        :param ioast: TritonAst to evaluate
        :type ioast: TritonAst
        :param input: Input on which to evaluate the AST. All variables of ioast must
                      be defined in input
        :type input: :py:obj:`qsynthesis.types.Input`
        :returns: Output which is the result of evaluation (made by Triton)
        :rtype: :py:obj:`qsynthesis.types.Output`
        """
        self.call_to_eval += 1
        tup_inputs = tuple(input.items())

        ast_hash = ioast.hash
        if ast_hash in self.eval_cache:
            if tup_inputs in self.eval_cache[ast_hash]:
                return self.eval_cache[ast_hash][tup_inputs]
        else:
            self.eval_cache[ast_hash] = {}

        output = ioast.eval_oracle(input)
        self.eval_count += 1
        self.eval_cache[ast_hash][tup_inputs] = output
        return output
