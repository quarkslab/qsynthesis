#!/usr/bin/env python3

import argparse
import time
import lief
import json
from pathlib import Path
import sys
import logging
from typing import Dict, Generator, Tuple
from enum import Enum
from collections import Counter

from qtracedb import DatabaseManager

from qsynthesis.utils.qtrace_symexec import QtraceSymExec, Mode
from qsynthesis import InputOutputOracleLevelDB, TopDownSynthesizer, PlaceHolderSynthesizer, TritonAst, TritonGrammar
from qsynthesis.tables.base import _EvalCtx

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


class AnalysisType(Enum):
    TopDown = 0
    Placeholder = 1


class ResType(Enum):
    OK = 0
    KO = 1
    UNK = 2


class ProgramWrapper(object):
    def __init__(self, program_path, trace):
        """
        NOTE 1: In order to work program must be compiled with -no-pie option
        NOTE 2: This only support non-branching callers
        """
        # Init attributes
        self.program = lief.ELF.parse(program_path)
        self.trace = trace
        self.target_functions = self.get_targets_address()
        self.ground_truth = {}
        self.g = TritonGrammar([(chr(i), 64) for i in range(97, 123)], [])
        self.ctx = _EvalCtx(self.g)

    @property
    def is_ground_truth(self):
        return self.ground_truth != {}

    def get_original(self, f_name):
        return self.ground_truth[f_name][0]

    def get_obfuscated(self, f_name):
        return self.ground_truth[f_name][1]

    def get_stats(self, f_name):
        clear, obfu = self.ground_truth[f_name]
        c_node, c_depth = clear.node_count, clear.depth
        obf_node, obf_depth = obfu.node_count, obfu.depth
        return [(c_node, c_depth), (obf_node, obf_depth)]

    def get_function_address(self, f_name: str) -> int:
        return self.program.get_function_address(f_name)

    def get_targets_address(self) -> Dict[int, str]:
        funs = {}
        for f_sym in (x for x in self.program.symbols if x.is_function and x.name.startswith("target_")):
            funs[self.program.get_function_address(f_sym.name)] = f_sym.name
        return funs

    def get_one_call(self, target_fun: str) -> Tuple[int, int]:
        inst = self.trace.get_first_instr_at_addr(self.get_function_address(target_fun))
        prev_i = self.trace.get_instr(inst.id-1)
        returnsite_inst = self.trace.get_first_instr_at_addr(prev_i.addr+len(prev_i.opcode))
        return prev_i.id, returnsite_inst.id

    def get_all_calls(self) -> Generator[Tuple[str, int, int], None, None]:

        prev_inst = None
        cur_start = None
        cur_fun = None
        expected_return_addr = None

        for i in self.trace.get_all_instr():
            if i.addr in self.target_functions and cur_start is None:
                cur_start = prev_inst.id  # Take the id of the call
                expected_return_addr = prev_inst.addr + len(prev_inst.opcode)
                cur_fun = self.target_functions[i.addr]
            if i.addr == expected_return_addr:
                if cur_start is None:
                    print("Reach return address while start is not set ?")
                yield cur_fun, cur_start, i.id
                cur_start = None
                expected_return_addr = None
            prev_inst = i

    def load_ground_truth(self, file, var_size: int = 64):
        data = json.load(open(file))
        for k, (clear, obfu) in data.items():
            try:
                e1 = self.ctx.eval_str(clear)
            except SyntaxError:
                print(k, " clear is wrong")
                e1 = None
            try:
                e2 = self.ctx.eval_str(obfu)
            except SyntaxError:
                print(k, "obfuscated is wrong")
                e2 = None
            self.ground_truth[k] = (TritonAst.make_ast(self.ctx.ctx, e1), TritonAst.make_ast(self.ctx.ctx, e2))


class ResultEntry:
    def __init__(self, f_name, start, stop, orig: TritonAst, obfu: TritonAst, triton_ast: TritonAst, synt_ast: TritonAst, t_dse, t_synt):
        self.fun_name = f_name
        self.start = start
        self.stop = stop
        if orig:
            self.orig_ast = orig
            self.orig_node, self.orig_depth = self.orig_ast.node_count, self.orig_ast.depth
        if obfu:
            self.obfu_ast = obfu
            self.obfu_node, self.obfu_depth = self.obfu_ast.node_count, self.obfu_ast.depth
        self.triton_ast = triton_ast
        self.triton_node, self.triton_depth = self.triton_ast.node_count, self.triton_ast.depth
        self.synthesized_ast = synt_ast
        self.synth_node, self.synth_depth = self.synthesized_ast.node_count, self.synthesized_ast.depth
        self.dse_t = t_dse
        self.synthesis_t = t_synt
        self.sem_orig_obf = ResType.UNK
        self.sem_obf_trit = ResType.UNK
        self.sem_orig_synth = ResType.UNK

    @staticmethod
    def json_encoder(o):
        if isinstance(o, TritonAst):
            return o.pp_str
        elif isinstance(o, ResultEntry):
            return {**o.__dict__, **{'is_simplified': o.is_simplified, 'is_fully_synthesized': o.is_fully_synthesized}}
        elif isinstance(o, ResType):
            return o.name
        else:
            return repr(o)

    @property
    def is_simplified(self):
        # synthesized node below triton ones, or below original (as triton nodes append to be already simplified)
        return self.synth_node < self.triton_node or self.is_fully_synthesized

    @property
    def is_fully_synthesized(self):
        return self.synth_node <= self.orig_node  # synthesized node count is below or equal to original expression


class RunAnalysis:
    def __init__(self, program_p: Path, trace_p: Path, ground_t: Path, lkps: Path, type: AnalysisType, mode: Mode, timeout: int, first_match:int):
        self.type = type
        self.mode = mode
        logging.info(f"[*] Loading trace:{trace_p}")

        if Path(trace_p).exists():
            logging.info("[*] opening sqlite database file")
            self.dbm = DatabaseManager(f'sqlite:///{trace_p}')
            self.trace = self.dbm.get_trace("x86_64")

        logging.info(f"[*] Loading program: {program_p}")
        self.program = ProgramWrapper(program_p, self.trace)
        if ground_t:
            logging.info(f"[*] Loading ground truth: {ground_t}")
            self.program.load_ground_truth(ground_t, var_size=8)

        logging.info(f"[*] Loading lookup table {lkps}")
        self.ltms = [InputOutputOracleLevelDB.load(lkps)]

        if self.type == AnalysisType.TopDown:
            self.synthesizer = TopDownSynthesizer(self.ltms, bool(first_match))
        elif self.type == AnalysisType.Placeholder:
            self.synthesizer = PlaceHolderSynthesizer(self.ltms, bool(first_match))
        else:
            assert False

        self.symexec = None
        self.results = {}  # fun_name: ResultEntry

        self.cur_ast = None
        self.eval_count = []

    def run(self, check_sem: bool):
        self._run(self.program.get_all_calls(), check_sem)

    def run_function(self, f_name: str, check_sem: bool):
        start, end = self.program.get_one_call(f_name)
        self._run([(f_name, start, end)], check_sem)

    def _run(self, it, check_sem: bool):
        logging.info('[*] Starting test')

        t_begin = time.time()

        try:
            for fun_name, start_id, end_id in it:
                self._run_function(fun_name, start_id, end_id, check_sem)
        except KeyboardInterrupt:
            logging.warning("Execution interrupted!")

        if self.program.is_ground_truth:
            self.post_analysis()
            logging.info(f"Overall time:{time.time()-t_begin:.2f}s")

    def _run_function(self, f_name: str, start_id: int, end_id: int, check: bool):
        logging.info(f"================ Processing function {f_name} ====================")

        if self.program.is_ground_truth:
            orig = self.program.get_original(f_name)
            obfu = self.program.get_obfuscated(f_name)
            logging.info(f"[ORIGINAL]: {orig.pp_str}")
        else:
            orig, obfu = None, None

        # Build IOAst
        t0 = time.time()
        self.symexec = QtraceSymExec(self.trace, self.mode)
        self.symexec.process_instr_sequence(start_id, end_id)
        t_dse = time.time() - t0

        ast, new_ast, t_syn = self.synthesize_new_triton(self.symexec)

        result = ResultEntry(f_name, start_id, end_id, orig, obfu, ast, new_ast, t_dse, t_syn)
        self.results[f_name] = result

        self.show_stats(result)

        if check:
            self.check_semantic(result)

    def synthesize_new_triton(self, vs: QtraceSymExec):
        ast = vs.get_register_ast('rax')
        self.cur_ast = ast

        t0 = time.time()
        if ast.is_constant_expr():
            logging.info("[INFO] constant expression")
            cst_val = ast.eval_oracle({})
            final_ast = ast.mk_constant(cst_val, ast.size)  # evaluate to a constant in case it was an expression (yet cst)
        else:
            logging.debug(f"[DEBUG]: Variables number: {ast.var_num}")
            final_ast, simplified = self.synthesizer.synthesize(ast)
            if simplified:
                logging.debug(f"[INFO]: simplified in parts !")
            logging.debug(f"[CALL_EVAL]: {self.synthesizer.call_to_eval}")
            logging.debug(f"[EVAL_COUNT]: {self.synthesizer.eval_count}")
            self.eval_count.append(self.synthesizer.eval_count)
            logging.debug(f"[CACHE_HIT]: {self.synthesizer.cache_hit}")

            if final_ast is None:
                logging.info("[RESULT]: None (use Triton)")
                final_ast = ast  # Keep it as returned by Triton (to compute stats)

        self.denormalize_triton_ast(final_ast)

        t1 = time.time() - t0

        return ast, final_ast, t1

    def denormalize_triton_ast(self, ast: TritonAst) -> None:
        MAPPING = {'rdi': 'a', 'rsi': 'b', 'rdx': 'c', 'rcx': 'd', 'r8': 'e', 'r9': 'd'}
        for symv in ast.symvars:
            alias = symv.getAlias()
            if alias in MAPPING:
                symv.setAlias(MAPPING[alias])

    def show_stats(self, res: ResultEntry):
        logging.info(f"[RESULT]: {res.synthesized_ast.pp_str if res.synth_node < 50 else 'too large'}")
        logging.info(f"[TIME]: {res.synthesis_t:.3f}s")
        s_fmt = ""

        if self.program.is_ground_truth:
            s_fmt += f"Orig:[sz:{res.orig_node}, d:{res.orig_depth}] Obfu:[sz:{res.obfu_node}, d:{res.obfu_depth}] "
        s_fmt += f"Triton:[sz:{res.triton_node}, d:{res.triton_depth}] Synthesized:[sz:{res.synth_node}, d:{res.synth_depth}] "
        if self.program.is_ground_truth:
            s_fmt += f"[Simp:{'OK' if res.is_simplified else 'KO'}]  "  # Expr as been simplified
            s_fmt += f"[FullSynth:{'OK' if res.is_fully_synthesized else 'KO'}]"  # Expr is as small as original
        logging.info(s_fmt)

    def check_semantic(self, res: ResultEntry):
        pass
        # # Convert triton AST and synthesized expression to 8 bit expression
        # #triton_8bit_ast = res.triton_ast.expr_with_bitsize(8)
        # final_8bit_ast = res.synthesized_ast.expr_with_bitsize(8)
        #
        # # Check the semantic equivalence
        # orig_obf = ResType.UNK  # self.z3_status_restype(self.solver.check(res.orig_ast.expr != res.obfu_ast.expr))
        # obf_trit = ResType.UNK  # self.z3_status_restype(self.solver.check(res.obfu_ast.expr != triton_8bit_ast))
        # orig_final = self.z3_status_restype(self.solver.check(res.orig_ast.expr != final_8bit_ast))
        #
        # # Update the ResultEntry infos
        # #res.sem_orig_obf = orig_obf
        # #res.sem_obf_trit = obf_trit
        # res.sem_orig_synth = orig_final
        #
        # logging.info(f"[EQUIVALENCE]: [org-obf:{orig_obf.name}] [obf-trit:{obf_trit.name}] [org-final:{orig_final.name}]")

    def post_analysis(self):
        logging.info("================== FINAL RESULTS ==================")
        count = len(self.results)

        # Count of simplification number
        n_simpl, n_synth = 0, 0
        for res in self.results.values():
            if res.is_fully_synthesized:
                n_simpl += 1
                n_synth += 1
            elif res.is_simplified:
                n_simpl += 1
        #n_simpl = sum(x.is_simplified for x in self.results.values())
        #n_synth = sum(x.is_fully_synthesized for x in self.results.values())

        logging.info(f"Simplification: {n_simpl}/{count} ({n_simpl/count:.2%})")
        logging.info(f"Full synthesis: {n_synth}/{count} ({n_synth/count:.2%})")
        n_trace_len = sum(x.stop-x.start for x in self.results.values())
        logging.info(f"Mean trace length: {n_trace_len/count:.2f}")

        # Efficiency of the synthesis simplification
        sum_n_o = sum(x.orig_node for x in self.results.values())
        sum_n_b = sum(x.obfu_node for x in self.results.values())
        sum_n_t = sum(x.triton_node for x in self.results.values())
        sum_n_s = sum(x.synth_node for x in self.results.values())
        logging.info(f"Mean sizes: [orig: {sum_n_o/count:.2f}] [obfu:{sum_n_b/count:.2f}] [trit:{sum_n_t/count:.2f}] [synth:{sum_n_s/count:.2f}]")
        logging.info(f"Scale size: [obf/orig:{sum_n_b/sum_n_o:.2f}] [synth/trit:{sum_n_s/sum_n_t:.2f}] [synth/orig:{sum_n_s/sum_n_o:.2f}]")

        # Semantic efficiency of the synthesis
        counter = Counter(x.sem_orig_synth for x in self.results.values())
        logging.info(f"Semantic equivalence: {counter[ResType.OK]}/{count} (KO:{counter[ResType.KO]}, UNK:{counter[ResType.UNK]})")

        # Time results
        t_dse = sum(x.dse_t for x in self.results.values())
        t_synth = sum(x.synthesis_t for x in self.results.values())
        tot_t = t_dse + t_synth
        m_d, s_d = int(t_dse/60), int(t_dse % 60)
        m_s, s_s = int(t_synth/60), int(t_synth % 60)
        logging.info(f"Time DSE:{m_d}m{s_d}s Synthesis:{m_s}m{s_s}s Total:{tot_t:.2f}s (Mean:{tot_t/count:.2f}s/expr)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run synthesis on a dataset.')
    parser.add_argument('program', type=str, help='obfuscated program to synthesize')
    parser.add_argument('sqlite_db', type=str, help='an sqlite db containing the program\'s trace')
    parser.add_argument('--lookup-tables', type=str, help='instruct to use the lookup-table based exhaustive '
                                                          'search backend using the given lookup-table')
    parser.add_argument('--type', type=str, default="TopDown", help='analysis kind to launch (legacy, new)')
    parser.add_argument('-g', '--ground-truth', type=str, default="", help="Ground truth with original functions")
    parser.add_argument('--check', action="store_true", help="Check by Z3 equivalence of expression")
    parser.add_argument('-v', default=0, action="count", help='Enable verbose mode')
    parser.add_argument('-n', '--name', default="", help="Function target to synthesize")
    parser.add_argument('-m', '--mode', default="", help="Symbolic execution mode")
    parser.add_argument('-o', '--output', type=str, default=None, help="Output file to log result as JSON")
    parser.add_argument('-t', '--timeout', default=5, type=int, help="SMT timeout for semantic equivalence check")
    parser.add_argument('--first-match', default=0, action="count", help="Only keep first match in table lookup")

    args = parser.parse_args()

    for f in [Path(x) for x in [args.program, args.sqlite_db, args.lookup_tables, args.ground_truth]]:
        pass
        #if not f.exists():
        #    print(f"File: {f} not found")
        #    sys.exit(1)

    if args.v:
        print("Enabling verbose")
        logging.root.handlers = []
        logging.basicConfig(level=logging.DEBUG, format='%(message)s')

    ana_types = [x.name for x in AnalysisType]
    if args.type not in ana_types:
        print(f"Invalid type: {args.type}, valid ones are: {ana_types}")
        sys.exit(1)

    mode = Mode.PARAM_SYMBOLIC
    if args.mode:
        if args.mode in list(x.name for x in Mode):
            mode = Mode[args.mode]
        else:
            print(f"Invalid mode {args.mode} provided")
            sys.exit(1)

    typ = AnalysisType[args.type]
    to = args.timeout
    fm = args.first_match

    analyzer = RunAnalysis(args.program, args.sqlite_db, args.ground_truth, args.lookup_tables, typ, mode, to, fm)

    if args.name:
        analyzer.run_function(args.name, args.check)
    else:
        analyzer.run(args.check)

    if args.output:
        json.dump(analyzer.results, open(args.output, 'w'), default=ResultEntry.json_encoder, indent=2)

    analyzer.trace.close()
