import pickle
from pathlib import Path
from qsynthesis.grammar import TritonGrammar
import logging

from typing import Optional, List, Dict, Union, Generator, Tuple, Any, TypeVar
from time import time

Expr = TypeVar('Expr')  # Expression type in the associated grammar


class LookupTable:
    def __init__(self, gr: TritonGrammar, inputs: Union[int, List[Dict[Any, Any]]], f_name: str = ""):
        self._name = Path(f_name)
        self.lookup_table = {}
        self.grammar = gr
        self._bitsize = self.grammar.size
        self.expr_cache = {}
        self.lookup_count = 0
        self.lookup_found = 0
        self.cache_hit = 0

        if isinstance(inputs, int):
            self.inputs = self.grammar.gen_test_inputs(inputs)
        else:
            self.inputs = inputs

    @property
    def size(self):
        return len(self.lookup_table)

    def lookup(self, outputs: Union[Tuple[int], List[int]], *args,  use_cache=True) -> Optional[Expr]:
        self.lookup_count += 1
        outputs = outputs if isinstance(outputs, tuple) else tuple(outputs)
        #h = hash(outputs)
        h = outputs
        if h in self.expr_cache and use_cache:
            self.cache_hit += 1
            return self.expr_cache[h]
        else:
            v = self.lookup_table.get(h, None)
            if v:
                self.lookup_found += 1
                e = self.grammar.str_to_expr(v, *args)
                self.expr_cache[h] = e
                return e
            else:
                return None

    def lookup_raw(self, outputs: Union[Tuple[int], List[int]]) -> Optional[str]:
        outputs = outputs if isinstance(outputs, tuple) else tuple(outputs)
        h = hash(outputs)
        return self.lookup_table.get(h, None)

    @property
    def name(self) -> Path:
        return self._name

    @property
    def bitsize(self):
        return self._bitsize

    @property
    def var_number(self):
        return len(self.grammar.vars)

    @property
    def operator_number(self):
        return len(self.grammar.ops)

    def generate(self, depth, max_count=0, unsigned: bool = False):
        def to_uint(x):
            return (x if x > 0 else (1 << self._bitsize) + x) if unsigned else x
        #inputs = {k.decl().name(): [x.as_long() for x in v] for k, v in self.inputs.items()}
        # convert z3 to pure str->int values
        inputs = {k.decl().name(): [inp[k].as_long() for inp in self.inputs] for k in self.inputs[0].keys()}

        t0 = time()
        worklist: List[Tuple[str, Tuple[int]]] = [(k, v) for k, v in inputs.items()]
        self.lookup_table = {hash(tuple(v)): k for k, v in inputs.items()}  # initialize lookup table with vars singleton
        ops = self.grammar.non_terminal_operators
        cur_depth = depth-1
        while cur_depth > 0:
            # Start a new depth
            n_items = len(worklist)
            t = time() - t0
            print(f"Depth {depth-cur_depth} (size:{n_items}) (Time:{int(t/60)}m{t%60:.5f}s)")

            for op in ops:  # Iterate over all operators
                print(f"  op: {op.symbol}")
                if op.arity == 1:
                    for i1 in range(n_items):  # iterate once the list
                        name, vals = worklist[i1]
                        new_vals = tuple(map(lambda x: to_uint(op.op_py(x)), vals))
                        h = hash(new_vals)
                        if h not in self.lookup_table:
                            fmt = f"{op.symbol}{name}"
                            logging.debug(f"[add] {fmt}")
                            self.lookup_table[h] = fmt
                            worklist.append((fmt, new_vals))  # add it in worklist if not already in LUT
                        else:
                            logging.debug(f"[drop] {op.symbol}{name}  [{self.lookup_table[h]}]")
                else:  # arity is 2
                    blacklist = set()
                    for i1 in range(n_items):
                        if len(worklist) > max_count > 0:
                            print("Max count exceeded, break")
                            break
                        name1, vals1 = worklist[i1]
                        for i2 in range(n_items):
                            name2, vals2 = worklist[i2]

                            # for identity (a op a) ignore it if the result is known to be 0 or a
                            if i1 == i2 and (op.id_eq or op.id_zero):
                                continue

                            # Ignore expression if they are in the blacklist
                            fmt = f"{op.symbol}({name1},{name2})" if op.is_prefix else f"({name1}{op.symbol}{name2})"
                            if fmt in blacklist:
                                continue

                            new_vals = tuple(map(lambda x: to_uint(op.op_py(*x)), zip(vals1, vals2)))  # compute new vals
                            h = hash(new_vals)
                            if h not in self.lookup_table:
                                logging.debug(f"[add] {fmt}")
                                self.lookup_table[h] = fmt
                                worklist.append((fmt, new_vals))

                                if op.commutative:
                                    fmt = f"{op.symbol}({name2},{name1})" if op.is_prefix else f"({name2}{op.symbol}{name1})"
                                    blacklist.add(fmt)  # blacklist commutative equivalent e.g for a+b blacklist: b+a
                                    logging.debug(f"[blacklist] {fmt}")
                            else:
                                logging.debug(f"[drop] {fmt}  [{self.lookup_table[h]}]")
            cur_depth -= 1

        # In the end
        t = time() - t0
        print(f"Depth {depth - cur_depth} (size:{len(worklist)}) (Time:{int(t/60)}m{t%60:.5f}s)")

    def dump(self, file: Union[Path, str]) -> None:
        with open(file, 'wb') as f:
            pickle.dump(self.grammar.to_dict(), f)
            pickle.dump(self.grammar.dump_inputs(self.inputs), f)
            pickle.dump(self.lookup_table, f)

    @staticmethod
    def load(file: Union[Path, str], grammar_class) -> 'LookupTable':
        f = Path(file)
        with open(f, 'rb') as f:
            gr = grammar_class.from_dict(pickle.load(f))
            inp_l = pickle.load(f)
            inputs = grammar_class.load_inputs(inp_l)
            lkp = LookupTable(gr, inputs, f.name)
            lkp.lookup_table = pickle.load(f)
            return lkp
