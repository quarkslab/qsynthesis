import pickle
from pathlib import Path
from qsynthesis.grammar import TritonGrammar
import logging
from enum import IntEnum
import array
import hashlib
import threading
import psutil

from typing import Optional, List, Dict, Union, Generator, Tuple, Any, TypeVar, Iterable
from time import time, sleep

Expr = TypeVar('Expr')  # Expression type in the associated grammar
Hash = Union[bytes, int, Tuple[int]]


class HashType(IntEnum):
    RAW = 1
    FNV1A_128 = 2
    MD5 = 3


class _EvalCtx(object):
    def __init__(self, grammar):
        from triton import TritonContext, ARCH, AST_REPRESENTATION
        # Create the context
        self.ctx = TritonContext(ARCH.X86_64)
        self.ctx.setAstRepresentationMode(AST_REPRESENTATION.PYTHON)
        self.ast = self.ctx.getAstContext()

        # Create symbolic variables for grammar variables
        self.symvars = {}
        self.vars = {}
        for v, sz in grammar.vars_dict.items():
            sym_v = self.ctx.newSymbolicVariable(sz, v)
            self.symvars[v] = sym_v
            self.vars[v] = self.ast.variable(sym_v)

        # Create mapping to triton operators
        self.tops = {x: getattr(self.ast, x) for x in dir(self.ast) if not x.startswith("__")}

    def eval_str(self, s: str) -> 'AstNode':
        e = eval(s, self.tops, self.vars)
        if isinstance(e, int):  # In case the expression was in fact an int
            return self.ast.bv(e, 64)
        else:
            return e

    def set_symvar_values(self, args) -> None:
        for v_name, value in args.items():
            self.ctx.setConcreteVariableValue(self.symvars[v_name], value)


class BaseTable:
    """
    Base Lookup table class. Specify the interface that child
    class have to implement to be interoperable with other the synthesizer
    """
    def __init__(self, gr: TritonGrammar, inputs: Union[int, List[Dict[str, int]]], hash_mode: HashType=HashType.RAW, f_name: str = ""):
        self._name = Path(f_name)
        self.grammar = gr
        self._bitsize = self.grammar.size
        self.expr_cache = {}
        self.lookup_count = 0
        self.lookup_found = 0
        self.cache_hit = 0
        self.hash_mode = hash_mode
        self._ectx = None
        # generation related fields
        self.watchdog = None
        self.max_mem = 0
        self.stop = False

        self.inputs = inputs

    @property
    def size(self):
        raise NotImplementedError("Should be implemented by child class")

    def _get_item(self, h: Hash) -> Optional[str]:
        raise NotImplementedError("Should be implemented by child class")

    def lookup(self, outputs: Union[Tuple[int], List[int]], *args,  use_cache=True) -> Optional[Expr]:
        self.lookup_count += 1
        h = self.hash(outputs)
        if h in self.expr_cache and use_cache:
            self.cache_hit += 1
            return self.expr_cache[h]
        else:
            v = self._get_item(h)
            if v:
                self.lookup_found += 1
                e = self.grammar.str_to_expr(v, *args)
                self.expr_cache[h] = e
                return e
            else:
                return None

    def lookup_raw(self, outputs: Union[Tuple[int], List[int]]) -> Optional[str]:
        h = self.hash(outputs)
        return self._get_item(h)

    def lookup_hash(self, h: Hash) -> Optional[str]:
        return self._get_item(h)

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

    @property
    def input_number(self):
        return len(self.inputs)

    @staticmethod
    def fnv1a_128(outs) -> int:
        a = array.array('Q', outs)
        FNV1A_128_OFFSET = 0x6c62272e07bb014262b821756295c58d
        FNV1A_128_PRIME = 0x1000000000000000000013b  # 2^88 + 2^8 + 0x3b

        # Set the offset basis
        hash = FNV1A_128_OFFSET

        # For each character
        for byte in a.tobytes():
            # Xor with the current character
            hash ^= byte
            # Multiply by prime
            hash *= FNV1A_128_PRIME
            # Clamp
            hash &= 0xffffffffffffffffffffffffffffffff
        # Return the final hash as a number
        return hash

    @staticmethod
    def md5(outs) -> bytes:
        a = array.array('Q', outs)
        h = hashlib.md5(a.tobytes())
        return h.digest()

    def hash(self, outs):
        if self.hash_mode == HashType.RAW:
            return tuple(outs)
        elif self.hash_mode == HashType.FNV1A_128:
            return self.fnv1a_128(outs)
        elif self.hash_mode == HashType.MD5:
            return self.md5(outs)

    def __iter__(self) -> Iterable[Tuple[Hash, str]]:
        raise NotImplementedError("Should be implemented by child class")

    def get_expr(self, expr: str):
        if self._ectx is None:
            self._ectx = _EvalCtx(self.grammar)
        return self._ectx.eval_str(expr)

    def set_input_lcontext(self, i: Union[int, Dict]):
        if self._ectx is None:
            self._ectx = _EvalCtx(self.grammar)
        self._ectx.set_symvar_values(self.inputs[i] if isinstance(i, int) else i)

    def eval_expr_inputs(self, expr) -> List[int]:
        outs = []
        for i in range(len(self.inputs)):
            self.set_input_lcontext(i)
            outs.append(expr.evaluate())
        return outs

    def watchdog_worker(self, threshold):
        while not self.stop:
            sleep(2)
            mem = psutil.virtual_memory()
            self.max_mem = max(mem.used, self.max_mem)
            if mem.percent >= threshold:
                logging.warning(f"Threshold reached: {mem.percent}%")
                self.stop = True  # Should stop self and also main thread

    @staticmethod
    def try_linearize(s: str, symbols) -> str:
        import sympy
        try:
            lin = eval(s, symbols)
            if isinstance(lin, sympy.boolalg.BooleanFalse):
                logging.error(f"[linearization] expression {s} False")
            logging.debug(f"[linearization] expression linearized {s} => {lin}")
            return str(lin).replace(" ", "")
        except TypeError:
            return s
        except AttributeError as e:
            return s

    def generate(self, depth, max_count=0, do_watch=False, watchdog_threshold=90, linearize=True):
        if do_watch:
            self.watchdog = threading.Thread(target=self.watchdog_worker, args=[watchdog_threshold], daemon=True)
            logging.info("Start watchdog")
            self.watchdog.start()
        if linearize:
            logging.info("Linearization enabled")
            import sympy
            symbols = {x: sympy.symbols(x) for x in self.grammar.vars}
        t0 = time()

        import pydffi
        FFI = pydffi.FFI()
        N = self.input_number
        ArTy = FFI.arrayType(FFI.ULongLongTy, N)

        hash_fun = lambda x: hashlib.md5(bytes(x)).digest() if self.hash_mode == HashType.MD5 else self.hash
        worklist = [(k, ArTy()) for k in self.grammar.vars]
        for i, inp in enumerate(self.inputs):
            for k, v in worklist:
                v[i] = inp[k]
        hash_set = set(hash_fun(x[1]) for x in worklist)

        ops = self.grammar.non_terminal_operators
        cur_depth = depth-1
        blacklist = set()

        try:
            while cur_depth > 0:
                # Start a new depth
                n_items = len(worklist)
                t = time() - t0
                print(f"Depth {depth-cur_depth} (size:{n_items}) (Time:{int(t/60)}m{t%60:.5f}s)")

                for op in ops:  # Iterate over all operators
                    print(f"  op: {op.symbol}")

                    if op.arity == 1:
                        for i1 in range(n_items):  # iterate once the list
                            if self.stop:
                                logging.warning("Threshold reached, generation interrupted")
                                raise KeyboardInterrupt()
                            name, vals = worklist[i1]

                            new_vals = ArTy()
                            op.eval_a(new_vals, vals, N)
                            #new_vals = tuple(map(lambda x: op.eval(x), vals))

                            #h = self.hash([x.value for x in new_vals])
                            h = hash_fun(new_vals)
                            if h not in hash_set:
                                fmt = f"{op.symbol}({name})" if len(name) > 1 else f"{op.symbol}{name}"
                                fmt = self.try_linearize(fmt, symbols) if linearize else fmt
                                logging.debug(f"[add] {fmt: <20} {h}")
                                hash_set.add(h)
                                worklist.append((fmt, new_vals))  # add it in worklist if not already in LUT
                            else:
                                logging.debug(f"[drop] {op.symbol}{name}  ")

                    else:  # arity is 2
                        for i1 in range(n_items):
                            if len(worklist) > max_count > 0:
                                print("Max count exceeded, break")
                                break
                            name1, vals1 = worklist[i1]
                            for i2 in range(n_items):
                                if self.stop:
                                    logging.warning("Threshold reached, generation interrupted")
                                    raise KeyboardInterrupt()
                                name2, vals2 = worklist[i2]

                                # for identity (a op a) ignore it if the result is known to be 0 or a
                                if i1 == i2 and (op.id_eq or op.id_zero):
                                    continue

                                sn1 = f'{name1}' if len(name1) == 1 else f'({name1})'
                                sn2 = f'{name2}' if len(name2) == 1 else f'({name2})'
                                fmt = f"{op.symbol}({name1},{name2})" if op.is_prefix else f"{sn1}{op.symbol}{sn2}"

                                if not linearize:
                                    if fmt in blacklist:  # Ignore expression if they are in the blacklist
                                        continue

                                new_vals = ArTy()
                                op.eval_a(new_vals, vals1, vals2, N)
                                #new_vals = tuple(map(lambda x: op.eval(*x), zip(vals1, vals2)))  # compute new vals

                                #h = self.hash([x.value for x in new_vals])   # Strip pydffi before hashing
                                h = hash_fun(new_vals)
                                if h not in hash_set:
                                    if linearize:
                                        fmt = self.try_linearize(fmt, symbols) if linearize else fmt
                                        if fmt in blacklist:  # if linearize check blacklist here
                                            continue

                                    logging.debug(f"[add] {fmt: <20} {h}")
                                    hash_set.add(h)
                                    worklist.append((fmt, new_vals))

                                    if op.commutative:
                                        fmt = f"{op.symbol}({name2},{name1})" if op.is_prefix else f"{sn2}{op.symbol}{sn1}"
                                        fmt = self.try_linearize(fmt, symbols) if linearize else fmt
                                        blacklist.add(fmt)  # blacklist commutative equivalent e.g for a+b blacklist: b+a
                                        logging.debug(f"[blacklist] {fmt}")
                                else:
                                    logging.debug(f"[drop] {op.symbol}({name1},{name2})" if op.is_prefix else f"[drop] ({name1}){op.symbol}({name2})")

                cur_depth -= 1
        except KeyboardInterrupt:
            logging.info("Stop required")
        # In the end
        self.stop = True
        t = time() - t0
        print(f"Depth {depth - cur_depth} (size:{len(worklist)}) (Time:{int(t/60)}m{t%60:.5f}s) [RAM:{self.__size_to_str(self.max_mem)}]")
        self.add_entries(worklist)
        if do_watch:
            self.watchdog.join()

    def add_entry(self, hash: Hash, value: str) -> None:
        raise NotImplementedError("Should be implemented by child class")

    def add_entries(self, worklist):
        raise NotImplementedError("Should be implemented by child class")

    @staticmethod
    def create(filename: Union[str, Path], grammar: TritonGrammar, inputs: List[Dict[str, int]], hash_mode: HashType = HashType.RAW) -> 'BaseTable':
        raise NotImplementedError("Should be implemented by child class")

    @staticmethod
    def load(file: Union[Path, str]) -> 'BaseTable':
        raise NotImplementedError("Should be implemented by child class")

    def save(self, file: Optional[Union[Path, str]]):
        raise NotImplementedError("Should be implemented by child class")

    @staticmethod
    def __size_to_str(value):
        units = [(float(1024), "Kb"), (float(1024 **2), "Mb"), (float(1024 **3), "Gb")]
        for unit, s in units[::-1]:
            if value / unit < 1:
                continue
            else:  # We are on the right unit
                return f"{value/unit:.2f}{s}"
        return f"{value}B"