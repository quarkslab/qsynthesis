from pathlib import Path
from qsynthesis.grammar import TritonGrammar, BvOp
import logging
from enum import IntEnum, Enum
import array
import hashlib
import threading
import psutil
from pony.orm import Database, Required, PrimaryKey, db_session, IntArray, StrArray, count, commit
from pony.orm.dbapiprovider import IntConverter
from triton import TritonContext, ARCH
from typing import Optional, List, Dict, Union, Tuple, TypeVar, Iterable
from time import time, sleep

Expr = TypeVar('Expr')  # Expression type in the associated grammar


class HashType(IntEnum):
    RAW = 1
    FNV1A_128 = 2
    MD5 = 3


db = Database()


class TableEntry(db.Entity):
    hash = PrimaryKey(bytes)
    expression = Required(str)


class Metadata(db.Entity):
    hash_mode = Required(HashType)
    operators = Required(IntArray)


class Variable(db.Entity):
    name = PrimaryKey(str)
    size = Required(int)


class Input(db.Entity):
    id = PrimaryKey(int, auto=True)
    variables = Required(StrArray)
    values = Required(IntArray)


class EnumConverter(IntConverter):
    def validate(self, val, obj=None):
        if not isinstance(val, Enum):
            raise ValueError('Must be an Enum.  Got {}'.format(type(val)))
        return val

    def py2sql(self, val):
        return val.value

    def sql2py(self, value):
        # Any enum type can be used, so py_type ensures the correct one is used to create the enum instance
        return self.py_type(value)


class _EvalCtx(object):
    def __init__(self, grammar):
        # Create the context
        self.ctx = TritonContext(ARCH.X86_64)
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


class LookupTableDB:
    def __init__(self, grammar: TritonGrammar, inputs: List[Dict[str, int]], hash_mode: HashType = HashType.RAW, f_name: str = ""):
        if db.provider_name is None:
            logging.error("LookupTableDB object should be created with a databased already binded !")
        self._name = Path(f_name)
        self.grammar = grammar
        self._bitsize = self.grammar.size
        self.expr_cache = {}
        self.lookup_count = 0
        self.lookup_found = 0
        self.cache_hit = 0
        self.hash_mode = hash_mode
        self._ectx = None
        # generation related fields
        self.watchdog = None
        self.stop = False

        self.inputs = inputs

    @staticmethod
    def create(filename: Union[str, Path], grammar: TritonGrammar, inputs: List[Dict[str, int]], hash_mode: HashType = HashType.RAW) -> 'LookupTableDB':
        db.bind(provider='sqlite', filename=str(filename), create_db=True)
        db.provider.converter_classes.append((Enum, EnumConverter))
        db.generate_mapping(create_tables=True)
        with db_session:
            Metadata(hash_mode=hash_mode, operators=[x.value for x in grammar.ops])
            for name, sz in grammar.vars_dict.items():
                Variable(name=name, size=sz)
            for i in inputs:
                Input(variables=list(i.keys()), values=list(i.values()))
        return LookupTableDB(grammar=grammar, inputs=inputs, hash_mode=hash_mode, f_name=filename)

    @staticmethod
    def load(file: Union[Path, str]) -> 'LookupTableDB':
        file = file.absolute() if isinstance(file, Path) else Path(file).absolute()
        db.bind(provider='sqlite', filename=str(file), create_db=False)
        db.provider.converter_classes.append((Enum, EnumConverter))
        db.generate_mapping(create_tables=False)

        with db_session:
            inputs = [{n: v for n, v in zip(i.variables, i.values)} for i in Input.select()]
            vars = [(x.name, x.size) for x in Variable.select()]
            m = Metadata.select().first()
            ops = [BvOp(x) for x in m.operators]
            gr = TritonGrammar(vars=vars, ops=ops)
            return LookupTableDB(grammar=gr, inputs=inputs, hash_mode=HashType(m.hash_mode), f_name=file)

    def _add_entry(self, hash: bytes, value: str):
        with db_session:
            TableEntry(hash=hash, expression=value)

    # @db_session
    def _add_entries(self, entries: List[Tuple[str, List[int]]], chunk_size=10000) -> None:
        count = len(entries)
        hash_fun = lambda x: hashlib.md5(bytes(x)).digest() if self.hash_mode == HashType.MD5 else self.hash
        for step in range(0, count, chunk_size):
            print(f"process {step}/{count}\r", end="")
            with db_session:
                for s, outs in entries[step:step+chunk_size]:
                    TableEntry(hash=hash_fun(outs), expression=s)

        # for i, (s, outs) in enumerate(entries):
        #     if i % 1000 == 0:
        #         print(f"process {i}/{count}\r", end="")
        #     TableEntry(hash=self.hash(outs), expression=s)

    @db_session
    def __iter__(self):
        for entry in TableEntry.select():
            yield entry.hash, entry.expression

    @property
    def size(self):
        with db_session:
            return count(TableEntry.select())

    def lookup(self, outputs: Union[Tuple[int], List[int]], *args,  use_cache=True) -> Optional[Expr]:
        self.lookup_count += 1
        h = self.hash(outputs)
        if h in self.expr_cache and use_cache:
            self.cache_hit += 1
            return self.expr_cache[h]
        else:
            with db_session:
                v = TableEntry.get(hash=h)
            if v:
                self.lookup_found += 1
                e = self.grammar.str_to_expr(v.expression, *args)
                self.expr_cache[h] = e
                return e
            else:
                return None

    def lookup_raw(self, outputs: Union[Tuple[int], List[int]]) -> Optional[str]:
        h = self.hash(outputs)
        with db_session:
            entry = TableEntry.get(hash=h)
        return entry.expression if entry else None

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

    def watchdog_worker(self, threshold):
        while not self.stop:
            sleep(2)
            mem = psutil.virtual_memory()
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

                                fmt = f"{op.symbol}({name1},{name2})" if op.is_prefix else f"({name1}){op.symbol}({name2})"

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
                                        fmt = f"{op.symbol}({name2},{name1})" if op.is_prefix else f"({name2}){op.symbol}({name1})"
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
        print(f"Depth {depth - cur_depth} (size:{len(worklist)}) (Time:{int(t/60)}m{t%60:.5f}s)")
        self._add_entries(worklist)
        if do_watch:
            self.watchdog.join()
