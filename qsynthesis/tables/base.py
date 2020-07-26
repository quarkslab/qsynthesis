# built-in libs
from pathlib import Path
import logging
from enum import IntEnum
import array
import hashlib
import threading
from time import time, sleep

# third-party libs
import psutil

# qsynthesis deps
from qsynthesis.grammar import TritonGrammar
from qsynthesis.tritonast import TritonAst
from qsynthesis.types import AstNode, Hash, Optional, List, Dict, Union, Tuple, Iterable, Input, Output, BitSize, Any, \
                             Generator


class HashType(IntEnum):
    """
    Hash types supported by the Lookup table database. In practice solely md5
    is used, has it is the fastest of all
    """
    RAW = 1
    FNV1A_128 = 2
    MD5 = 3


class _EvalCtx(object):
    """
    Small debugging Triton evaluation context. It is used when manipulating
    tables in a standalone manner. It enables obtaining TritonAst out of
    the databqse entries.
    """
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

    def eval_str(self, s: str) -> AstNode:
        """Eval the string expression to create an AstNode object"""
        e = eval(s, self.tops, self.vars)
        if isinstance(e, int):  # In case the expression was in fact an int
            return self.ast.bv(e, 64)
        else:
            return e

    def set_symvar_values(self, args: Input) -> None:
        for v_name, value in args.items():
            self.ctx.setConcreteVariableValue(self.symvars[v_name], value)


class LookupTable:
    """
    Base Lookup table class. Specify the interface that child class have to
    implement to be interoperable with other the synthesizer.
    """
    def __init__(self, gr: TritonGrammar, inputs: List[Input], hash_mode: HashType = HashType.RAW, f_name: str = ""):
        """
        Constructor making a lookuptable from a grammar a set of inputs and an hash type.

        :param gr: triton grammar
        :param inputs: List of inputs
        :param hash_mode: type of hash to be used as keys in tables
        :param f_name: file name of the table (when being loaded)
        """
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
    def size(self) -> int:
        """Size of the table (number of entries)"""
        raise NotImplementedError("Should be implemented by child class")

    def _get_item(self, h: Hash) -> Optional[str]:
        """
        From a given hash return the associated expression string if
        found in the lookup table.

        :param h: hash of the item to get
        :returns: raw expression string if found
        """
        raise NotImplementedError("Should be implemented by child class")

    def lookup(self, outputs: List[Output], *args,  use_cache: bool = True) -> Optional[TritonAst]:
        """
        Perform a lookup in the table with a given set of outputs corresponding
        to the evaluation of an AST against the Input of this exact same table.
        If an entry is found a TritonAst is created and returned.

        :param outputs: list of output result of evaluating an ast against the inputs of this table
        :param args: args forwarded to grammar and ultimately to the tritonAst in charge of build a
                     new TritonAst.
        :param use_cache: Boolean enabling caching the the hash of outputs. A second call if the same outputs
                          (which is common) will not trigger a lookup in the database
        :returns: optional TritonAst corresponding of the expression found in the table
        """
        self.lookup_count += 1
        h = self.hash(outputs)
        if h in self.expr_cache and use_cache:
            self.cache_hit += 1
            return self.expr_cache[h]
        else:
            v = self._get_item(h)
            if v:
                self.lookup_found += 1
                try:
                    e = self.grammar.str_to_expr(v, *args)
                    self.expr_cache[h] = e
                    return e
                except NameError:
                    return None
                except TypeError:
                    return None
            else:
                return None

    def lookup_raw(self, outputs: List[Output]) -> Optional[str]:
        """
        Identical to :meth:`~LookupTable.lookup` but does not create the TritonAst
        object. Simply returns the expression string.

        :param outputs: list of output result of evaluating an ast against the inputs of this table
        :returns: string of the expression if found
        """
        h = self.hash(outputs)
        return self._get_item(h)

    def lookup_hash(self, h: Hash) -> Optional[str]:
        """
        Raw lookup for a given key in database.

        :param h: hash key to look for in database
        :returns: string of the expression if found
        """
        return self._get_item(h)

    @property
    def is_writable(self) -> bool:
        """ Whether the table enable being written (with new expressions) """
        return False

    @property
    def name(self) -> str:
        """ Name of the table """
        return str(self._name)

    @property
    def bitsize(self) -> BitSize:
        """ Size of expression in bit """
        return self._bitsize

    @property
    def var_number(self) -> int:
        """ Maximum number of variables contained in the table """
        return len(self.grammar.vars)

    @property
    def operator_number(self) -> int:
        """ Number of operators used in this table """
        return len(self.grammar.ops)

    @property
    def input_number(self) -> int:
        """ Number of inputs used in this table """
        return len(self.inputs)

    @staticmethod
    def fnv1a_128(outs: List[Output]) -> Hash:
        """
        Hash the outputs using fnv1a_128 algorithm

        :param outs: list of outputs to hash
        :returns: Hash value (int) corresponding to the fnv1a of outputs
        """
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
    def md5(outs: List[Output]) -> Hash:
        """
        Hash the outputs using MD5 algorithm. Outputs are transformed into an array.
        That means the final bytes hashed are the concatenation of uint64 in little
        endian.

        :param outs: list of outputs to hash
        :returns: Bytes corresponding to MD5 hash
        """
        a = array.array('Q', outs)
        h = hashlib.md5(a.tobytes())
        return h.digest()

    def hash(self, outs: List[Output]) -> Hash:
        """
        Main hashing method that dispatch the outputs to the appropriate hashing
        function depending on the ``hash_mode`` of the table.

        :param outs: list of outputs to hash
        :returns: Hash type (bytes, int ..) of the outputs
        """
        if self.hash_mode == HashType.RAW:
            return tuple(outs)
        elif self.hash_mode == HashType.FNV1A_128:
            return self.fnv1a_128(outs)
        elif self.hash_mode == HashType.MD5:
            return self.md5(outs)

    def __iter__(self) -> Iterable[Tuple[Hash, str]]:
        """ Iterator of all the entries as an iterator of pair, hash, expression as string """
        raise NotImplementedError("Should be implemented by child class")

    def get_expr(self, expr: str) -> TritonAst:
        """
        Utility function that returns a TritonAst from a given expression string.
        A TritonContext local to the table is created to enable generating such ASTs.

        :param expr: Expression
        :returns: TritonAst resulting of the parsing of s
        """
        if self._ectx is None:
            self._ectx = _EvalCtx(self.grammar)
        return self._ectx.eval_str(expr)

    def set_input_lcontext(self, i: Union[int, Input]) -> None:
        """
        Set the given concrete values of variables in the local TritonContext.
        The parameter is either the ith input of the table, or directly an Input
        given a valuation for each variables. This function must be called before
        performing any evaluation of an AST.

        :param i: index of the input, or Input object (dict)
        :returns: None
        """
        if self._ectx is None:
            self._ectx = _EvalCtx(self.grammar)
        self._ectx.set_symvar_values(self.inputs[i] if isinstance(i, int) else i)

    def eval_expr_inputs(self, expr: AstNode) -> List[Output]:
        """
        Evaluate a given Triton AstNode object on all inputs of the
        table. The result is a list of Output values.

        :param expr: Triton AstNode to evaluate
        :returns: list of output values (ready to be hashed)
        """
        outs = []
        for i in range(len(self.inputs)):
            self.set_input_lcontext(i)
            outs.append(expr.evaluate())
        return outs

    def watchdog_worker(self, threshold: Union[float, int]) -> None:
        """
        Function where the memory watchdog thread is running. This function
        allows interrupting table generation when it happens to fill the
        given threshold of RAM.

        :param threshold: percentage of RAM load that triggers the stop of generation
        """
        while not self.stop:
            sleep(2)
            mem = psutil.virtual_memory()
            self.max_mem = max(mem.used, self.max_mem)
            if mem.percent >= threshold:
                logging.warning(f"Threshold reached: {mem.percent}%")
                self.stop = True  # Should stop self and also main thread

    @staticmethod
    def try_linearize(s: str, symbols: Dict[str, object]) -> str:
        """
        Try applying sympy to linearize ``s`` with the variable symbols
        ``symbols``. If any exception is raised in between to expression
        string is returned unchanged.

        :param s: expression string to linearize
        :param symbols: dictionnary of variables names to sympy symbol objects
        """
        import sympy
        try:
            lin = eval(s, symbols)
            if isinstance(lin, sympy.boolalg.BooleanFalse):
                logging.error(f"[linearization] expression {s} False")
            logging.debug(f"[linearization] expression linearized {s} => {lin}")
            return str(lin).replace(" ", "")
        except TypeError:
            return s
        except AttributeError as _:
            return s

    @staticmethod
    def custom_permutations(l: List[Any]) -> Generator[Tuple[bool, Any, Any], None, None]:
        """
        Custom generator generating all the possible tuples from a list. But instead
        of iterating item i with all others 0..n, iterates i with all the previous 0..i.
        It generates a somewhat sorted generated that ensure pairs of items appearing
        first in the list will be yielded before.

        :param l: list of any item
        :returns: genreator of tuples generating all possibles pairs
        """
        for i in range(len(l)):
            for j in range(0, i):
                yield False, l[i], l[j]
                yield False, l[j], l[i]
            yield True, l[i], l[i]

    def generate(self, do_watch: bool = False, watchdog_threshold: Union[int, float] = 90, linearize: bool = False, do_use_blacklist: bool = False) -> None:
        """
        Generate a new lookup table from scratch with the variables and operators
        set in the constructor of the table.

        :param do_watch: Enable RAM watching thread to monitor memory
        :param watchdog_threshold: threshold to be sent to the memory watchdog
        :param linearize: whether or not to apply linearization on expressions
        :param do_use_blacklist: enable blacklist mechanism on commutative operators. Slower but less memory consuming
        :returns: None
        """
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
        worklist = [(ArTy(), k) for k in self.grammar.vars]
        for i, inp in enumerate(self.inputs):
            for v, k in worklist:
                v[i] = inp[k]
        hash_set = set(hash_fun(x[0]) for x in worklist)

        ops = sorted(self.grammar.non_terminal_operators, key=lambda x: x.arity == 1)  # sort operators to iterate on unary first
        cur_depth = 2
        blacklist = set()

        try:
            while cur_depth > 0:
                # Start a new depth
                n_items = len(worklist)
                t = time() - t0
                print(f"Depth {cur_depth} (size:{n_items}) (Time:{int(t/60)}m{t%60:.5f}s)")
                c = 0

                for i, (same, (vals1, name1), (vals2, name2)) in enumerate(self.custom_permutations(worklist)):
                    if same:
                        c += 1
                        print(f"process: {(c*100)/n_items:.2f}%\r", end="")
                    if self.stop:
                        logging.warning("Threshold reached, generation interrupted")
                        raise KeyboardInterrupt()

                    for op in ops:  # Iterate over all operators
                        if op.arity == 1:
                            new_vals = ArTy()
                            op.eval_a(new_vals, vals1, N)

                            h = hash_fun(new_vals)
                            if h not in hash_set:
                                fmt = f"{op.symbol}({name1})" if len(name1) > 1 else f"{op.symbol}{name1}"
                                fmt = self.try_linearize(fmt, symbols) if linearize else fmt
                                logging.debug(f"[add] {fmt: <20} {h}")
                                hash_set.add(h)
                                worklist.append((new_vals, fmt))  # add it in worklist if not already in LUT
                            else:
                                logging.debug(f"[drop] {op.symbol}{name1}  ")

                        else:  # arity is 2
                            # for identity (a op a) ignore it if the result is known to be 0 or a
                            if same and (op.id_eq or op.id_zero):
                                continue

                            sn1 = f'{name1}' if len(name1) == 1 else f'({name1})'
                            sn2 = f'{name2}' if len(name2) == 1 else f'({name2})'
                            fmt = f"{op.symbol}({name1},{name2})" if op.is_prefix else f"{sn1}{op.symbol}{sn2}"

                            if not linearize:
                                if fmt in blacklist:  # Ignore expression if they are in the blacklist
                                    continue

                            new_vals = ArTy()
                            op.eval_a(new_vals, vals1, vals2, N)

                            h = hash_fun(new_vals)
                            if h not in hash_set:
                                if linearize:
                                    fmt = self.try_linearize(fmt, symbols) if linearize else fmt
                                    if fmt in blacklist:  # if linearize check blacklist here
                                        continue

                                logging.debug(f"[add] {fmt: <20} {h}")
                                hash_set.add(h)
                                worklist.append((new_vals, fmt))

                                if op.commutative and do_use_blacklist:
                                    fmt = f"{op.symbol}({name2},{name1})" if op.is_prefix else f"{sn2}{op.symbol}{sn1}"
                                    fmt = self.try_linearize(fmt, symbols) if linearize else fmt
                                    blacklist.add(fmt)  # blacklist commutative equivalent e.g for a+b blacklist: b+a
                                    logging.debug(f"[blacklist] {fmt}")
                            else:
                                logging.debug(f"[drop] {op.symbol}({name1},{name2})" if op.is_prefix else f"[drop] ({name1}){op.symbol}({name2})")

                cur_depth += 1
        except KeyboardInterrupt:
            logging.info("Stop required")
        # In the end
        self.stop = True
        t = time() - t0
        print(f"Depth {cur_depth} (size:{len(worklist)}) (Time:{int(t/60)}m{t%60:.5f}s) [RAM:{self.__size_to_str(self.max_mem)}]")
        self.add_entries(worklist, calc_hash=True)
        if do_watch:
            self.watchdog.join()

    def add_entry(self, hash: Hash, value: str) -> None:
        """
        Abstract function to add an entry in the lookuptable.

        :param hash: already computed hash to add
        :param value: expression value to add in the table
        """
        raise NotImplementedError("Should be implemented by child class")

    def add_entries(self, worklist: List[Tuple[Hash, str]], calc_hash: bool = False) -> None:
        """
        Add the given list of entries in the database. The boolean ``calc_hash`` indicates
        whether hashes are already computed or not. If false the function should hash the
        hash first.

        :param worklist: list of entries to add
        :param calc_hash: whether or not hash should be performed on entries keys
        :returns: None
        """
        raise NotImplementedError("Should be implemented by child class")

    @staticmethod
    def create(filename: Union[str, Path], grammar: TritonGrammar, inputs: List[Input], hash_mode: HashType = HashType.RAW) -> 'LookupTable':
        """
        Create a new empty lookup table with the given initial parameters, grammars, inputs
        and hash_mode.

        :param filename: filename of the table to create
        :param grammar: TritonGrammar object representing variables and operators
        :param inputs: list of inputs on which to perform evaluation
        :param hash_mode: Hashing mode for keys
        :returns: lookuptable instance object
        """
        raise NotImplementedError("Should be implemented by child class")

    @staticmethod
    def load(file: Union[Path, str]) -> 'LookupTable':
        """
        Load the given lookup table and returns an instance object.

        :param file: Database file to load
        :returns: LookupTable object
        """
        raise NotImplementedError("Should be implemented by child class")

    @staticmethod
    def __size_to_str(value: int) -> str:
        """ Return pretty printed representation of RAM usage for table generation """
        units = [(float(1024), "Kb"), (float(1024 ** 2), "Mb"), (float(1024 ** 3), "Gb")]
        for unit, s in units[::-1]:
            if value / unit < 1:
                continue
            else:  # We are on the right unit
                return f"{value/unit:.2f}{s}"
        return f"{value}B"
