import triton
from triton import TritonContext, AST_NODE, SYMBOLIC, ARCH
import logging
# Imports used only for typing
from typing import List, Tuple, Generator, Dict, Union, Optional, TypeVar
from qtracedb.archs.manager import ArchsManager
from qtracedb.archs.arch import Arch, Instr
from enum import IntEnum
import random
from functools import reduce


AstType = IntEnum("AstType", {k: v for k, v in triton.AST_NODE.__dict__.items() if isinstance(v, int)})
AstNode = TypeVar('AstNode')  # Triton AstNode object instance


class ReassemblyError(Exception):
    pass


class SymVarType(IntEnum):
    REGISTER = SYMBOLIC.REGISTER_VARIABLE
    MEMORY = SYMBOLIC.MEMORY_VARIABLE


op_str = {AST_NODE.ANY: "any", AST_NODE.ASSERT: "assert", AST_NODE.BV: "bv", AST_NODE.BVADD: "+",
          AST_NODE.BVAND: "&", AST_NODE.BVASHR: "bvashr", AST_NODE.BVLSHR: "bvlshr", AST_NODE.BVMUL: "*",
          AST_NODE.BVNAND: "bvnand", AST_NODE.BVNEG: "-", AST_NODE.BVNOR: "bvnor", AST_NODE.BVNOT: "not",
          AST_NODE.BVOR: "|", AST_NODE.BVROL: "bvrol", AST_NODE.BVROR: "bvror", AST_NODE.BVSDIV: "bvsdiv",
          AST_NODE.BVSGE: ">=s", AST_NODE.BVSGT: ">s", AST_NODE.BVSHL: "<<", AST_NODE.BVSLE: "<=s",
          AST_NODE.BVSLT: "<s", AST_NODE.BVSMOD: "bvsmod", AST_NODE.BVSREM: "bvsrem", AST_NODE.BVSUB: "-",
          AST_NODE.BVUDIV: "bvudiv", AST_NODE.BVUGE: ">=u", AST_NODE.BVUGT: ">u", AST_NODE.BVULE: "<=u",
          AST_NODE.BVULT: "<u", AST_NODE.BVUREM: "bvurem", AST_NODE.BVXNOR: "bvxnor", AST_NODE.BVXOR: "^",
          AST_NODE.COMPOUND: "compound", AST_NODE.CONCAT: "concat", AST_NODE.DECLARE: "declare", AST_NODE.DISTINCT: "distinct",
          AST_NODE.EQUAL: "=", AST_NODE.EXTRACT: "extract", AST_NODE.FORALL: "forall", AST_NODE.IFF: "iff",
          AST_NODE.INTEGER: "integer", AST_NODE.ITE: "ite", AST_NODE.LAND: "land", AST_NODE.LET: "let",
          AST_NODE.LNOT: "lnot", AST_NODE.LOR: "lor", AST_NODE.REFERENCE: "reference", AST_NODE.STRING: "string",
          AST_NODE.SX: "sx", AST_NODE.VARIABLE: "variable", AST_NODE.ZX: "zx"}


class TritonAst:
    """
    Helper class to wrap Triton AstNode and
    use it as an oracle
    """
    def __init__(self, ctx: TritonContext, node: AstNode, node_c, depth, vars, children):
        """
        Initialize IOAst

        :param ctx: Triton contest
        :param node: Triton AstNode to wrap
        """
        self.ctx = ctx
        self.ast = self.ctx.getAstContext()
        self.expr = node  # It needs to be unrolled !!!  # node
        self.size = self.expr.getBitvectorSize()
        self._symvars = vars  # SymVarName -> SymbolicVariable object
        self._parents = set()

        self._node_count = node_c
        self._depth = depth
        self._children = children

    @property
    def parents(self):
        return list(self._parents)

    def is_root(self):
        return not(bool(self.parents))

    @property
    def mapping(self):
        return {x[0]: x[1] for x in zip((chr(x) for x in range(97, 127)), self.symvars)}

    @property
    def sub_map(self):
        return {x[0]: self.ast.variable(x[1]) for x in zip((chr(x) for x in range(97, 127)), self.symvars)}

    @property
    def type(self):
        return AstType(self.expr.getType())

    @property
    def hash(self):
        return self.expr.getHash()

    @property
    def ptr_id(self):
        return hash(self.expr)

    def is_constant_expr(self):
        return len(self.symvars) == 0

    def is_variable(self):
        return self.type == AstType.VARIABLE

    def is_constant(self):
        return self.type == AstType.BV

    @property
    def variable_id(self) -> int:
        if self.is_variable():
            return self.expr.getSymbolicVariable().getId()
        else:
            raise KeyError("not a variable")

    @property
    def var_num(self):
        return len(self._symvars)

    @property
    def pp_str(self):
        return str(self.expr).replace(" & 0xFFFFFFFFFFFFFFFF", "")

    def visit_expr(self) -> Generator['TritonAst', None, None]:
        """ Pre-Order visit """
        def rec(e):
            yield e
            for c in e.get_children():
                yield from rec(c)
        yield from rec(self)

    @property
    def symvars(self):
        return list(self._symvars.values())

    @staticmethod
    def symvar_type(v):
        return SymVarType(v.getType())

    def get_children(self):
        return self._children

    def has_children(self):
        return self.get_children() != []

    def is_leaf(self):
        return self.get_children() == []

    @property
    def node_count(self) -> int:
        return self._node_count

    @property
    def depth(self):
        return self._depth

    @property
    def symbol(self):
        t = self.type
        if t == AstType.BV:
            return str(self.expr)
        elif t == AstType.VARIABLE:
            return str(self.expr)
        else:
            return op_str[t]

    def mk_constant(self, v: int, size: int) -> 'TritonAst':
        return TritonAst(self.ctx, self.ast.bv(v, size), 1, 1, {}, [])

    def mk_variable(self, alias: str, size: int) -> 'TritonAst':
        s = self.ctx.newSymbolicVariable(size, alias)
        s.setAlias(alias)
        ast_s = self.ast.variable(s)
        return TritonAst(self.ctx, ast_s, 1, 1, {s.getName(): s}, [])

    def normalized_str_to_ast(self, s: str) -> Optional['TritonAst']:
        """
        Evaluate expression like "a + b - 1" creating a triton AST
        expression out of it. All variables have to be present in the AST
        :param s: expression to evaluate
        :return: Triton AST node of the expression
        WARNING: the str expr must be obtain through the eval_oracle of the exact same TritonAst (otherwise
        names would be shuffled)
        """
        try:
            e = eval(s, self.sub_map)
        except NameError:
            logging.warning(f"Expression {s} evaluation failed {self.sub_map}")
            return None
        ast = self.make_ast(self.ctx, e)
        return ast

    def to_normalized_str(self) -> str:
        back = {}
        for name, ast_v in self.sub_map:  # Substitute aliases with the normalized names
            sym_v = ast_v.getSymbolicVariable()
            back[name] = sym_v.getAlias()
            sym_v.setAlias(name)
        final_s = self.pp_str  # FIXME: Make a proper iteration of the expression to generate something compliant with lookup tables
        for name, ast_v in self.sub_map:  # Restore true aliases
            sym_v = ast_v.getSymbolicVariable()
            sym_v.setAlias(back[name])
        return final_s

    def update_all(self) -> None:
        def rec(a):
            chs = [rec(x) for x in a.get_children()]
            if chs:  # one of their child might have been updated so update
                a.update()
                return a
            else:
                return a
        rec(self)

    def update(self) -> None:
        chs = self.get_children()
        if chs:
            self._symvars = reduce(lambda acc, x: dict(acc, **x._symvars), chs, {})
            self._depth = max((x.depth for x in chs), default=0)+1
            self._node_count = sum(x.node_count for x in chs)+1

    def update_parents(self, recursive=True):
        for p in self.parents:
            p.update()
        if recursive:
            for p in self.parents:
                p.update_parents(recursive=recursive)

    def set_child(self, i, ast, update_node=True, update_parents=False) -> None:
        if isinstance(i, TritonAst):
            i = {c: i for i, c in enumerate(self.get_children())}[i]  # retrieve number from instance
        if update_node:
            if not self.expr.setChild(i, ast.expr):
                assert False
        self._children[i] = ast  # implicitely unlink previous ast and replace it by the new
        ast._parents.add(self)   # add self as parent of the new ast
        self.update()            # update its own fields
        if update_parents:
            self.update_parents()

    def replace_self(self, repl: 'TritonAst', update_parents=True) -> None:
        if len(self.parents):
            logging.debug("replace self while multiple parents !")
        is_first = True
        for p in self.parents:
            p.set_child(self, repl, update_node=is_first, update_parents=update_parents)
            self._parents.remove(p)  # remove its own parent
            # is_first = False

    @staticmethod
    def make_ast(ctx, exp):
        ptr_map = {}  # hash -> TritonAst

        def rec(e):
            h = hash(e)
            if h in ptr_map:  # if the pointer as already been seen
                return ptr_map[h]  # return it early
            typ = e.getType()
            if typ == AST_NODE.REFERENCE:
                return rec(e.getSymbolicExpression().getAst())
            elif typ == AST_NODE.BV:
                t = TritonAst(ctx, e, 1, 1, {}, [])
            elif typ == AST_NODE.INTEGER:
                t = TritonAst(ctx, e, 1, 1, {}, [])
            elif typ == AST_NODE.VARIABLE:
                symvar = e.getSymbolicVariable()
                name = symvar.getName()
                t = TritonAst(ctx, e, 1, 1, {name: symvar}, [])
            else:
                chs, symvs, node_c = [], {}, 1
                for ast_child in e.getChildren():
                    c = rec(ast_child)
                    symvs.update(c._symvars)
                    node_c += c.node_count
                    chs.append(c)
                depth = max(c.depth for c in chs)+1
                t = TritonAst(ctx, e, node_c, depth, symvs, chs)
                for c in chs:
                    c._parents.add(t)
            ptr_map[hash(e)] = t
            return t
        return rec(exp)

    def duplicate(self) -> 'TritonAst':
        """
        Create a new distinct instance of TritonAst
        """
        new_expr = self.ast.duplicate(self.expr)
        return TritonAst.make_ast(self.ctx, new_expr)

    def random_sampling(self, n: int) -> List[Tuple[Dict[str, int], int]]:
        """
        Generates a random list of I/O samples

        :param n: number of samples to generate
        :return: a list of n (inputs, output) tuples
        """
        samples = []

        for _ in range(n):
            inputs = {k: random.getrandbits(v.getBitvectorSize()) for k, v in self.sub_map.items()}
            output = self.eval_oracle(inputs)
            samples.append((inputs, output))
        return samples

    def eval_oracle(self, args: Dict[str, int]) -> int:
        """
        Oracle corresponding to the wrapped AST

        :param args: a list of input integers which will be used
            as concrete values for the symbolic variables in the
            wrapped AST
        :return: The result computed by the AST
        """
        for v_name, symvar in self.mapping.items():
            self.ctx.setConcreteVariableValue(symvar, args[v_name])
        return self.expr.evaluate()

    def to_z3(self) -> 'z3.z3.ExprRef':
        return self.ast.tritonToZ3(self.expr)

    @staticmethod
    def from_z3(expr) -> 'TritonAst':
        """
        Create an IOAst out of a Z3 expressions
        :param expr: Z3 expression
        :return:
        """
        raise NotImplementedError("")
        # TODO: Do it using new triton version

    def is_semantically_equal(self, other) -> bool:
        cst = self.ast.distinct(self.expr, other.expr)
        return not(self.ctx.isSat(cst))

    @staticmethod
    def dyn_node_count(expr) -> int:
        def rec(e):
            typ = e.getType()
            if typ == AST_NODE.REFERENCE:
                return rec(e.getSymbolicExpression().getAst())
            elif typ == AST_NODE.BV:
                return 1  # prevent counting BV childs as nodes
            else:
                return 1+sum(map(rec, e.getChildren()))
        return rec(expr)

    @staticmethod
    def dyn_depth(expr) -> int:
        def rec(e):
            typ = e.getType()
            if typ == AST_NODE.REFERENCE:
                return rec(e.getSymbolicExpression().getAst())
            elif typ == AST_NODE.BV:
                return 1  # prevent couting BV childs as nodes
            else:
                return 1+max(map(rec, e.getChildren()), default=0)
        return rec(expr)

    @staticmethod
    def _visit_replacement(ast: 'TritonAst') -> Generator[Union['TritonAst', Tuple['TritonAst', bool]], 'TritonAst', None]:
        rep = yield ast  # First (Top-Down yield)

        if rep is not None:  # We should replace this expr
            yield rep, True  # Final (yield)
        else:
            rep_took_place = False
            for i, c in enumerate(ast.get_children()):  # Iterate (and simplify) all childrens
                g = TritonAst._visit_replacement(c)
                recv = None
                while 1:
                    it = g.send(recv)  # forward the new expression to the sub-generator
                    if isinstance(it, tuple):  # is a final yield
                        if it[1]:  # A replacement took place
                            rep_took_place = True
                            ast.set_child(i, it[0])
                        break
                    else:
                        recv = yield it  # re-yield it above (and receive a new expression)
            yield ast, rep_took_place

    def _inplace_replace(self, other: 'TritonAst') -> None:
        logging.debug("Inplace replace")
        self.ctx, self.ast = other.ctx, other.ast
        self.expr = other.expr
        self.size = other.size
        self._symvars = other._symvars
        self._parents = other._parents
        self._node_count, self._depth = other._node_count, other._depth
        self._children = other._children

    def visit_replacement(self, update=True) -> Generator['TritonAst', 'TritonAst', None]:
        """
        Triton AST expression replacement visitor in a Top-Down and then Bottom-Up manner.
        It yield every sub-expression and replace it with the expression received
        throught the send mechanism. While rebuilding the final expression bottom-up
        re-yield modified expression to know if we can replace it even further.
        """
        v = self._visit_replacement(self)
        new_expr_to_send = None
        while True:
            cur_expr = v.send(new_expr_to_send)
            if isinstance(cur_expr, tuple):
                if update:
                    self.update_all()
                break
            else:
                new_expr_to_send = yield cur_expr
                if new_expr_to_send is not None and cur_expr == self:  # Changing root node so fusion fields and return
                    self._inplace_replace(new_expr_to_send)
                    return

    def reassemble(self, dst_reg: str, target_arch: Optional[Union['Arch', str]] = None) -> bytes:
        try:
            from arybo.tools.triton_ import tritonast2arybo
            from arybo.lib.exprs_asm import asm_binary
            if all(TritonAst.symvar_type(x) == SymVarType.REGISTER for x in self.symvars):
                if target_arch is None:
                    m = {ARCH.X86: "x86", ARCH.X86_64: "x86_64", ARCH.ARM32: "arm", ARCH.AARCH64: "aarch64"}
                    arch_name = m[self.ctx.getArchitecture()]
                else:
                    arch_name = target_arch if isinstance(target_arch, str) else target_arch.NAME.lower()
                arybo_expr = tritonast2arybo(self.expr, use_exprs=True, use_esf=False)
                inps = {x.getName(): (x.getAlias(), x.getBitSize()) for x in self.symvars}
                return asm_binary(arybo_expr, (dst_reg, self.size), inps, f"{arch_name}-unknown-unknwon")
            else:
                raise ReassemblyError("Can only reassemble if variable are registers (at the moment)")
        except ImportError:
            raise ReassemblyError("Cannot import arybo, while it is required (pip3 install arybo)")
        except NameError:
            raise ReassemblyError(f"Invalid target architecture '{target_arch}' provided")
        except Exception as e:
            raise ReassemblyError(f"Something went wrong during reassembly: {e}")

    def reassemble_to_insts(self, dst_reg: str, target_arch: Optional[Union[Arch, str]] = None) -> List[Instr]:
        # Note: let all exception being raised above if any
        asm = self.reassemble(dst_reg, target_arch)
        if target_arch is None:
            m = {ARCH.X86: "x86", ARCH.X86_64: "x86_64", ARCH.ARM32: "arm", ARCH.AARCH64: "aarch64"}
            arch = ArchsManager.get_arch(m[self.ctx.getArchitecture()])
        else:
            arch = ArchsManager.get_arch(target_arch) if isinstance(target_arch, str) else target_arch
        return arch.disasm(asm, 0x0)
