# Standard modules
from __future__ import annotations
from enum import IntEnum
import random
from functools import reduce
import logging

# Third-party modules
from triton import TritonContext, AST_NODE, SYMBOLIC, ARCH

# QSynthesis imports
from qsynthesis.types import AstNode, List, Tuple, Generator, Dict, Union, Optional, AstType, SymVarMap, \
                             SymbolicVariable, Char, IOVector, Input, Output


logger = logging.getLogger('qsynthesis')


class ReassemblyError(Exception):
    """
    Wrapping exception for all exceptions that might be raised during the
    reassembly process.
    """
    pass


class SymVarType(IntEnum):
    """
    Enum representing the different types of SymbolicVariables of Triton
    """
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
    Wrapping class on top of Triton AstNode objects. This is the main entity manipulated
    throughout the synthesis process. It provides many utility fonctions on these ASTs
    like :attr:`TritonAst.node_count` holding the number of node of the AST, or
    :meth:`TritonAst.reassembly` that allows reassembling the AST into assembly.


    """

    def __init__(self, ctx: TritonContext, node: AstNode, node_c: int, depth: int, vars: SymVarMap, children: List['TritonAst']) -> 'TritonAst':
        """
        Instanciate a TritonAst with some precomputed fields given in parameters.

        :param ctx: Triton context
        :type ctx: `TritonContext <https://triton.quarkslab.com/documentation/doxygen/py_TritonContext_page.html>`_
        :param node: Triton AstNode to wrap
        :type node: :py:obj:`qsynthesis.types.AstNode`
        :param node_c: Number of nodes contained in the expression
        :type node_c: int
        :param depth: Depth of the expression (depth of the AST)
        :type depth: int
        :param vars: Variables contained in this expression
        :type vars: Dict[str, :py:obj:`qsynthesis.types.SymbolicVariable`]
        :param children: List of children as TritonAst instances
        :type children: List[TritonAst]

        .. warning:: This class is not meant to be instanciated directly. It must be instanciated
             trough the :meth:`~TritonAst.make_ast` method.
        """
        self.ctx = ctx
        self.ast = self.ctx.getAstContext()
        self.expr = node  # It needs to be unrolled !!!
        self.size = self.expr.getBitvectorSize()
        self._symvars = vars  # SymVarName -> SymbolicVariable object  e.g: {'SymVar_1': rdi, 'SymVar_2': rsi}
        self._mapping = {chr(x[0]): x[1] for x in zip(range(97, 127), self._symvars.values())}  # {'a': SymVar, 'b': SymVar}
        self._parents = set()

        self._node_count = node_c
        self._depth = depth
        self._children = children

    @property
    def parents(self) -> List['TritonAst']:
        """
        Return the list of parents of a given AST. An AST is meant to have only
        ONE parent but Triton share common expression with multiple parents (wihtin
        the same expression)

        :rtype: List[TritonAst]
        """
        return list(self._parents)

    @property
    def mapping(self) -> Dict[Char, SymbolicVariable]:
        """
        Mapping a placeholder character ('a', 'b', 'c' ..) to all the SymbolicVariable
        of the object.

        :rtype: Dict[:py:obj:`qsynthesis.types.Char`, :py:obj:`SymbolicVariable`]
        """
        return self._mapping

    @mapping.setter
    def mapping(self, value: Dict[Char, SymbolicVariable]) -> None:
        """
        Set the given mapping of placeholder to their SymbolicVariable in the object.
        """
        self._mapping = value

    @property
    def sub_map(self) -> Dict[Char, AstNode]:
        """
        Similar to mapping but map a placeholder character ('a', 'b', 'c' ..) to
        the AstNode counterpart of SymbolicVariables.

        :rtype: Dict[:py:obj:`qsynthesis.types.Char`, :py:obj:`qsynthesis.types.AstNode`]
        """
        return {k: self.ast.variable(v) for k, v in self.mapping.items()}

    @property
    def type(self) -> AstType:
        """
        Returns the type of current AstNode object. The
        type is identical to the AST_NODE enum of Triton.

        :rtype: :py:obj:`qsynthesis.types.AstType`
        """
        return AstType(self.expr.getType())

    @property
    def hash(self) -> int:
        """
        Returns the Triton hash of the AstNode. This hash is meant to be unique
        for all AstNode, but is also meant to be similar to commutative expressions.

        :rtype: int
        """
        return self.expr.getHash()

    @property
    def ptr_id(self) -> int:
        """
        Returns the hash of the AstNode object. This attribute is meant to differentiate
        to different python object have the exact same AST structure.

        :rtype: int
        """
        return hash(self.expr)

    def is_constant_expr(self) -> bool:
        """Returns whether the AST expression is constant or not (namely does not
        have any symbolic variables in it)."""
        return len(self.symvars) == 0

    def is_variable(self) -> bool:
        """Returns of the TritonAst is a variable node."""
        return self.type == AstType.VARIABLE

    def is_constant(self) -> bool:
        """Returns True if the type of the object is Bitvector (namely constant)"""
        return self.type == AstType.BV

    @property
    def variable_id(self) -> int:
        """
        Get the Triton unique id for a variable.

        :raises: KeyError
        """
        if self.is_variable():
            return self.expr.getSymbolicVariable().getId()
        else:
            raise KeyError("not a variable")

    @property
    def var_num(self) -> int:
        """Returns the number of different symbolic variables of the expression

        :rtype: int
        """
        return len(self._symvars)

    @property
    def pp_str(self) -> str:
        """Hacky function that strips masks used in the AST_REPRESENTATION.PYTHON
        of Triton.

        :rtype: str
        """
        return str(self.expr).replace(" & 0xFFFFFFFFFFFFFFFF", "").replace(" & 0xffffffffffffffff", "")


    def visit_expr(self) -> Generator['TritonAst', None, None]:
        """ Pre-Order visit of all the sub-AstNode"""
        def rec(e):
            yield e
            for c in e.get_children():
                yield from rec(c)
        yield from rec(self)

    @property
    def symvars(self) -> List[SymbolicVariable]:
        """Returns the list of SymbolicVariable object of the current object

        :rtype: List[:py:obj:`qsynthesis.types.SymbolicVariable`]
        """
        return list(self._symvars.values())

    @staticmethod
    def symvar_type(v: SymbolicVariable) -> SymVarType:
        """
        Static method returning the type of a given symbolic variable object

        :param v: symbolic variable object
        :type v: :py:obj:`qsynthesis.types.SymbolicVariable`
        :return: Type of the symbolic variables
        :rtype: :py:obj:`qsynthesis.types.SymVarType`
        """
        return SymVarType(v.getType())

    def get_children(self) -> List['TritonAst']:
        """Return the list of children TritonAst"""
        return self._children

    def has_children(self) -> bool:
        """True whether the object has children or not"""
        return self.get_children() != []

    def is_root(self) -> bool:
        """
        Return True whether the object is a root node (namely does not have
        any parents).
        """
        return not(bool(self.parents))

    def is_leaf(self) -> bool:
        """True if the AST has no children"""
        return self.get_children() == []

    @property
    def node_count(self) -> int:
        """Pre-computed O(1) count of the number of node contained in this AST.

        :rtype: int
        """
        return self._node_count

    @property
    def depth(self) -> int:
        """Pre-computed depth of the AST *(complexity O(1))*

        :rtype: int
        """
        return self._depth

    @property
    def symbol(self) -> str:
        """Returns the symbol of the current AstNode, operator if binary expression
        variable name if variable or constant value if constant.

        :rtype: str
        """
        t = self.type
        if t in [AstType.BV, AstType.VARIABLE]:
            return str(self.expr)
        elif t == AstType.INTEGER:
            return str(self.expr.getInteger())
        else:
            return op_str[t]

    def mk_constant(self, v: int, size: int) -> 'TritonAst':
        """Create a new constant as a TritonAst (holding a Triton bv object)."""
        return TritonAst(self.ctx, self.ast.bv(v, size), 1, 1, {}, [])

    def mk_variable(self, alias: str, size: int) -> 'TritonAst':
        """
        Create a new variable node as a TritonAst (holding a Triton variable object).
        The variable is created in the TritonContext of the current object.
        """
        s = self.ctx.newSymbolicVariable(size, alias)
        s.setAlias(alias)
        ast_s = self.ast.variable(s)
        return TritonAst(self.ctx, ast_s, 1, 1, {s.getName(): s}, [])

    def normalized_str_to_ast(self, s: str) -> 'TritonAst':
        """
        Evaluate expression like "a + b - 1" creating a triton AST expression out
        of it. All variables have to be present in the AST.

        :param s: expression to evaluate
        :return: Triton AST node of the expression

        .. warning:: the str expr must be obtained through the eval_oracle of the
                     exact same TritonAst (otherwise names would be shuffled)
        """
        try:
            e = eval(s, self.sub_map)
        except NameError as e:
            logger.warning(f"Expression {s} evaluation failed {self.sub_map}")
            raise e
        except TypeError as exc:
            logger.error(f"Type error when evaluating: {s}")
            # Try mangling expressions
            nast = self._try_mangle_ast(s, exc)
            if nast:
                return nast
            else:
                raise exc
        ast = self.make_ast(self.ctx, e)
        return ast

    def _try_mangle_ast(self, s: str, e: Exception) -> Optional['TritonAst']:
        """
        Hacky function to mangle expression like 'a**2' generated by sympy.
        Triton AST and SMT does not support pow operators thus they have to
        be unrolled. This function tries to do it and if succeed try to generate
        an AST out of it.

        :param s: expression to evaluate
        :return: Triton AST node of the expression
        """
        logger.warning(f"Try mangling {s}: {str(e)}")
        import re
        mod = False
        s2 = s
        if str(e) == "unsupported operand type(s) for ** or pow(): 'AstNode' and 'int'":
            for c, v in re.findall("([^\(*]+)\*\*(\d+)", s):
                sub_c = c
                trail = ""
                if c.count("(") != c.count(")") and c.endswith(")"):
                    sub_c = sub_c[:-1]
                    trail = ")"
                # logger.info(f"Found instance {c} with {v}")
                mod = True
                s2 = re.sub(re.escape(c+"**"+v), '*'.join([f"({sub_c})"]*int(v))+trail, s2)
            logger.warning(f"Mangle expr {s} to {s2}")
        if mod and s2 != s:
            return self.normalized_str_to_ast(s2)
        else:
            return None

    def to_normalized_str(self) -> str:
        """
        Normalize the AST (replace variables by placeholder 'a', 'b' ..) and return
        it as a string.
        """
        back = {}
        for name, ast_v in self.sub_map.items():  # Substitute aliases with the normalized names
            sym_v = ast_v.getSymbolicVariable()
            back[name] = sym_v.getAlias()
            sym_v.setAlias(name)
        final_s = self.pp_str  # FIXME: Make a proper iteration of the expression to generate something compliant with lookup tables
        for name, ast_v in self.sub_map.items():  # Restore true aliases
            sym_v = ast_v.getSymbolicVariable()
            sym_v.setAlias(back[name])
        return final_s

    def update_all(self) -> None:
        """
        Update all children recursively of the current object. Fields being updated
        are node_count, depth and symvars. This might be used when some of the AST
        has been rewritten. All pre-computed values are then 'dirty' and have to be
        updated.
        """
        def rec(a):
            chs = [rec(x) for x in a.get_children()]
            if chs:  # one of their child might have been updated so update
                a.update()
                return a
            else:
                return a
        rec(self)

    def update(self) -> None:
        """
        Update the current AST node, with information of its directly children.
        Information of children are thus considered genuine.
        """
        chs = self.get_children()
        if chs:
            self._symvars = reduce(lambda acc, x: dict(acc, **x._symvars), chs, {})
            self._depth = max((x.depth for x in chs), default=0)+1
            self._node_count = sum(x.node_count for x in chs)+1

    def update_parents(self, recursive: bool = True) -> None:
        """
        Update the parent of the current TritonAst. `recursive` indicates
        if it has to be performed recrusively. If so the complexity of the
        operation O(depth).
        :param recursive: whether to apply it recursively on parents
        """
        for p in self.parents:
            p.update()
        if recursive:
            for p in self.parents:
                p.update_parents(recursive=recursive)

    def set_child(self, i: int, ast: 'TritonAst', update_node: bool = True, update_parents: bool = False) -> None:
        """
        Replace the ith child of the current TritonAst with new given ast object. Optional
        parameters indicates if inner fields of the object and its parent have to be updated.

        :param i: index of the child to replace
        :param ast: TritonAst to be used as replacement of the child
        :param update_node: whether to update internal field of the current node (node_count, depth, symvars)
        :param update_parents: whether to update parents
        """
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

    def replace_self(self, repl: 'TritonAst', update_parents: bool = True) -> None:
        """
        Replace the current object by the given TritonAst. This function thus replace
        parents by replace the child that correspond to the current object by the replacement.
        :param repl: TritonAst used to replace the current object
        :param update_parents: whether to update parents or not
        """
        if len(self.parents):
            logger.debug("replace self while multiple parents !")
        is_first = True
        for p in self.parents:
            p.set_child(self, repl, update_node=is_first, update_parents=update_parents)
            self._parents.remove(p)  # remove its own parent
            # is_first = False

    @staticmethod
    def make_ast(ctx: TritonContext, exp: AstNode) -> 'TritonAst':
        """
        Main staticmethod meant to create all TritonAst object. This method iterates
        all the given expression ``expr`` recursively to create TritonAst's
        all the way down and pre-computing along the way the important fields like
        node_count, depth and symvars.

        :param ctx: Triton context on which to work on
        :param exp: AstNode object to iterate
        :returns: TritonAst instance wrapping the exp object
        """
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

    def random_sampling(self, n: int) -> IOVector:
        """
        Generates a random list of I/O samples pair.

        :param n: number of samples to generate
        :return: a list of n (inputs, output) tuples
        :rtype: :py:obj:`qsynthesis.types.IOVector`
        """
        samples = []

        for _ in range(n):
            inputs = {k: random.getrandbits(v.getBitvectorSize()) for k, v in self.sub_map.items()}
            output = self.eval_oracle(inputs)
            samples.append((inputs, output))
        return samples

    def eval_oracle(self, inp: Input) -> Output:
        """
        Oracle corresponding to the wrapped AST. It takes a valuation for all its
        symbolic variables and as an oracle returns the associated output.

        :param inp: a mapping of variable to a given input value which will be used
                    as concrete values for the symbolic variables in the wrapped AST
        :type inp: :py:obj:`qsynthesis.types.Input`
        :return: The result computed by means of evaluating the AST
        :rtype: :py:obj:`qsynthesis.types.Output`
        """
        for v_name, symvar in self.mapping.items():
            self.ctx.setConcreteVariableValue(symvar, inp[v_name])
        return self.expr.evaluate()

    def compare_behavior(self, ast: TritonAst, inps: List[Input]) -> int:
        """
        Compare the current expression with the one given in parameter wrt theirs
        behavior on the given set of inputs. The comparison returns -1 if not applicable
        (as involving different variables, 0 if different and 1 if equal.

        :param ast: other ast to compare against
        :param inps: Set of inputs to use for evaluation
        :return: -1 if not applicable, 0 if different and 1 if equal
        :rtype: int
        """
        if set(x.getAlias() for x in self.mapping.values()).symmetric_difference(set(x.getAlias() for x in ast.mapping.values())):
            return -1  # Some variables are different
        backup = self.mapping
        print([self.eval_oracle(i) for i in inps])
        rev = {v.getAlias(): k for k, v in ast.mapping.items()}  # Map name -> pld   e.g: {'rsi': 'a', 'rdi': 'b'}
        self.mapping = {rev[vs.getAlias()]: vs for pld, vs in self.mapping.items()}  # keep own vars (vs) but remap on other ast identifier
        print(backup, self.mapping)
        tmp1 = [self.eval_oracle(i) for i in inps]
        tmp2 = [ast.eval_oracle(i) for i in inps]
        print(tmp1, "\n", tmp2)
        res = tmp1 == tmp2
        self.mapping = backup  # Restore backup submap
        return res

    def to_z3(self) -> 'z3.z3.ExprRef':
        """Returns the Z3 expression associated with the Triton AST expression"""
        return self.ast.tritonToZ3(self.expr)

    @staticmethod
    def from_z3(ctx: TritonContext, expr: 'z3.z3.ExprRef') -> 'TritonAst':
        """
        Create a TritonAst out of a Z3 expressions

        :param ctx: Triton Context in which to create the expression
        :param expr: Z3 expression
        :return: TritonAst

        .. warning:: This function is mostly untested !
        """
        astctx = ctx.getAstContext()
        ast = astctx.z3ToTriton(expr)
        return TritonAst.make_ast(ctx, ast)

    def is_semantically_equal(self, other: 'TritonAst') -> bool:
        """
        Allows checking if the current AST is semantically equal to the one provided.

        :param other: TritonAst on which to test against
        :returns: bool -- True if both ASTs are semantically equals
        """
        cst = self.ast.distinct(self.expr, other.expr)
        return not(self.ctx.isSat(cst))

    @staticmethod
    def dyn_node_count(expr: AstNode) -> int:
        """
        Returns the effective count of node of the expression by iterating the
        AstNode object recursively. The complexity is O(N) with N the number of node.

        :param expr: AstNode to iterate
        :type expr: :py:obj:`qsynthesis.types.AstNode`
        :returns: Number of nodes in the AST.

        .. note:: The way of counting nodes is different from the number of nodes of
                  Triton for which bitvector values are composed of 3 nodes. We count
                  them as one.
        """
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
    def dyn_depth(expr: AstNode) -> int:
        """
        Returns the effective depth of the node of the expression by iterating the
        AstNode object recursively. The complexity is O(N) with N the depth of the AST.

        :param expr: AstNode to iterate
        :type expr: :py:obj:`qsynthesis.types.AstNode`
        :returns: AST depth
        """
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
        logger.debug("Inplace replace")
        self.ctx, self.ast = other.ctx, other.ast
        self.expr = other.expr
        self.size = other.size
        self._symvars = other._symvars
        self._parents = other._parents
        self._node_count, self._depth = other._node_count, other._depth
        self._children = other._children

    def visit_replacement(self, update: bool = True) -> Generator['TritonAst', 'TritonAst', None]:
        """
        Triton AST expression replacement visitor in a Top-Down manner. It yields
        every sub-expression and replace it with the expression received throught
        the send mechanism.

        :param update: whether to update each node after having been replaced
        :return: generator of TritonAst, which for each AST yielded wait to
                 receive either None meaning the it should not be replaced or a new
                 TritonAst to be put in replacement.
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

    def reassemble(self, dst_reg: str, target_arch: Optional[str] = None) -> bytes:
        """
        Reassemble the TritonAst in assembly. ``dst_reg`` is the destination register of the
        result of the computation of the AST. Parameter ``target_arch`` is either a QtraceDB
        architecture object or the string identifier of the architecture as defined by the
        LLVM architecture triple: https://llvm.org/doxygen/classllvm_1_1Triple.html#a547abd13f7a3c063aa72c8192a868154
        If no architecture is provided, use the same than the one that the AST.

        :param dst_reg: destination register as lowercase string
        :param target_arch: target architecture in which to reassemble the AST
        :returns: bytes of the AST reassembled in the given architecture
        :raises: ReassemblyError

        .. warning:: This method requires the ``arybo`` library that can be installed with
           (pip3 install arybo).
        """
        def my_asm_binary(arybo_expr, dst_regs, inps, target):
            # FIXME: ``asm_binary`` of arybo is broken as it used the function
            # ObjectFileRef of llvmlite which is itself broken. Up until a fix
            # is pushed I have to reimplement asm_binary and retrieve bytes with
            # lief. cf: https://github.com/numba/llvmlite/issues/632
            mod = asm_module(arybo_expr, dst_regs, inps, target)
            M = llvm.parse_assembly(str(mod))
            M.verify()
            target = llvm_get_target(target)
            machine = target.create_target_machine()
            obj_bin = machine.emit_object(M)
            p = lief.parse(obj_bin)
            return bytes(p.get_section('.text').content)
        try:
            import lief
            from arybo.tools.triton_ import tritonast2arybo
            from arybo.lib.exprs_asm import asm_binary, asm_module, llvm, llvm_get_target
            if all(TritonAst.symvar_type(x) == SymVarType.REGISTER for x in self.symvars):
                if target_arch is None:
                    m = {ARCH.X86: "x86", ARCH.X86_64: "x86_64", ARCH.ARM32: "arm", ARCH.AARCH64: "aarch64"}
                    arch_name = m[self.ctx.getArchitecture()]
                else:
                    arch_name = target_arch.lower()
                arybo_expr = tritonast2arybo(self.expr, use_exprs=True, use_esf=False)
                inps = {x.getName(): (x.getAlias(), x.getBitSize()) for x in self.symvars}
                return my_asm_binary(arybo_expr, (dst_reg, self.size), inps, f"{arch_name}-unknown-unknwon")
            else:
                raise ReassemblyError("Can only reassemble if variable are registers (at the moment)")
        except ImportError as e:
            raise ReassemblyError(f"Cannot import arybo, while it is required (pip3 install arybo): {e}")
        except NameError:
            raise ReassemblyError(f"Invalid target architecture '{target_arch}' provided")
        except Exception as e:
            raise ReassemblyError(f"Something went wrong during reassembly: {e}")

    def make_graph(self) -> 'Graph':
        """
        Generate a graph object representing the AST
        as a graph_tool object.

        .. warning:: This method requires the ``graph_tool`` python library that can be installed
           by following `https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions`
        """
        from graph_tool import Graph
        graph = Graph(directed=True)

        graph.vp['sym'] = graph.new_vertex_property('string')
        graph.vp['vmap'] = graph.new_vertex_property('int64_t')
        n = graph.add_vertex(1)

        worklist = [(None, (self, n))]
        while worklist:
            na, (b, nb) = worklist.pop(0)

            graph.vp['sym'][nb] = b.symbol
            graph.vp['vmap'][nb] = id(b)

            if na is not None:
                graph.add_edge(na, nb)

            for c in b.get_children():
                nc = graph.add_vertex(1)
                worklist.append((nb, (c, nc)))
        return graph
