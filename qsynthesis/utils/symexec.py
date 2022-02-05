# built-in libs
from __future__ import annotations
from typing import List, Set, Iterable, Union, Optional

# third-party libs
from triton import ARCH, CALLBACK, MODE, MemoryAccess, Instruction, AST_REPRESENTATION, OPERAND
from triton import TritonContext

# qsynthesis deps
from qsynthesis.tritonast import TritonAst
from qsynthesis.types import SymbolicExpression, Register, Addr, ByteSize, SymbolicVariable


class SimpleSymExec:
    """
    Helper class used to process a sequential bunch of instructions using Triton
    and mark as symbolic all memory and registers reads that occoured before a
    write to that same register or memory address
    """

    def __init__(self, arch: Union[ARCH, str]):
        """
        Initialize symbolic execution

        :param arch: Triton architecture identifier or string of it
        :type: `ARCH <https://triton.quarkslab.com/documentation/doxygen/py_ARCH_page.html>`_
        """
        arch = getattr(ARCH, arch.upper()) if isinstance(arch, str) else arch
        self.ctx = TritonContext(arch)
        self.ctx.setMode(MODE.ALIGNED_MEMORY, True)
        self.ctx.setMode(MODE.AST_OPTIMIZATIONS, True)
        self.ctx.setMode(MODE.CONSTANT_FOLDING, True)
        self.ctx.setMode(MODE.ONLY_ON_SYMBOLIZED, True)
        self.ctx.setAstRepresentationMode(AST_REPRESENTATION.PYTHON)

        self.mem_addr_seen = set()  # All addresses read / written during execution
        self.mem_symvars = []       # All memory symbolic variables
        self.reg_id_seen = set()    # Triton Id of registers written directly or initialized with a value
        self.reg_symvars = []       # All registers symbolic variables

        self._capturing = False

        # Add callbacks
        self.ctx.addCallback(CALLBACK.SET_CONCRETE_MEMORY_VALUE, self._mem_write_callback)
        self.ctx.addCallback(CALLBACK.GET_CONCRETE_MEMORY_VALUE, self._mem_read_callback)
        self.ctx.addCallback(CALLBACK.SET_CONCRETE_REGISTER_VALUE, self._reg_write_callback)
        self.ctx.addCallback(CALLBACK.GET_CONCRETE_REGISTER_VALUE, self._reg_read_callback)

        self.cur_inst = None
        self.inst_id = 0
        self._expr_id = 0
        self._inst_cbs = []

    @property
    def expr_id(self) -> int:
        """
        Within a single instruction, used to represent an expression id.
        Programmed as an auto-increment variable upon each read.

        :rtype: int
        """
        self._expr_id += 1
        return self._expr_id - 1

    @expr_id.setter
    def expr_id(self, value: int) -> None:
        """Set the given expr_id"""
        self._expr_id = value

    @property
    def arch(self) -> ARCH:
        """Return the triton architecture identifier of the current context

        :rtype: `ARCH <https://triton.quarkslab.com/documentation/doxygen/py_ARCH_page.html>`_
        """
        return self.ctx.getArchitecture()

    @property
    def flags_reg(self) -> Register:
        """
        Portable function the get the flag register accross all Triton
        supported architectures.

        :rtype: `Register <https://triton.quarkslab.com/documentation/doxygen/py_Register_page.html>`_
        """
        _mapper = {ARCH.X86: "eflags", ARCH.X86_64: "eflags", ARCH.ARM32: "cpsr", ARCH.AARCH64: "spsr"}  # Not really true for spsr
        return getattr(self.ctx.registers, _mapper[self.arch])

    @property
    def ins_ptr_reg(self) -> Register:
        """
        Portable function get get the instruction pointer register depending
        on the current architecture.

        :rtype: `Register <https://triton.quarkslab.com/documentation/doxygen/py_Register_page.html>`_
        """
        _mapper = {ARCH.X86: "eip", ARCH.X86_64: "rip", ARCH.ARM32: "pc", ARCH.AARCH64: "pc"}
        return getattr(self.ctx.registers, _mapper[self.arch])

    @property
    def current_address(self) -> Addr:
        """Return the address of the current instruction.

        :rtype: :py:obj:`qsynthesis.types.Addr`
        """
        return self.cur_inst.getAddress()

    def _turn_on(self) -> None:
        """Turn on capturing"""
        self._capturing = True

    def _turn_off(self) -> None:
        """Turn off capturing"""
        self._capturing = False

    def _mem_write_callback(self, _: TritonContext, ma: MemoryAccess, __: int) -> None:
        """
        Callback called by Triton upon each memory write. It is used to update a map
        of all addresses written.
        """
        if not self._capturing:
            return

        # Add bytes to set of written bytes
        self.mem_addr_seen |= self._memacc_to_all_addr(ma)

    def _mem_read_callback(self, ctx: TritonContext, ma: MemoryAccess) -> None:
        """
        Callback called by Triton upon each memory read. It is used to dynamically
        symbolized memory cells accessed that have not been seen before.
        """
        if not self._capturing:
            return

        # Retrieve all addresses of a given mem access
        addrs = self._memacc_to_all_addr(ma)

        # Get addrs which have not been written to and are not
        # yet been symbolized (eg. are not yet parameters)
        new_addrs = addrs - self.mem_addr_seen

        # Stop if we don't have any new address. This is also needed to stop
        # recursion since convertMemoryToSymbolicVariable will trigger again this callback
        if not new_addrs:
            return

        # Update mem_operands
        self.mem_addr_seen |= new_addrs

        # Symbolize all the memory cells that have not yet been seen
        # Coalesce adjacent bytes and create memory accesses' symvars
        for ma in self._coalesce_bytes_to_mas(new_addrs):
            symvar = self.symbolize_memory(ma)
            self.mem_symvars.append(symvar)

    def _reg_write_callback(self, ctx: TritonContext, reg: Register, _: int) -> None:
        """
        Callback called by Triton on each register write.
        """
        if not self._capturing:
            return

        # Add parent register to set of written registers)
        parent_reg = ctx.getParentRegister(reg)
        self.reg_id_seen.add(parent_reg.getId())

    def _reg_read_callback(self, ctx: TritonContext, reg: Register) -> None:
        """
        Callback called by Triton on each register read.
        """
        if not self._capturing:
            return

        parent_reg = ctx.getParentRegister(reg)
        parent_reg_id = parent_reg.getId()

        # Skip reg if it has already been written or is already an operand
        if parent_reg_id in self.reg_id_seen:
            return

        # Ignore flags and pc registers (and make sure we know it!)
        if parent_reg in [self.flags_reg, self.ins_ptr_reg]:
            return

        # Ignore registers which are immutables
        if not reg.isMutable():
            return

        # Symbolize the (full) register
        self.symbolize_register(parent_reg, 0)

    def get_register_ast(self, reg_name: Union[str, Register]) -> TritonAst:
        """
        Get the TritonAst associated with the given register. The register
        can either be a string or a triton register object.

        :param reg_name: register name or triton register
        :type reg_name: Union[str, `Register <https://triton.quarkslab.com/documentation/doxygen/py_Register_page.html>`_]
        :returns: the TritonAst associated to that register
        """
        reg = getattr(self.ctx.registers, reg_name.lower()) if isinstance(reg_name, str) else reg_name
        reg_se = self.ctx.getSymbolicRegister(reg)
        actx = self.ctx.getAstContext()
        if reg_se is None:
            e = actx.bv(self.ctx.getConcreteRegisterValue(reg), reg.getBitSize())
        else:
            e = actx.unroll(reg_se.getAst())
        return TritonAst.make_ast(self.ctx, e)

    def get_memory_ast(self, addr: Addr, size: ByteSize) -> TritonAst:
        """
        Get the TritonAst associated with the given address and size.

        :param addr: address of which to create the AST
        :type addr: :py:obj:`qsynthesis.types.Addr`
        :param size: Size of the read in memory (in bytes)
        :type size: :py:obj:`qsynthesis.types.ByteSize`
        :returns: the TritonAst of the memory content
        :rtype: TritonAst
        """
        ast = self.ctx.getMemoryAst(MemoryAccess(addr, size))
        actx = self.ctx.getAstContext()
        return TritonAst.make_ast(self.ctx, actx.unroll(ast))

    def get_operand_ast(self, op_num: int, inst: Optional[Instruction] = None) -> TritonAst:
        """
        Get the TritonAst of the of the ith operand. The instruction can be provided
        as an optional parameter. If not provided it takes the current instruction
        having been processed.

        :param op_num: operand number (starting at 0)
        :param inst: Triton Instruction of which to get the operand
        :type inst: Optional[`Instruction <https://triton.quarkslab.com/documentation/doxygen/py_Instruction_page.html>`_]
        :returns: the TritonAst of the operand
        :rtype: TritonAst
        """
        inst = self.cur_inst if inst is None else inst
        op = inst.getOperands()[op_num]
        actx = self.ctx.getAstContext()
        t = op.getType()
        if t == OPERAND.IMM:
            e = actx.bv(op.getValue(), op.getBitSize())
        elif t == OPERAND.REG:
            e = self.ctx.getRegisterAst(op)
        elif t == OPERAND.MEM:
            e = self.ctx.getMemoryAst(op)
        else:
            assert False
        return TritonAst.make_ast(self.ctx, actx.unroll(e))

    def get_register_symbolic_expression(self, reg_name: Union[str, Register]) -> SymbolicExpression:
        """
        Get the current SymbolicExpression (triton object) of the register given in parameter.

        :param reg_name: register name, or Register object
        :type reg_name: Union[str, `Register <https://triton.quarkslab.com/documentation/doxygen/py_Register_page.html>`_]
        :returns: current symbolic expression of the register
        :rtype: `SymbolicExpression <https://triton.quarkslab.com/documentation/doxygen/py_SymbolicExpression_page.html>`_
        """
        reg = getattr(self.ctx.registers, reg_name.lower()) if isinstance(reg_name, str) else reg_name
        return self.ctx.getSymbolicRegister(reg)

    def get_memory_symbolic_expression(self, addr: Addr, size: ByteSize) -> SymbolicExpression:
        """
        Get the current SymbolicExpression of the given address. As no symbolic expression are
        assigned to single memory bytes. The function creates a new symbolic expression representing
        the addr+size expression.

        :param addr: address in memory
        :type addr: :py:obj:`qsynthesis.types.Addr`
        :param size: size in bytes of the memory read
        :type ByteSize: :py:obj:`qsynthesis.types.ByteSize`
        :returns: symbolic expression representing the memory content value
        :rtype: `SymbolicExpression <https://triton.quarkslab.com/documentation/doxygen/py_SymbolicExpression_page.html>`_
        """
        ast = self.ctx.getMemoryAst(MemoryAccess(addr, size))
        sym = self.ctx.newSymbolicExpression(ast)
        sym.setComment(self._fmt_comment())
        return sym

    def get_operand_symbolic_expression(self, op_num: int) -> SymbolicExpression:
        """
        Get the symbolic expression of the ith operand of the current instruction being
        processed by the symbolic executor. Depending on the type of the operand
        calls :meth:`SimpleSymExec.get_register_symbolic_expression`, or
        :meth:`SimpleSymExec.get_memory_symbolic_expression`. For constants create
        a new symbolic expression.

        :param op_num: operand number
        :type op_num: int
        :returns: the symbolic expression of the operand
        :rtype: `SymbolicExpression <https://triton.quarkslab.com/documentation/doxygen/py_SymbolicExpression_page.html>`_
        """
        op = self.cur_inst.getOperands()[op_num]
        t = op.getType()
        if t == OPERAND.IMM:
            ast = self.ctx.getAstContext().bv(op.getValue(), op.getBitSize())
            sym = self.ctx.newSymbolicExpression(ast)
            sym.setComment(self._fmt_comment())
        elif t == OPERAND.REG:
            sym = self.get_register_symbolic_expression(op)
        elif t == OPERAND.MEM:
            sym = self.get_memory_symbolic_expression(op.getAddress(), op.getSize())
        else:
            assert False
        return sym

    def symbolize_register(self, reg: Register, value: int) -> SymbolicVariable:
        """
        Symbolize the given register with the associated concrete value.

        :param reg: Register to symbolize
        :type reg: `Register <https://triton.quarkslab.com/documentation/doxygen/py_Register_page.html>`_
        :param value: Concrete value to assign the register (required for soundness)
        :type value: int
        :returns: the symbolic variable created for the register
        :rtype: `SymbolicVariable <https://triton.quarkslab.com/documentation/doxygen/py_SymbolicVariable_page.html>`_
        """
        self.reg_id_seen.add(reg.getId())
        symvar = self.ctx.symbolizeRegister(reg, reg.getName())
        #symvar.setComment(comment)

        # Set comment on the register reference
        sreg = self.ctx.getSymbolicRegister(reg)
        sreg.setComment(self._fmt_comment())

        # We also set the symbolic var to the actual value of the register
        self.ctx.setConcreteVariableValue(symvar, value)
        self.reg_symvars.append(symvar)
        return symvar

    def symbolize_memory(self, mem: MemoryAccess) -> SymbolicVariable:
        """
        Symbolize the given triton MemoryAccess. Into a new SymbolicVariable.

        :param mem: memory access to symbolize
        :type mem: `MemAccess <https://triton.quarkslab.com/documentation/doxygen/py_MemoryAccess_page.html>`_
        :returns: symbolic variable representing the content
        :rtype: `SymbolicVariable <https://triton.quarkslab.com/documentation/doxygen/py_SymbolicVariable_page.html>`_
        """
        # The issue here is that we need to assign a specific comment
        # on symbolic expression to attach them to a specific instruction
        # and being able to perform slicing afterward. But there is no
        # direct manner to get symbolic expressions created by the
        # symbolizeMemory

        # symbolize the MemAccess and obtain a SymbolicVariable object
        alias = f"mem_{mem.getAddress():#x}_{mem.getSize()}_{self.inst_id}"
        symvar = self.ctx.symbolizeMemory(mem, alias)
        symvar.setComment(self._fmt_comment())

        # Iterate each address of the MemAccess to obtain the SymbolicExpression
        # of the address. Put for each of them the right comment
        addr = mem.getAddress()
        end = addr + mem.getSize()
        cur_mem_exp = None
        while addr < end:
            cur_mem_exp = self.ctx.getSymbolicMemory(addr)
            cur_mem_exp.setComment(self._fmt_comment())
            addr += 1

        # Each SymbolicExpression have a uniq Id. The way they are created
        # (I heuristically know that the one before the last cur_mem_exp)
        # is the expression concating all of them thus also add a comment on it
        var_exp = self.ctx.getSymbolicExpression(cur_mem_exp.getId()-1)
        var_exp.setComment(self._fmt_comment())
        return symvar

    def initialize_register(self, reg: Union[str, Register], value: int) -> None:
        """
        Initialize a register by giving it an initial value. Register can
        either be a string or a Register object.

        :param reg: reg name string or register object
        :type reg: Union[str, `Register <https://triton.quarkslab.com/documentation/doxygen/py_Register_page.html>`_]
        :param value: integer value of the register
        """
        reg = getattr(self.ctx.registers, reg.lower()) if isinstance(reg, str) else reg
        self.reg_id_seen.add(reg.getId())
        self.ctx.setConcreteRegisterValue(reg, value)

    def disassemble(self, opcode: bytes, addr: Optional[Addr] = None) -> Instruction:
        """
        Disassemble a given opcode using Triton. Returns a Triton Instruction object.
        The Instruction is not been symbolically executed but the internal `cur_inst`
        attribute is set. The address is optional.
        If not provided the current instruction pointer in the internal context is used.

        :param opcode: bytes of the instruction
        :type opcode: bytes
        :param addr: address of the instruction
        :type addr: Optional[:py:obj:`qsynthesis.types.Addr`]
        :returns: Triton Instruction object
        :rtype: `Instruction <https://triton.quarkslab.com/documentation/doxygen/py_Instruction_page.html>`_
        """
        inst = Instruction(addr, opcode[:16]) if addr is not None else Instruction(opcode[:16])
        self.ctx.disassembly(inst)
        self.cur_inst = inst  # Set it if require to query get_operand_symbolic_expression
        return inst

    def execute_blob(self, data: bytes, addr: Addr) -> bool:
        """
        Execute instructions of the blob while the program counter
        remains in the blob. Consume the whole data and execute it
        symbolically.

        .. warning: This method writes payload into triton context.

        :param data: bytes of instructions to execute
        :type data: bytes
        :param addr:  address of the first instruction
        :type addr: :py:obj:`qsynthesis.types.Addr`
        :returns: True if execution of all instructions succeeded
        :rtype: bool
        """
        self.ctx.setConcreteMemoryAreaValue(addr, data)  # set concrete data in memory
        self.ctx.setConcreteRegisterValue(self.ins_ptr_reg, addr) # set program counter value
        pc = addr

        while addr <= pc < (addr + len(data)):  # while we remain in the blob
            opcode = self.ctx.getConcreteMemoryAreaValue(pc, 16) 
            if not self.execute(opcode, pc):
                return False
            pc =  self.ctx.getConcreteRegisterValue(self.ins_ptr_reg)
            
        return True

    def execute_basic_block(self, data: bytes, addr: Optional[Addr] = None) -> bool:
        """
        Execute a whole bunch of bytes as instructions. Consume the whole
        data and execute it symbolically. No payload is written in the triton
        context. The basic block is executed 'out of thin air'.

        :param data: bytes of instructions to execute
        :type data: bytes
        :param addr: optional address of the first instruction
        :type addr: Optional[:py:obj:`qsynthesis.types.Addr`]
        :returns: True if execution of all instructions succeeded
        :rtype: bool
        """
        blob = data[:]
        while blob:
            i = self.disassemble(blob, addr)
            addr = None  # reset to None if it was provided
            if not self.execute_instruction(i):
                return False
            blob = blob[i.getSize():]
        return True

    def execute(self, opcode: bytes, addr: Optional[Addr] = None) -> bool:
        """
        Symbolically execute the given opcode at the given optional address.

        :param opcode: bytes of the instruction
        :type opcode: bytes
        :param addr: optional address of the instruction
        :type addr: Optional[:py:obj:`qsynthesis.types.Addr`]
        :returns: True if the instruction has sucessfully been processed
        :rtype: bool
        """
        inst = Instruction(addr, opcode) if addr is not None else Instruction(opcode)
        return self.execute_instruction(inst)

    def execute_instruction(self, instr: Instruction) -> bool:
        """
        Symbolically execute the given triton Instruction already instanciated.

        :param instr: Triton Instruction
        :type instr: `Instruction <https://triton.quarkslab.com/documentation/doxygen/py_Instruction_page.html>`_
        :returns: True if the processing has successfully been performed
        :rtype: bool
        """
        # Update object values
        self.cur_inst = instr
        self.inst_id += 1
        self.expr_id = 0

        # Process instruction
        self._turn_on()

        # Call any instruction callback if any
        if self._inst_cbs:
            for cb in self._inst_cbs:
                cb(instr)
        
        r = self.ctx.processing(instr)

        # Set a unique comment on each symbolic expressions
        for e in instr.getSymbolicExpressions():
            e.setComment(self._fmt_comment())

        self._turn_off()
        return r

    def _fmt_comment(self) -> str:
        """Return a string identifying a SymbolicExpression in a unique manner"""
        return f"{self.inst_id}#{self.expr_id}#{self.current_address}"

    def add_instruction_callback(self, cb: Callable) -> None:
        """
        Add an Instruction callback that will be called for every
        instruction executed.
        """
        self._inst_cbs.append(cb)

    @staticmethod
    def _memacc_to_all_addr(ma: MemoryAccess) -> Set[int]:
        """Return a set of all addresses of a MemoryAccess"""
        addr = ma.getAddress()
        return set(range(addr, addr+ma.getSize()))

    @staticmethod
    def _split_unaligned_access(addr: Addr, size: ByteSize) -> List[MemoryAccess]:
        max_ma_size = 64  # Max aligned memory access size
        ma_size = 1
        splitted_ma = []
        current_addr = addr + size
        while ma_size < max_ma_size:
            if size & 0x1:
                current_addr -= ma_size
                splitted_ma.append(MemoryAccess(current_addr, ma_size))
            ma_size <<= 1
            size >>= 1
        for _ in range(size):
            current_addr -= ma_size
            splitted_ma.append(MemoryAccess(current_addr, ma_size))
        return splitted_ma

    @staticmethod
    def _coalesce_bytes_to_mas(ma_bytes: Iterable[Addr]) -> List[MemoryAccess]:
        """Convert a bunch of addresses into a list of MemoryAccess"""
        tmp_mem_accesses = []
        sorted_bytes = sorted(ma_bytes)
        addr, size = sorted_bytes[0], 1
        for b in sorted_bytes[1:]:
            if b == addr + size:
                size += 1
            else:
                tmp_mem_accesses.append((addr, size))
                addr = b
                size = 1
        tmp_mem_accesses.append((addr, size))
        coalesced_ma = []
        for addr, size in tmp_mem_accesses:
            coalesced_ma.extend(SimpleSymExec._split_unaligned_access(addr, size))
        return coalesced_ma
