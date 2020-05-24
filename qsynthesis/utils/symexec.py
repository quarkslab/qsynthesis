# built-in libs
from typing import List, Set, Iterable, TypeVar, Union

# third-party libs
from triton import ARCH, CALLBACK, MODE, MemoryAccess, Instruction, AST_REPRESENTATION
from triton import TritonContext

# qsynthesis deps
from qsynthesis.tritonast import TritonAst


Register = TypeVar("Register")  # Triton Register type


class SimpleSymExec:
    """
    Helper class used to process a sequential bunch of instructions
    using Triton and mark as symbolic all memory and registers
    reads that occoured before a write to that same register
    or memory address
    """

    def __init__(self, arch: ARCH):
        """
        Initialize symbolic execution

        :param arch: Triton architecture identifier
        """

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
        self.ctx.addCallback(self._mem_write_callback, CALLBACK.SET_CONCRETE_MEMORY_VALUE)
        self.ctx.addCallback(self._mem_read_callback, CALLBACK.GET_CONCRETE_MEMORY_VALUE)
        self.ctx.addCallback(self._reg_write_callback, CALLBACK.SET_CONCRETE_REGISTER_VALUE)
        self.ctx.addCallback(self._reg_read_callback, CALLBACK.GET_CONCRETE_REGISTER_VALUE)

    @property
    def arch(self):
        return self.ctx.getArchitecture()

    @property
    def flags_reg(self):
        _mapper = {ARCH.X86: self.ctx.registers.eflags,
                   ARCH.X86_64: self.ctx.registers.eflags,
                   ARCH.ARM32: self.ctx.registers.cpsr,
                   ARCH.AARCH64: self.ctx.registers.spsr}  # Not really true for spsr
        return _mapper[self.arch]

    @property
    def ins_ptr_reg(self):
        _mapper = {ARCH.X86: self.ctx.registers.eip,
                   ARCH.X86_64: self.ctx.registers.rip,
                   ARCH.ARM32: self.ctx.registers.pc,
                   ARCH.AARCH64: self.ctx.registers.pc}
        return _mapper[self.arch]

    def turn_on(self) -> None:
        """
        Turn on capturing
        """
        self._capturing = True

    def turn_off(self) -> None:
        """
        Turn off capturing
        """
        self._capturing = False

    def _mem_write_callback(self, _: TritonContext, ma: MemoryAccess, __: int) -> None:
        if not self._capturing:
            return

        # Add bytes to set of written bytes
        self.mem_addr_seen |= self.memacc_to_all_addr(ma)

    def _mem_read_callback(self, ctx: TritonContext, ma: MemoryAccess) -> None:
        if not self._capturing:
            return

        # Retrieve all addresses of a given mem access
        addrs = self.memacc_to_all_addr(ma)

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
        for ma in self.coalesce_bytes_to_mas(new_addrs):
            comment = "mem_{:#x}_{}".format(ma.getAddress(), ma.getSize())
            symvar = ctx.symbolizeMemory(ma, comment)

            self.mem_symvars.append(symvar)

    def _reg_write_callback(self, ctx: TritonContext, reg: Register, _: int) -> None:
        if not self._capturing:
            return

        # Add parent register to set of written registers)
        parent_reg = ctx.getParentRegister(reg)
        self.reg_id_seen.add(parent_reg.getId())

    def _reg_read_callback(self, ctx: TritonContext, reg: Register) -> None:
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

        # Symbolize the register
        comment = f"reg_{reg.getName()}_at_{self.ctx.getConcreteRegisterValue(self.ins_ptr_reg)}"
        self.symbolize_register(reg, 0, comment)

    def get_register_ast(self, reg_name: Union[str, Register]) -> TritonAst:
        reg = getattr(self.ctx.registers, reg_name) if isinstance(reg_name, str) else reg_name
        reg_se = self.ctx.getSymbolicRegister(reg)
        actx = self.ctx.getAstContext()
        if reg_se is None:
            e = actx.bv(self.ctx.getConcreteRegisterValue(reg), reg.getBitSize())
        else:
            e = actx.unroll(reg_se.getAst())
        return TritonAst.make_ast(self.ctx, e)

    def get_memory_ast(self, addr: int, size: int):
        raise NotImplementedError("getting AST of memory not implemented yet")

    def symbolize_register(self, reg, value, comment):
        self.reg_id_seen.add(reg.getId())
        symvar = self.ctx.symbolizeRegister(reg, comment)
        symvar.setAlias(reg.getName())

        # We also set the symbolic var to the actual value of the register
        self.ctx.setConcreteVariableValue(symvar, value)
        self.reg_symvars.append(symvar)

    def initialize_register(self, reg: Union[str, Register], value: int):
        reg = getattr(self.ctx.registers, reg) if isinstance(reg, str) else reg
        self.reg_id_seen.add(reg.getId())
        self.ctx.setConcreteRegisterValue(reg, value)

    def execute(self, addr: int, opcode: bytes) -> bool:
        inst = Instruction(addr, opcode)
        return self.execute_instruction(inst)

    def execute_instruction(self, instr: Instruction) -> bool:
        # Process instruction
        self.turn_on()
        r = self.ctx.processing(instr)
        self.turn_off()
        return r

    @staticmethod
    def memacc_to_all_addr(ma: MemoryAccess) -> Set[int]:
        addr = ma.getAddress()
        return set(range(addr, addr+ma.getSize()))

    @staticmethod
    def split_unaligned_access(addr: int, size: int) -> List[MemoryAccess]:
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
    def coalesce_bytes_to_mas(ma_bytes: Iterable[int]) -> List[MemoryAccess]:
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
            coalesced_ma.extend(SimpleSymExec.split_unaligned_access(addr, size))
        return coalesced_ma
