# built-in libs
from enum import Enum
import struct
import logging
from typing import List

# third-party libs
from triton import ARCH, MemoryAccess
from triton import TritonContext

from qtracedb.trace import Trace, MemAccessType
from qtracedb.archs import ArchsManager


from qsynthesis.utils.symexec import SimpleSymExec, Register


TRITON_ARCH_MAP = {
    'x86_32': ARCH.X86,
    'x86_64': ARCH.X86_64,
    'ARM': ARCH.ARM32,
    'ARM64': ARCH.AARCH64
}


class Mode(Enum):
    """
    Mode to select the initial state of the symbolic executor.
    FULL_SYMBOLIC: Keeps everything symbolic (all new unread register / memory will be new inputs
    PARAM_SYMBOLIC: Concretize everything but function parameters (use arch calling convention)
                    This mode is usefull when executing only a single function.
    """
    FULL_SYMBOLIC = 1  # Symbolize any reg/mem newly read without having been previously written
    PARAM_SYMBOLIC = 2  # Symbolize only parameters keeps all the rest concrete


class QtraceSymExec(SimpleSymExec):
    """
    Helper class to perform symbolic execution on a Qtrace-DB trace.
    It allows to process a range of instructions. Provide two simple
    concretization modes.
    """

    def __init__(self, trace: Trace, mode: Mode = Mode.FULL_SYMBOLIC):
        """
        Instanciate QtraceSymExec with a trace object and a mode to

        :param trace: a qtrace-db trace object
        :param mode: a Mode enum value
        """
        self.trace_arch = trace.get_arch()
        if self.trace_arch.NAME not in TRITON_ARCH_MAP:
            raise TypeError("Trace architecture not supported by Triton")
        arch = TRITON_ARCH_MAP[self.trace_arch.NAME]
        super(QtraceSymExec, self).__init__(arch)
        self.trace = trace
        self.mode = mode
        self._cur_db_inst = None  # object attribute holding current instruction being analysed
        self._sup_regs = None

    @property
    def inst_id(self) -> int:
        """Overwrite Full Symbolic instruction identifier by the trace one"""
        return self._cur_db_inst.id

    @inst_id.setter
    def inst_id(self, value):
        """Set the current inst_id value"""
        pass

    @property
    def parameter_regs(self) -> List[Register]:
        """
        Return the ordered list of Register for the call convention of the current architecture.
        :returns: list of registers involved in the calling convention
        """
        return [getattr(self.ctx.registers, x.name.lower()) for x in self.trace_arch.registers_cc]

    @property
    def supported_regs(self) -> List[Register]:
        """
        Map Qtrace-DB supported registers to Triton registers.
        It assumes registers strings exists in Triton in lowercase
        :returns: list of Triton registers supported in the trace (namely gathered and present in DB)
        """
        if self._sup_regs is None:  # Lazily compute it. Only done once
            rgs = ArchsManager.get_supported_regs(self.trace_arch)
            self._sup_regs = [getattr(self.ctx.registers, x.name.lower()) for x in rgs]
        return self._sup_regs

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

        # Retrieve qtracedb's memory accesses for this instr
        qtracedb_mas = self.trace.get_memacc_by_instr(self._cur_db_inst)

        if self.mode == Mode.FULL_SYMBOLIC:
            # Symbolize all the memory cells that have not yet been seen

            # Coalesce adjacent bytes and create memory accesses' symvars
            for ma in self._coalesce_bytes_to_mas(new_addrs):
                symvar = self.symbolize_memory(ma)

                # Set symbolic variable value according to memory content
                for qtracedb_ma in qtracedb_mas:

                    start1 = ma.getAddress()
                    end1 = start1 + ma.getSize()

                    start2 = qtracedb_ma.addr
                    end2 = start2 + len(qtracedb_ma.data)

                    # If this symvar is contained in this qtracedb memory access
                    # we can extract the value and set it
                    if start1 >= start2 and end1 <= end2:
                        overlap_data = qtracedb_ma.data[start1 - qtracedb_ma.addr: end1 - qtracedb_ma.addr]
                        overlap_data = overlap_data + b'\x00' * (8 - len(overlap_data))  # Little-Endian 0 padding

                        # FIXME: It assume to be qword but might be shorter
                        mem_value = struct.unpack("<Q", overlap_data)[0]
                        ctx.setConcreteVariableValue(symvar, mem_value)
                        break
                self.mem_symvars.append(symvar)

        elif self.mode == Mode.PARAM_SYMBOLIC:
            logging.debug(f"Concretize unseen memory:{new_addrs}")
            # Concretize all the memory cell
            addr, size = ma.getAddress(), ma.getSize()

            mems = [x for x in qtracedb_mas if x.kind == MemAccessType.read]
            if not mems:
                logging.warning("no memory read for the instruction")
            mapping = {mem.addr + i: mem.data[i] for mem in mems for i in range(len(mem.data))}
            logging.debug(f"Mem read: Triton:[{addr:#x}:{size}] Qtrace-DB:[{mems[0].addr:#x}:{len(mems[0].data)}]")

            for index in range(addr, addr + size):
                if index in mapping:
                    self.ctx.setConcreteMemoryValue(index, mapping[index])
                else:
                    logging.error(f"address {index:02X} not in the mapping")

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

        if parent_reg not in self.supported_regs:
            return

        if self.mode == Mode.FULL_SYMBOLIC or (self.mode == Mode.PARAM_SYMBOLIC and parent_reg in self.parameter_regs):
            self.symbolize_register(reg, 0)
        else:
            logging.debug(f"Concretizing the never seen register: {reg.getName()}")
            self.concretize_register(reg)

    def register_value(self, reg: Register) -> int:
        """
        Get concrete value of the given triton Register in the trace.

        :param reg: triton Register
        :returns: dynamic value in the trace
        """
        return getattr(self._cur_db_inst, reg.getName().upper())

    def concretize_register(self, reg: Register) -> None:
        """
        Concretize the given register using its dynamic value
        read in the trace.

        :param reg: triton Register
        :returns: None
        """
        self.reg_id_seen.add(reg.getId())
        self.ctx.setConcreteRegisterValue(reg, self.register_value(reg))

    def process_instr_sequence(self, start_id: int, stop_id: int, check_regs: bool = False) -> None:
        """
        Process a given instruction sequence in the trace.

        :param start_id: Trace ID (offset) where to start
        :param stop_id: trace ID (offset) where to stop (the instruction at this address is not executed).
        :param check_regs: Activate checking register values for desynchronization
        :returns: None
        """

        # Process instructions
        for qtracedb_inst_id in range(start_id, stop_id):

            # Save current qtracedb instr into the object in order (to make it accessible from callbacks)
            self._cur_db_inst = self.trace.get_instr(qtracedb_inst_id)

            if check_regs:
                self.sync_registers()

            self.execute(opcode=self._cur_db_inst.opcode, addr=self._cur_db_inst.addr)

    def sync_registers(self) -> None:
        """
        Synchronize Triton registers and trace registers. If the value
        mismatch, patch the triton register value with the one of the trace.
        """
        # Find and fix mismatches in registers values
        for reg in self.supported_regs:

            # Note: using getConcreteRegisterValue doesn't necessarily gives the
            # right value in case of symbolized register.
            # triton_reg_val = self.ctx.getConcreteRegisterValue(triton_reg)
            triton_reg_val = self.ctx.getRegisterAst(reg).evaluate()

            qtracedb_reg_val = self.register_value(reg)

            if qtracedb_reg_val is None:
                continue

            if triton_reg_val != qtracedb_reg_val:
                logging.debug(f'{self.inst_id} {reg} <- {qtracedb_reg_val:#x} (was {triton_reg_val:#x})')

                self.ctx.setConcreteRegisterValue(reg, qtracedb_reg_val)

                # Check if register is symbolized. Afterword there are three possible
                # ways to handle this, see comments below
                reg_sym_expr = self.ctx.getSymbolicRegister(reg)
                if reg_sym_expr is not None and reg_sym_expr.isSymbolized():
                    # 1) We simply warn the user we are about to concretize a symbolic
                    #    register. This is the safest option but may result in some symbolic
                    #    variables not being propagated to the end result (concrete values
                    #    will be propagated instead)
                    logging.debug(f'Warning: symbolic register {reg} will be concretized')
                    # TODO show symbolic variables

                self.ctx.concretizeRegister(reg)
