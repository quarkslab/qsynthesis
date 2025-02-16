from typing import List, Optional

import capstone
from capstone import x86_const, arm_const, arm64_const, CS_AC_READ, CS_AC_WRITE
from enum import IntEnum
from collections import namedtuple

Reg = namedtuple("Reg", "name")


class Opnd:
    def __init__(self, cs_op, arch):
        self._cs_op = cs_op
        self._arch = arch

    @property
    def type(self):
        return self._cs_op.type

    def is_read(self) -> bool:
        return bool(self._cs_op.access & CS_AC_READ)

    def is_written(self) -> bool:
        return bool(self._cs_op.access & CS_AC_WRITE)

    def is_register(self) -> bool:
        return self.type == self._arch.optypes.REG

    def is_memory(self) -> bool:
        return self.type == self._arch.optypes.MEM


class Instr:
    def __init__(self, cs_ins, arch):
        self._cs_ins = cs_ins
        self._arch = arch

    @property
    def bytes(self):
        return bytes(self._cs_ins.bytes)

    def __str__(self):
        return self._cs_ins.mnemonic + " " + self._cs_ins.op_str

    @property
    def operands(self):
        return [Opnd(x, self._arch) for x in self._cs_ins.operands]


class Arch:
    _CSD = None

    def disasm(cls, asm: bytes, addr: int) -> List[Instr]:
        cls._CSD.detail = True
        return [Instr(x, cls) for x in cls._CSD.disasm(asm, addr)]

    def disasm_one(cls, asm: bytes, addr: int) -> Instr:
        r = cls.disasm(asm, addr)
        return r[0] if r else None



class _ArchX86(Arch):
    NAME = "x86"
    INS_PTR = Reg('eip')
    STK_PTR = Reg('esp')
    _CSD = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_32)
    nop_instruction = b"\x90"

    class optypes(IntEnum):
        INVALID = x86_const.X86_OP_INVALID
        IMM = x86_const.X86_OP_IMM
        REG = x86_const.X86_OP_REG
        MEM = x86_const.X86_OP_MEM


class _ArchX64(_ArchX86):
    NAME = "x86_64"
    INS_PTR = Reg('rip')
    STK_PTR = Reg('rsp')
    _CSD = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)



class _ArchARM(Arch):
    NAME = "ARM"
    INS_PTR = Reg('pc')
    STK_PTR = Reg('sp')
    _CSD = capstone.Cs(capstone.CS_ARCH_ARM, capstone.CS_MODE_ARM)
    nop_instruction = b"\x00\xf0\x20\xe3"

    class optypes(IntEnum):
        INVALID = arm_const.ARM_OP_INVALID
        REG = arm_const.ARM_OP_REG
        IMM = arm_const.ARM_OP_IMM
        MEM = arm_const.ARM_OP_MEM
        FP = arm_const.ARM_OP_FP
        CIMM = arm_const.ARM_OP_CIMM
        PIMM = arm_const.ARM_OP_PIMM
        SETEND = arm_const.ARM_OP_SETEND
        SYSREG = arm_const.ARM_OP_SYSREG


class _ArchARM64(Arch):
    NAME = "AARCH64"
    INS_PTR = Reg('x28')
    STK_PTR = Reg('sp')
    _CSD = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)
    nop_instruction = b"\x1f\x20\x03\xd5"

    class optypes(IntEnum):
        INVALID = arm64_const.ARM64_OP_INVALID
        REG = arm64_const.ARM64_OP_REG
        IMM = arm64_const.ARM64_OP_IMM
        MEM = arm64_const.ARM64_OP_MEM
        FP = arm64_const.ARM64_OP_FP
        CIMM = arm64_const.ARM64_OP_CIMM
        REG_MRS = arm64_const.ARM64_OP_REG_MRS
        REG_MSR = arm64_const.ARM64_OP_REG_MSR
        PSTATE = arm64_const.ARM64_OP_PSTATE
        SYS = arm64_const.ARM64_OP_SYS
        PREFETCH = arm64_const.ARM64_OP_PREFETCH
        BARRIER = arm64_const.ARM64_OP_BARRIER


ArchX86 = _ArchX86()
ArchX64 = _ArchX64()
ArchARM = _ArchARM()
ArchARM64 = _ArchARM64()


class ArchsManager:
    @staticmethod
    def get_supported_regs(arch: Arch) -> List[Reg]:
        if isinstance(arch, _ArchX64):
            return [Reg('RAX'), Reg('RBX'), Reg('RCX'), Reg('RDX'), Reg('RDI'), Reg('RSI'), Reg('RBP'), Reg('RSP'),
                    Reg('RIP'), Reg('EFLAGS'), Reg('R8'), Reg('R9'), Reg('R10'), Reg('R11'), Reg('R12'), Reg('R13'),
                    Reg('R14'), Reg('R15')]
        elif isinstance(arch, _ArchX86):
            return [Reg('EAX'), Reg('EBX'), Reg('ECX'), Reg('EDX'), Reg('EDI'), Reg('ESI'), Reg('EBP'), Reg('ESP'), Reg('EIP'), Reg('EFLAGS')]
        elif isinstance(arch, _ArchARM):
            return [Reg('R0'), Reg('R1'), Reg('R2'), Reg('R3'), Reg('R4'), Reg('R5'), Reg('R6'), Reg('R7'), Reg('R8'),
                    Reg('R9'), Reg('R10'), Reg('R11'), Reg('R12'), Reg('R13'), Reg('R14'), Reg('R15'), Reg('CPSR')]
        elif isinstance(arch, _ArchARM64):
            return [Reg('X0'), Reg('X1'), Reg('X2'), Reg('X3'), Reg('X4'), Reg('X5'), Reg('X6'), Reg('X7'), Reg('X8'),
                    Reg('X9'), Reg('X10'), Reg('X11'), Reg('X12'), Reg('X13'), Reg('X14'), Reg('X15'), Reg('X16'),
                    Reg('X17'), Reg('X18'), Reg('X19'), Reg('X20'), Reg('X21'), Reg('X22'), Reg('R23'), Reg('X24'),
                    Reg('X25'), Reg('X26'), Reg('X27'), Reg('X28'), Reg('X29'), Reg('X30')]
        else:
            assert False
