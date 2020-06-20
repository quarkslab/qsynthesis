#
# Disclaimer:
# This file is a partial rip of the processor module of idacore
# it is meant to be replaced by idacore in a near future

from enum import IntEnum
from qsynthesis.plugin.dependencies import ida_idp, ida_idaapi

from triton import ARCH
from qtracedb.archs.arch import Arch
from qtracedb.archs.x86 import ArchX86, ArchX64
from qtracedb.archs.arm import ArchARM
from qtracedb.archs.arm64 import ArchARM64


class ProcessorType(IntEnum):
    UNKNOWN = -1
    PLFM_386 = ida_idp.PLFM_386       # Intel 80x86
    PLFM_ARM = ida_idp.PLFM_ARM       # Advanced RISC Machines


SUPPORTED_PROC = [ProcessorType.PLFM_386.value, ProcessorType.PLFM_ARM]


class ArchMode(IntEnum):
    MODE32 = 0
    MODE64 = 1


class Processor:
    name = ida_idp.get_idp_name()
    id = ida_idp.ph_get_id()
    flag = ida_idp.ph_get_flag()
    type = ProcessorType.UNKNOWN if id not in SUPPORTED_PROC else ProcessorType(id)
    mode = ArchMode.MODE64 if ida_idaapi.get_inf_structure().is_64bit() else ArchMode.MODE32


def processor_to_triton_arch() -> ARCH:
    if Processor.type == ProcessorType.PLFM_386:
        return ARCH.X86_64 if Processor.mode == ArchMode.MODE64 else ARCH.X86
    elif Processor.type == ProcessorType.PLFM_ARM:
        return ARCH.AARCH64 if Processor.mode == ArchMode.MODE64 else ARCH.ARM32
    else:
        assert False


def processor_to_qtracedb_arch() -> Arch:
    if Processor.type == ProcessorType.PLFM_386:
        return ArchX64 if Processor.mode == ArchMode.MODE64 else ArchX86
    elif Processor.type == ProcessorType.PLFM_ARM:
        return ArchARM64 if Processor.mode == ArchMode.MODE64 else ArchARM
    else:
        assert False
