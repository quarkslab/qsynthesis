# Disclaimer:
# This file is a partial rip of the processor module of idacore
# it is meant to be replaced by idacore in a near future

# built-in modules
from enum import IntEnum

# third-party modules
from triton import ARCH

# qsynthesis modules
from qsynthesis.plugin.dependencies import ida_idp, ida_idaapi
from qsynthesis.plugin.dependencies import ArchX86, ArchX64, ArchARM, ArchARM64


class ProcessorType(IntEnum):
    """
    Enum of IDA Processor supported
    """
    UNKNOWN = -1
    PLFM_386 = ida_idp.PLFM_386       # Intel 80x86
    PLFM_ARM = ida_idp.PLFM_ARM       # Advanced RISC Machines


SUPPORTED_PROC = [ProcessorType.PLFM_386.value, ProcessorType.PLFM_ARM]


class ArchMode(IntEnum):
    """
    Enum of supported modes for each architectures
    """
    MODE32 = 0
    MODE64 = 1


class Processor:
    """
    Small idp wrapper to represent the current processor.
    """
    name = ida_idp.get_idp_name()
    id = ida_idp.ph_get_id()
    flag = ida_idp.ph_get_flag()
    type = ProcessorType.UNKNOWN if id not in SUPPORTED_PROC else ProcessorType(id)
    mode = ArchMode.MODE64 if ida_idaapi.get_inf_structure().is_64bit() else ArchMode.MODE32


def processor_to_triton_arch() -> ARCH:
    """
    Get the current IDB processor as a Triton ARCH object.

    :return: Triton ARCH from IDA processor type
    """
    if Processor.type == ProcessorType.PLFM_386:
        return ARCH.X86_64 if Processor.mode == ArchMode.MODE64 else ARCH.X86
    elif Processor.type == ProcessorType.PLFM_ARM:
        return ARCH.AARCH64 if Processor.mode == ArchMode.MODE64 else ARCH.ARM32
    else:
        assert False


def processor_to_arch():
    """
    Get the current IDB processor as a Qtrace-DB Arch object

    :return: Qtrace-DB Arch object from IDA processor type
    """
    if Processor.type == ProcessorType.PLFM_386:
        return ArchX64 if Processor.mode == ArchMode.MODE64 else ArchX86
    elif Processor.type == ProcessorType.PLFM_ARM:
        return ArchARM64 if Processor.mode == ArchMode.MODE64 else ArchARM
    else:
        assert False
