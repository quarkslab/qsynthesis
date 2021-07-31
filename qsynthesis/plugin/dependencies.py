from unittest.mock import MagicMock

try:
    import ida_idaapi
    import ida_kernwin
    import ida_gdl
    import ida_funcs
    import ida_bytes
    import ida_idp
    import ida_nalt
    import ida_ua
    import ida_loader
    import ida_auto
    import ida_graph
    import ida_lines
    IDA_ENABLED = True
except ImportError:
    # If we cannot import ida Mock most of its API to works equally without
    class ida_kernwin:
        class PluginForm:
            pass
        action_handler_t = MagicMock()
    ida_idaapi = MagicMock()
    ida_gdl = MagicMock()
    ida_bytes = MagicMock()
    ida_idp = MagicMock()
    ida_nalt = MagicMock()
    ida_funcs = MagicMock()
    ida_ua = MagicMock()
    ida_loader = MagicMock()
    ida_auto = MagicMock()
    ida_graph = MagicMock()
    ida_lines = MagicMock()

    IDA_ENABLED = False


try:
    import qtraceida
    QTRACEIDA_ENABLED = True
except ImportError:
    QTRACEIDA_ENABLED = False


try:
    import qtracedb
    from qtracedb import DatabaseManager
    from qtracedb.trace import Trace, InstrCtx
    from qtracedb.archs.arch import Instr, Arch
    from qtracedb.manager import ArchsManager
    from qtracedb.archs.x86 import ArchX86, ArchX64
    from qtracedb.archs.arm import ArchARM
    from qtracedb.archs.arm64 import ArchARM64
    QTRACEDB_ENABLED = True
except ImportError as e:
    QTRACEDB_ENABLED = False
    from .arch import Arch, ArchX64, ArchARM, ArchARM64, ArchX86, Instr, ArchsManager
    # Set dummy vars
    DatabaseManager, Trace, InstrCtx = None, None, None


try:
    import triton
    TRITON_ENABLED = True
except ImportError:
    TRITON_ENABLED = False
