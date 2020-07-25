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
    QTRACEDB_ENABLED = True
except ImportError:
    QTRACEDB_ENABLED = False

try:
    import triton
    TRITON_ENABLED = True
except ImportError:
    TRITON_ENABLED = False
