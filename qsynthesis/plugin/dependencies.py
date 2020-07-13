
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
    IDA_ENABLED = True
except ImportError:
    # If we cannot import ida Mock most of its API to works equally without
    class ida_kernwin:
        class PluginForm:
            pass
        class action_handler_t:
            pass

    class ida_idaapi:
        PLUGIN_UNL = None
        PLUGIN_OK = None

        class plugin_t:
            pass

        class get_inf_structure:
            def is_64bit(self):
                return True

    class ida_gdl:
        pass

    class ida_bytes:
        pass

    class ida_idp:
        PLFM_386 = 0
        PLFM_ARM = 1

        @staticmethod
        def get_idp_name():
            return "plop"
        @staticmethod
        def ph_get_id():
            return ida_idp.PLFM_386
        @staticmethod
        def ph_get_flag():
            return 12

    class ida_nalt:
        pass

    class ida_funcs:
        pass

    class ida_ua:
        pass

    class ida_loader:
        pass

    class ida_auto:
        pass

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