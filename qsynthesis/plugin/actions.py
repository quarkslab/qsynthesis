from qsynthesis.plugin.dependencies import ida_kernwin, TRITON_ENABLED
from qsynthesis.plugin.view import SynthesizerView


class SynthetizerViewHook(ida_kernwin.action_handler_t):

    view_id = "QSynthesis"

    def __init__(self, qtrace):
        ida_kernwin.action_handler_t.__init__(self)
        from qtraceida.icons.raw_icons import ICON_DEBUG_ID, IDAIcon
        self.view = None
        self.name = SynthesizerView.NAME
        self.icon = IDAIcon.AST_TREE
        self.tooltip = "Synthesizing arithmetic expressions along the trace"
        self.qtrace = qtrace

    @property
    def is_enabled_by_trace(self):
        return True

    def activate(self, ctx):
        """
        Overridden method called on hook creation
        """
        # pylint: disable=unused-argument
        if self.view is None or self.view.closed:
            self.view = SynthesizerView(self.qtrace)
            # self.view.init()
            self.view.Show()
        return True

    def update(self, ctx):
        """
        Overridden method called on hook update
        """
        if TRITON_ENABLED:
            return ida_kernwin.AST_DISABLE if self.is_enabled_by_trace else ida_kernwin.AST_ENABLE_ALWAYS
        else:
            self.tooltip = "Require Triton to run"
            return ida_kernwin.AST_DISABLE_ALWAYS
