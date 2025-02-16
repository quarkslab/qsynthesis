from qsynthesis.plugin.dependencies import ida_kernwin, TRITON_ENABLED, Trace


class SynthetizerViewHook(ida_kernwin.action_handler_t):
    """
    Standard hook to all Qtrace-IDA plugins. When used with
    Qtace-IDA, Qsynthesis is launched through this action_handler_t
    which will be registered in the qtrace-ida menu.
    """

    view_id = "QSynthesis"

    def __init__(self, qtrace: 'QtraceIDA'):
        """
        Constructor that receive a QtraceIDA instance (which is the main object of Qtrace-IDA
        holding the trace etc)
        """
        ida_kernwin.action_handler_t.__init__(self)
        from qtraceida.icons.raw_icons import ICON_DEBUG_ID, IDAIcon
        self.view = None
        self.name = "QSynthesis"  # SynthesizerView.NAME
        self.icon = IDAIcon.AST_TREE
        self.tooltip = "Synthesizing arithmetic expressions along the trace"
        self.qtrace = qtrace

    @property
    def is_enabled_by_trace(self) -> bool:
        """
        Boolean indicating if this qtraceida plugin should be enabled
        when a trace is opened.
        """
        return True

    def on_trace_opened(self, t: Trace) -> None:
        """
        Callback called when a trace is opened in Qtrace-IDA.
        Here we just forward the call to the main view.
        """
        if self.view:  # If the view has been instanciated
            self.view.on_trace_opened(t)

    def on_trace_closed(self) -> None:
        pass # TODO: Implementing proper deactivation of the view

    def activate(self, _) -> bool:
        """
        Main callback called when the menu entry is clicked or the Short-keys
        typed. This `action_handler_t` has to be implemented. If called the
        action open the QSynthesis view if not already done.
        """
        # pylint: disable=unused-argument
        from qsynthesis.plugin.view import SynthesizerView
        if self.view is None or self.view.closed:
            self.view = SynthesizerView(self.qtrace)
            self.view.Show()  # will call OnCreate and init and on_trace_opened by recursivity
        return True

    def update(self, _) -> int:
        """
        Callback called whenever the the hook is updated.
        If dependency are not satisfied always disabled otherwise ok.
        """
        if TRITON_ENABLED:
            return ida_kernwin.AST_DISABLE if self.is_enabled_by_trace else ida_kernwin.AST_ENABLE_ALWAYS
        else:
            self.tooltip = "Require Triton to run"
            return ida_kernwin.AST_DISABLE_ALWAYS
