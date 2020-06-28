from enum import Enum
from PyQt5 import QtWidgets, QtCore, QtGui
from pathlib import Path

from qsynthesis.plugin.dependencies import ida_kernwin, IDA_ENABLED, QTRACEIDA_ENABLED, QTRACEDB_ENABLED, ida_bytes, ida_nalt
from qsynthesis.tables import LookupTableDB, LookupTableREST
from qsynthesis.algorithms import TopDownSynthesizer, PlaceHolderSynthesizer
from qsynthesis.utils.symexec import SimpleSymExec
from qsynthesis.plugin.processor import processor_to_triton_arch, processor_to_qtracedb_arch, Arch, Processor, ProcessorType
from qsynthesis.plugin.ast_viewer import AstViewer
from qsynthesis.plugin.popup_actions import SynthetizeFromHere, SynthetizeToHere, SynthetizeOperand
from qtraceanalysis.slicing import Slicer
from qtracedb import DatabaseManager
from qtracedb.trace import Trace
from qtracedb.archs import ArchsManager
from .ui.synthesis_ui import Ui_synthesis_view


class TraceDbType(Enum):
    CONFIG = 0
    SQLITE = 1


class TargetType(Enum):
    REG = 0
    MEMORY = 1
    OPERAND = 2


class TableType(Enum):
    SQLITE = 0
    HTTP = 1


class AlgorithmType(Enum):
    TOPDOWN = "Top-Down"
    PLHDR = "PlaceHolders"


class AnalysisType(Enum):
    QTRACE = 0
    FULL_SYMBOLIC = 1


class QtraceSymType(Enum):
    FULL_SYMBOLIC = 0
    PARAM_SYMBOLIC = 1


class ShowDepState(Enum):
    SHOW = "Highlight Deps"
    HIDE = "Hide Deps"


class SynthesizerView(ida_kernwin.PluginForm, QtWidgets.QWidget, Ui_synthesis_view):

    NAME = "QSynthesis"

    def __init__(self, qtrace):
        QtWidgets.QWidget.__init__(self)
        ida_kernwin.PluginForm.__init__(self)
        self.qtrace = qtrace
        self.parent_widget = self

        # Visibility state
        self.closed = True

        # Popup actions
        self.popup_from_here = SynthetizeFromHere(self)
        self.popup_to_here = SynthetizeToHere(self)
        self.popup_operand = SynthetizeOperand(self)

        # Analysis variables
        self.symexec = None
        self.lookuptable = None
        self.ast = None
        self.synthesizer = None
        self.synth_ast = None
        # For operand synthesis
        self.op_num = None
        self.op_is_read = False  # Negation of if the operand is written

        # If working on its own
        self._dbm = None
        self._trace = None
        self._arch = processor_to_qtracedb_arch()

        # Expresssion highlighted
        self.highlighted_addr = {}  # addr -> backed_color


    @property
    def arch(self) -> Arch:
        if QTRACEIDA_ENABLED:
            return self.qtrace.arch
        else:
            return self._arch

    @property
    def trace(self) -> Trace:
        if QTRACEIDA_ENABLED:
            return self.qtrace.trace
        else:
            return self._trace

    def is_lookuptable_loaded(self):
        return self.lookuptable is not None

    def OnCreate(self, form):
        print("On Create called !")
        self.parent_widget = self.FormToPyQtWidget(form)
        #self.setupUi(self.parent_widget)
        # Init of the view has to be done here ?
        self.init()

        # Put that check here otherwise init will override this enabling
        # NOTE: Might be better to implement it in a proprer callback
        if self.trace:
            self.on_trace_opened(self.trace)
        else:
            print("Trace is none it should not show up like this")

    def Show(self):
        """
        Creates the form if not created or focuses it if it was
        """
        print("On Show called !")
        self.closed = False
        self.enable_popups()
        opts = ida_kernwin.PluginForm.WOPN_PERSIST
        r = ida_kernwin.PluginForm.Show(self, self.NAME, options=opts)
        ida_kernwin.set_dock_pos(self.NAME, "IDA View-A", ida_kernwin.DP_RIGHT)
        return r

    def OnClose(self, form):
        # Change visibility state
        self.closed = True
        self.disable_popups()

    def enable_popups(self):
        self.popup_from_here.register()
        self.popup_to_here.register()
        self.popup_operand.register()

    def disable_popups(self):
        self.popup_from_here.unregister()
        self.popup_to_here.unregister()
        self.popup_operand.unregister()

    def set_visible_all_layout(self, layout, val):
        for i in range(layout.count()):
            item = layout.itemAt(i)
            item_w = item.widget()
            if item_w is None:
                if isinstance(item, QtWidgets.QLayout):
                    self.set_visible_all_layout(item, val)
            else:
                item_w.setVisible(val)

    def set_enabled_all_layout(self, layout, val):
        for i in range(layout.count()):
            item = layout.itemAt(i)
            item_w = item.widget()
            if item_w is None:
                if isinstance(item, QtWidgets.QLayout):
                    self.set_enabled_all_layout(item, val)
            else:
                item_w.setEnabled(val)

    def init(self):
        self.setupUi(self.parent_widget)
        if QTRACEIDA_ENABLED or not QTRACEDB_ENABLED:  # The trace will be provided by qtraceida so not showing this
            self.set_visible_all_layout(self.traceLayout, False)
        else:  # Initialize the fields and actions
            self.trace_type_box.addItems([x.name for x in TraceDbType])
            self.trace_type_box.currentIndexChanged.connect(self.trace_type_changed)
            self.trace_line.focusInEvent = self.customfocusInEventTraceLine

        # Target configuration
        reg = QtCore.QRegExp("^(0x)?[a-fA-F0-9]+$")#(:\d+)?$")
        validator = QtGui.QRegExpValidator(reg)
        self.from_line.setValidator(validator)
        self.from_line.textChanged.connect(self.from_line_changed)
        self.from_line.editingFinished.connect(self.from_line_reset_style)
        self.to_line.textChanged.connect(self.to_line_changed)
        self.to_line.editingFinished.connect(self.to_line_reset_style)
        self.to_line.setValidator(validator)
        self.mem_line.textChanged.connect(self.mem_line_changed)
        self.mem_line.editingFinished.connect(self.mem_line_reset_style)
        self.mem_line.setValidator(validator)
        self.target_box.addItems([x.name for x in TargetType])
        self.target_box.currentIndexChanged.connect(self.target_changed)
        self.mem_line.setVisible(False)
        self.operand_label.setVisible(False)

        # Table configuration
        self.table_type_box.addItems([x.name for x in TableType])
        self.table_type_box.currentIndexChanged.connect(self.table_type_changed)
        self.table_line.focusInEvent = self.customfocusInEventLookupTableLine

        #Algorithm configuration
        self.algorithm_box.addItems([x.value for x in AlgorithmType])
        if QTRACEDB_ENABLED:
            self.algorithm_type_box.addItem(AnalysisType.QTRACE.name)
        if IDA_ENABLED:
            self.algorithm_type_box.addItem(AnalysisType.FULL_SYMBOLIC.name)
        self.algorithm_type_box.currentIndexChanged.connect(self.algorithm_type_changed)
        self.qtrace_sym_type_box.addItems([x.name for x in QtraceSymType])

        # Hide all experimental settings
        self.set_visible_all_layout(self.experimentalLayout, False)

        self.run_triton_button.clicked.connect(self.run_triton_clicked)
        self.show_deps_triton_button.setText(ShowDepState.SHOW.value)
        self.show_deps_triton_button.clicked.connect(self.triton_show_deps_clicked)
        self.show_ast_triton_button.clicked.connect(self.triton_show_ast_clicked)
        self.run_synthesis_button.clicked.connect(self.run_synthesis_clicked)
        self.show_ast_synthesis_button.clicked.connect(self.synthesis_show_ast_clicked)
        self.reassemble_button.clicked.connect(self.reassemble_clicked)

    def switch_to_target_operand(self):
        self.target_box.setCurrentIndex(TargetType.OPERAND.value)

    def algorithm_type_changed(self):
        self.qtrace_sym_type_box.setEnabled(self.analysis_type == AnalysisType.QTRACE)

    @property
    def qtrace_sym_type(self) -> QtraceSymType:
        return QtraceSymType(self.qtrace_sym_type_box.currentIndex())

    def from_line_reset_style(self):
        self.from_line.setStyleSheet("")

    def from_line_changed(self):
        color = "green" if self.from_line.hasAcceptableInput() else "red"
        self.from_line.setStyleSheet(f"border: 1px solid {color}")

    def to_line_reset_style(self):
        self.to_line.setStyleSheet("")

    def to_line_changed(self):
        color = "green" if self.to_line.hasAcceptableInput() else "red"
        self.to_line.setStyleSheet(f"border: 1px solid {color}")

    def mem_line_reset_style(self):
        self.mem_line.setStyleSheet("")

    def mem_line_changed(self):
        color = "green" if self.mem_line.hasAcceptableInput() else "red"
        self.mem_line.setStyleSheet(f"border: 1px solid {color}")

    def on_trace_opened(self, trace):
        print("Executer on trace_opened")
        # Activate all configuration lines
        self.set_enabled_all_layout(self.targetLayout, True)
        # Initialize registers of the targetcombobox
        self.register_box.addItems([x.name for x in ArchsManager.get_supported_regs(trace.get_arch())])
        self.set_enabled_all_layout(self.tableLayout, True)
        self.set_enabled_all_layout(self.algorithmLayout, True)
        self.run_triton_button.setEnabled(True)

        #QtWidgets.QApplication.processEvents()
        # if IDA_ENABLED:
        #     print("Try refreshing")
        #     widget = self.TWidgetToPyQtWidget(self.GetWidget())
        #     widget.update()

    def target_changed(self, index):
        type = TargetType(index)
        if type == TargetType.REG:
            self.register_box.setVisible(True)
            self.mem_line.setVisible(False)
            self.operand_label.setVisible(False)
        elif type == TargetType.MEMORY:
            self.register_box.setVisible(False)
            self.operand_label.setVisible(False)
            self.mem_line.setVisible(True)
        elif type == TargetType.OPERAND:
            self.register_box.setVisible(False)
            self.mem_line.setVisible(False)
            self.operand_label.setVisible(True)
            self.operand_label.clear()
        else:
            assert False

    @property
    def target_type(self) -> TargetType:
        return TargetType(self.target_box.currentIndex())

    @property
    def table_type(self) -> TableType:
        return TableType(self.table_type_box.currentIndex())

    def table_type_changed(self, idx):
        self.table_line.setText("")
        if self.table_type == TableType.HTTP:
            self.table_line.setPlaceholderText('e.g: http://localhost')
        else:
            self.table_line.setPlaceholderText("")

    def customfocusInEventLookupTableLine(self, event):
        print("Focus in table line !")
        self.table_line.setStyleSheet("")
        if self.table_type == TableType.SQLITE:
            filename = QtWidgets.QFileDialog.getOpenFileName()[0]
            filepath = Path(filename)
            if filepath.exists() and filepath.is_file():
                self.table_line.setText(str(filepath))
            else:
                print(f"Invalid file: {filepath}")

            self.trace_line.focusNextChild()

    @property
    def algorithm(self) -> AlgorithmType:
        return AlgorithmType(self.algorithm_box.currentText())

    @property
    def analysis_type(self) -> AnalysisType:
        return AnalysisType[self.algorithm_type_box.currentText()]

    def run_triton_clicked(self):
        self.ast = None  # Reset the AST variable
        self.triton_textarea.clear()  # clear
        self.set_enabled_synthesis_widgets(False)

        if self.is_lookuptable_loaded():
            if self.lookuptable.name != self.table_line.text():
                print("Reload lookup table")
                ret = self.load_lookup_table()
            else:
                ret = True
        else:
            ret = self.load_lookup_table()
        if not ret:
            return

        if self.analysis_type == AnalysisType.QTRACE:
            ret = self.run_triton_qtrace()
        elif self.analysis_type == AnalysisType.FULL_SYMBOLIC:
            ret = self.run_triton_fullsym()
        else:
            assert False

        if ret:
            self.on_triton_finished()

    def load_lookup_table(self) -> bool:
        if not self.table_line.text():
            self.table_line.setStyleSheet("border: 1px solid red")
            return False
        if self.table_type == TableType.HTTP:
            try:
                self.lookuptable = LookupTableREST.load(self.table_line.text())
            except ConnectionAbortedError as e:
                QtWidgets.QMessageBox.critical(self, "Table Loading", f"Error contacting {self.table_line}\n{e}")
                return False
        elif self.table_type == TableType.SQLITE:
            try:
                self.lookuptable = LookupTableDB.load(self.table_line.text())
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Table Loading", f"Error when loading database: {e}")
                return False
        return True

    def line_to_qtrace_inst(self, widget):
        if not widget.hasAcceptableInput():
            widget.setStyleSheet("border: 1px solid red")
            return None
        else:
            addr = int(widget.text(), 16)
            inst = self.trace.get_first_instr_at_addr(addr)
            if inst is None:
                QtWidgets.QMessageBox.critical(self, "Invalid parameter", f"No instruction in trace at address: {addr:#x}")
                return None
            else:
                return inst

    def run_triton_qtrace(self) -> bool:
        # Make sure a trace is loaded before proceeding
        if self.trace is None:
            QtWidgets.QMessageBox.critical(self, "No trace", f"A trace should be loaded first ")
            return False

        from_inst = self.line_to_qtrace_inst(self.from_line)
        if from_inst is None:
            return False
        to_inst = self.line_to_qtrace_inst(self.to_line)
        if to_inst is None:
            return False
        if to_inst.id < from_inst.id:
            QtWidgets.QMessageBox.critical(self, "Invalid order", f"Instruction {from_inst.addr:#x}{from_inst} higher in the trace than {to_inst.addr:#x}{to_inst} ({from_inst.id} > {to_inst.id})")
            return False

        if self.target_type == TargetType.MEMORY:
            if not self.mem_line.hasAcceptableInput():
                self.mem_line.setStyleSheet("border: 1px solid red")
                return False
            else:
                mem_addr = int(self.mem_line.text(), 16)

        # Do not execute last instruction if operand & read operand else always execute last instruction
        off = 0 if self.target_type == TargetType.OPERAND and self.op_is_read else 1

        # At this point we are sur to have valid parameters

        from qsynthesis.utils.qtrace_symexec import QtraceSymExec, Mode
        m = Mode[self.qtrace_sym_type.name]
        self.symexec = QtraceSymExec(self.trace, m)
        self.symexec.initialize_register(self.arch.INS_PTR.name, from_inst.INS_PTR)
        self.symexec.initialize_register(self.arch.STK_PTR.name, from_inst.STK_PTR)
        self.symexec.process_instr_sequence(from_inst.id, to_inst.id+off)
        if self.target_type == TargetType.REG:
            self.ast = self.symexec.get_register_ast(self.register_box.currentText())
        elif self.target_type == TargetType.MEMORY:
            self.ast = self.symexec.get_memory_ast(mem_addr, int(self.lookuptable.bitsize/8))
        elif self.target_type == TargetType.OPERAND:
            if self.op_is_read: # if read disassemble only the last instruction
                inst = self.symexec.disassemble(to_inst.opcode, to_inst.addr)
                self.ast = self.symexec.get_operand_ast(self.op_num, inst)
            else:  # operand is write can directly get its ast
                self.ast = self.symexec.get_operand_ast(self.op_num)
        else:
            assert False
        self.symexec.ctx.clearCallbacks()  # Fix the bug from space / can also be fixed by making self.symexec object attribute
        return True

    def run_triton_fullsym(self) -> bool:
        if Processor.type == ProcessorType.UNKNOWN:
            QtWidgets.QMessageBox.critical(self, "Unsupported arch", f"Architecture {Processor.name} is not supported by Triton")
            return False
        if not self.from_line.hasAcceptableInput():
            self.from_line.setStyleSheet("border: 1px solid red")
            return False
        if not self.to_line.hasAcceptableInput():
            self.to_line.setStyleSheet("border: 1px solid red")
            return False
        cur_addr = int(self.from_line.text(), 16)
        end_addr = int(self.to_line.text(), 16)
        if end_addr <= cur_addr:
            QtWidgets.QMessageBox.critical(self, "Invalid order", f"From {cur_addr:#x} must be lower than {end_addr:#x}")
            return False
        if not ida_bytes.is_mapped(cur_addr) or not ida_bytes.is_mapped(end_addr):
            QtWidgets.QMessageBox.critical(self, "Invalid address",
                                           f"From: {cur_addr:#x} or To: {end_addr:#x} is not mapped in memory")
            return False

        if self.target_type == TargetType.MEMORY:
            if not self.mem_line.hasAcceptableInput():
                self.mem_line.setStyleSheet("border: 1px solid red")
                return False
            else:
                mem_addr = int(self.mem_line.text(), 16)

        if self.target_type == TargetType.OPERAND and self.op_is_read:
            stop_addr = end_addr  # Do not execute last address
        else:
            stop_addr = ida_bytes.get_item_end(end_addr)  # Do execute 'to' address
        # At this point we are sur to have valid parameters

        # Create the purely symbolic executor
        self.symexec = SimpleSymExec(processor_to_triton_arch())

        self.symexec.initialize_register(self.arch.INS_PTR.name, cur_addr)
        self.symexec.initialize_register(self.arch.STK_PTR.name, 0x800000)

        # Execute the range of instructions
        while cur_addr < stop_addr:  # Retrieve directly bytes from IDA
            if ida_bytes.is_code(ida_bytes.get_flags(cur_addr)):
                opc = ida_bytes.get_bytes(cur_addr, ida_bytes.get_item_size(cur_addr))
                if not self.symexec.execute(opc):
                    QtWidgets.QMessageBox.critical(self, "Symbolic Execution Error", f"Instruction at address {cur_addr:#x} seems unsupported by Triton")
                    return False
            else:
                QtWidgets.QMessageBox.critical(self, "Invalid byte", f"Stop on address: {cur_addr:#x} which is not code")
                return False
            cur_addr = ida_bytes.next_head(cur_addr, stop_addr)

        if self.target_type == TargetType.REG:
            self.ast = self.symexec.get_register_ast(self.register_box.currentText())
        elif self.target_type == TargetType.MEMORY:
            self.ast = self.symexec.get_memory_ast(mem_addr, int(self.lookuptable.bitsize/8))
        elif self.target_type == TargetType.OPERAND:
            if self.op_is_read: # if read disassemble only the last instruction
                opc = ida_bytes.get_bytes(stop_addr, ida_bytes.get_item_size(stop_addr))
                inst = self.symexec.disassemble(opc, stop_addr)
                self.ast = self.symexec.get_operand_ast(self.op_num, inst)
            else:  # operand is write can directly get its ast
                self.ast = self.symexec.get_operand_ast(self.op_num)
        else:
            assert False

        self.symexec.ctx.clearCallbacks()
        return True

    def on_triton_finished(self):
        # Enable Triton fields
        self.triton_textarea.setEnabled(True)
        if IDA_ENABLED:
            self.show_deps_triton_button.setEnabled(True)
        if QTRACEIDA_ENABLED:
            self.show_ast_triton_button.setEnabled(True)

        # Fill the text area with the results
        self.fill_triton_results()

        # Enable all buttons related to synthesis
        self.set_enabled_synthesis_widgets(True)

    def fill_triton_results(self):
        self.triton_textarea.clear()
        self.triton_textarea.append("Inputs:")
        for symvars in self.ast.symvars:
            self.triton_textarea.append(f"  - {symvars}")
        self.triton_textarea.append(f"Node count: {self.ast.node_count}")
        self.triton_textarea.append(f"Depth: {self.ast.depth}")
        pass

    def triton_show_deps_clicked(self):
        st = ShowDepState(self.show_deps_triton_button.text())
        if st == ShowDepState.SHOW:
            if self.target_type == TargetType.REG:
                sym = self.symexec.get_register_symbolic_expression(self.register_box.currentText())
            elif self.target_type == TargetType.MEMORY:
                # FIXME: Sanitize memory value before using it
                mem_addr = int(self.mem_line.text(), 16)
                sym = self.symexec.get_memory_symbolic_expression(mem_addr, int(self.lookuptable.bitsize/8))
            elif self.target_type == TargetType.OPERAND:
                sym = self.symexec.get_operand_symbolic_expression(self.op_num)
            else:
                assert False

            # Instanciate the slicer
            sl = Slicer(self.trace)

            # Call the backslice with existing context and symbolic expression
            dg = sl._backslice(self.symexec.ctx, sym)

            # Iterate all addresses
            for addr in Slicer.to_address_set(dg):
                back = ida_nalt.get_item_color(addr)
                self.highlighted_addr[addr] = back
                ida_nalt.set_item_color(addr, 0xA1F7A1)

        else:
            for addr, color in self.highlighted_addr.items():
                ida_nalt.set_item_color(addr, color)
            self.highlighted_addr.clear()

        # Switch state
        self.show_deps_triton_button.setText(ShowDepState.HIDE.value if st == ShowDepState.SHOW else ShowDepState.SHOW.value)

    def triton_show_ast_clicked(self):
        viewer = AstViewer("Triton AST", self.ast)
        viewer.Show()

    def run_synthesis_clicked(self):
        if self.algorithm == AlgorithmType.TOPDOWN:
            synthesizer = TopDownSynthesizer(self.lookuptable)
        else:
            synthesizer = PlaceHolderSynthesizer(self.lookuptable)
        self.synth_ast, _ = synthesizer.synthesize(self.ast)

        self.on_synthesis_finished()

    def on_synthesis_finished(self):
        # Enable Synthesis fields
        self.synthesis_textarea.setEnabled(True)
        if QTRACEIDA_ENABLED:
            self.show_ast_synthesis_button.setEnabled(True)
        if IDA_ENABLED:
            self.reassemble_button.setEnabled(True)

        # Fill the text area with the results
        self.fill_synthesis_results()

    def fill_synthesis_results(self):
        self.synthesis_textarea.clear()
        self.synthesis_textarea.append(f"simplified: {self.ast.node_count > self.synth_ast.node_count}")
        self.synthesis_textarea.append(f"synthesized expression: {self.synth_ast.pp_str if self.synth_ast.node_count < 15 else 'too large'}")

        self.synthesis_textarea.append(f"Node count: {self.synth_ast.node_count}")
        self.synthesis_textarea.append(f"Depth: {self.synth_ast.depth}")

        self.synthesis_textarea.append(f"Scale reduction: {self.synth_ast.node_count / self.ast.node_count:.2f}")

    def synthesis_show_ast_clicked(self):
        viewer = AstViewer("Synthesized AST", self.synth_ast)
        viewer.Show()

    def reassemble_clicked(self):
        print("Not implemented yet")
        # TODO: to implement

    def set_enabled_synthesis_widgets(self, val):
        self.run_synthesis_button.setEnabled(val)
        self.synthesis_textarea.setEnabled(val)
        if not val:  # In case we disable everything
            self.synthesis_textarea.clear()  # Clear synthesis view
            self.show_ast_synthesis_button.setEnabled(val)
            self.reassemble_button.setEnabled(val)


    # =====================  Trace Database related fields  ======================
    @property
    def trace_type(self) -> TraceDbType:
        return TraceDbType(self.trace_type_box.currentIndex())

    def trace_type_changed(self, idx):
        self.trace_line.setText("")

    def customfocusInEventTraceLine(self, event):
        self.trace_line.setText("")
        if self.trace_type == TraceDbType.CONFIG:
            self._dbm = DatabaseManager.from_qtracedb_config()
            t_name = self.get_trace_from_db()
            if t_name:
                self._trace = self._dbm.get_trace(t_name)
                self._arch = self._trace.get_arch()
                self.trace_line.setText(t_name)
                self.on_trace_opened(self._trace)
        elif self.trace_type == TraceDbType.SQLITE:
            filename = QtWidgets.QFileDialog.getOpenFileName()[0]
            filepath = Path(filename)
            if filepath.exists() and filepath.is_file():
                try:
                    self._dbm = DatabaseManager(f'sqlite:///{filepath.absolute()}')
                    self._trace = self._dbm.get_trace("x86_64")
                    self._arch = self._trace.get_arch()
                    self.trace_line.setText(str(filepath))
                    self.on_trace_opened(self._trace)
                except Exception:
                    print(f"Invalid database: {filepath}")
            else:
                print(f"Invalid file: {filepath}")
        self.trace_line.focusNextChild()

    def get_trace_from_db(self):
        dlg = QtWidgets.QDialog(parent=self)
        dlg.setWindowTitle('Qtrace-DB connection')
        dlg.setObjectName("Dialog")

        self.verticalLayout = QtWidgets.QVBoxLayout(dlg)
        self.verticalLayout.setObjectName("verticalLayout")
        self.comboBox = QtWidgets.QComboBox(dlg)
        self.comboBox.setObjectName("comboBox")
        self.verticalLayout.addWidget(self.comboBox)
        self.buttonBox = QtWidgets.QDialogButtonBox(dlg)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        # Fill list of traces
        self.comboBox.addItems(self._dbm.list_traces())

        self.buttonBox.accepted.connect(dlg.accept)
        self.buttonBox.rejected.connect(dlg.reject)

        dlg.exec()

        if dlg.result():
            return self.comboBox.currentText()
        else:
            return None
    # ======================================================================
