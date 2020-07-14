from enum import Enum
from PyQt5 import QtWidgets, QtCore, QtGui
from pathlib import Path
from typing import Tuple, Optional, List

from qsynthesis.plugin.dependencies import ida_kernwin, IDA_ENABLED, QTRACEIDA_ENABLED, QTRACEDB_ENABLED
from qsynthesis.plugin.dependencies import ida_bytes, ida_nalt, ida_ua, ida_funcs, ida_gdl, ida_loader, ida_auto
from qsynthesis.tables import LookupTableDB, LookupTableREST
from qsynthesis.algorithms import TopDownSynthesizer, PlaceHolderSynthesizer
from qsynthesis.utils.symexec import SimpleSymExec
from qsynthesis.tritonast import ReassemblyError
from qsynthesis.plugin.processor import processor_to_triton_arch, processor_to_qtracedb_arch, Arch, Processor, ProcessorType
from qsynthesis.plugin.ast_viewer import AstViewer, BasicBlockViewer
from qsynthesis.plugin.popup_actions import SynthetizeFromHere, SynthetizeToHere, SynthetizeOperand
from qtraceanalysis.slicing import Slicer
from qtracedb import DatabaseManager
from qtracedb.trace import Trace
from qtracedb.archs import ArchsManager
from .ui.synthesis_ui import Ui_synthesis_view

TEMPLATE_TRITON = '''<!DOCTYPE html>
<html lang="en">
<head>
    <style>
        body { font-size: large; }
        big { font-size: x-large; }
        td { padding: 5px; padding-left:10px; padding-right:10px; }
    </style>
</head>
<body>
    <center><table>
        <tr><td>Node count</td><td>Depth</td></tr>
        <tr><td align="center"><big><b>%d</b></big></td><td align="center"><big><b>%d</b></big></td></tr>
    </table></center>
    <hr/>
    <center>Inputs
    <table border style="border-style:dashed; margin-top:8px;" >
        %s
    </table>
    </center>
</body>
</html>
'''

TEMPLATE_SYNTHESIS = '''<!DOCTYPE html>
<html lang="en">
<head>
    <style>
        body { font-size: large; }
        big { font-size: x-large; }
        td { padding: 5px; padding-left:10px; padding-right:10px; }
    </style>
</head>
<body>
    <center>
        Simplified: <big style="color:%s;">%s</big><br/>
        Synthesiszed Expression<br/>
        %s
    </center>
    <hr/>
    <center><table>
        <tr><td>Node count</td><td>Depth</td><td>Scale</td></tr>
        <tr>
            <td align="center"><big><b>%d</b></big></td>
            <td align="center"><big><b>%d</b></big></td>
            <td align="center"><big><b>%.2f%c</b></big></td>
        </tr>
    </table></center>

    
</body>
</html>
'''



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
        self.stop_addr = None  # Address where to analysis end
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

        # Set the stop addr (for operand reassembly)
        self.stop_addr = to_inst.addr

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
            self.stop_addr = end_addr  # Do not execute last address
        else:
            self.stop_addr = ida_bytes.get_item_end(end_addr)  # Do execute 'to' address
        # At this point we are sur to have valid parameters

        # Create the purely symbolic executor
        self.symexec = SimpleSymExec(processor_to_triton_arch())

        self.symexec.initialize_register(self.arch.INS_PTR.name, cur_addr)
        self.symexec.initialize_register(self.arch.STK_PTR.name, 0x800000)

        # Execute the range of instructions
        while cur_addr < self.stop_addr:  # Retrieve directly bytes from IDA
            if ida_bytes.is_code(ida_bytes.get_flags(cur_addr)):
                opc = ida_bytes.get_bytes(cur_addr, ida_bytes.get_item_size(cur_addr))
                if not self.symexec.execute(opc):
                    QtWidgets.QMessageBox.critical(self, "Symbolic Execution Error", f"Instruction at address {cur_addr:#x} seems unsupported by Triton")
                    return False
            else:
                QtWidgets.QMessageBox.critical(self, "Invalid byte", f"Stop on address: {cur_addr:#x} which is not code")
                return False
            cur_addr = ida_bytes.next_head(cur_addr, self.stop_addr)

        if self.target_type == TargetType.REG:
            self.ast = self.symexec.get_register_ast(self.register_box.currentText())
        elif self.target_type == TargetType.MEMORY:
            self.ast = self.symexec.get_memory_ast(mem_addr, int(self.lookuptable.bitsize/8))
        elif self.target_type == TargetType.OPERAND:
            if self.op_is_read: # if read disassemble only the last instruction
                opc = ida_bytes.get_bytes(self.stop_addr, ida_bytes.get_item_size(self.stop_addr))
                inst = self.symexec.disassemble(opc, self.stop_addr)
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
        if self.ast.symvars:
            tpl = '<tr><td align="center">%s</td><td align="center">%d</td></tr>'
            inp_s = "\n".join(tpl % (x.getAlias(), x.getBitSize()) for x in self.ast.symvars)
        else:
            inp_s = "<big><b>0</b></big>"
        self.triton_textarea.setHtml(TEMPLATE_TRITON % (self.ast.node_count, self.ast.depth, inp_s))

    def triton_show_deps_clicked(self):
        st = ShowDepState(self.show_deps_triton_button.text())
        if st == ShowDepState.SHOW:
            addrs = self.get_dependency_addresses()
            for addr in addrs:
                back = ida_nalt.get_item_color(addr)
                self.highlighted_addr[addr] = back
                ida_nalt.set_item_color(addr, 0xA1F7A1)

        else:
            for addr, color in self.highlighted_addr.items():
                ida_nalt.set_item_color(addr, color)
            self.highlighted_addr.clear()

        # Switch state
        self.show_deps_triton_button.setText(ShowDepState.HIDE.value if st == ShowDepState.SHOW else ShowDepState.SHOW.value)

    def get_dependency_addresses(self) -> List[int]:
        if self.target_type == TargetType.REG:
            sym = self.symexec.get_register_symbolic_expression(self.register_box.currentText())
        elif self.target_type == TargetType.MEMORY:
            # FIXME: Sanitize memory value before using it
            mem_addr = int(self.mem_line.text(), 16)
            sym = self.symexec.get_memory_symbolic_expression(mem_addr, int(self.lookuptable.bitsize / 8))
        elif self.target_type == TargetType.OPERAND:
            sym = self.symexec.get_operand_symbolic_expression(self.op_num)
        else:
            assert False

        # Instanciate the slicer
        sl = Slicer(self.trace)

        # Call the backslice with existing context and symbolic expression
        dg = sl._backslice(self.symexec.ctx, sym)

        # Iterate all addresses
        return sorted(Slicer.to_address_set(dg))

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
        color, simp = ("green", "Yes") if self.ast.node_count > self.synth_ast.node_count else ("red", "No")
        ssexpr = "<small>(too large)</small>" if self.synth_ast.node_count > 12 else f"<big><b>{self.synth_ast.pp_str}</b></big>"
        scale = -(((self.ast.node_count - self.synth_ast.node_count) * 100) / self.ast.node_count)
        self.synthesis_textarea.setHtml(TEMPLATE_SYNTHESIS % (color, simp, ssexpr, self.synth_ast.node_count, self.synth_ast.depth, scale, '%'))

    def synthesis_show_ast_clicked(self):
        viewer = AstViewer("Synthesized AST", self.synth_ast)
        viewer.Show()

    def set_enabled_synthesis_widgets(self, val):
        self.run_synthesis_button.setEnabled(val)
        self.synthesis_textarea.setEnabled(val)
        if not val:  # In case we disable everything
            self.synthesis_textarea.clear()  # Clear synthesis view
            self.show_ast_synthesis_button.setEnabled(val)
            self.reassemble_button.setEnabled(val)

    def reassemble_clicked(self):
        is_reg, reg = self.selected_expr_register()
        res = self.get_reassembly_options(ask_reg=not is_reg)
        if res:
            rg, patch_fun, shrink_fun, snap = res
            dst_reg = reg.lower() if is_reg else rg.lower()
            try:
                if patch_fun:
                    addrs = self.get_dependency_addresses()
                    asm_bytes = self.synth_ast.reassemble(dst_reg, self.arch)
                    if snap:  # Create a snapshot of the database
                        ss = ida_loader.snapshot_t()
                        ss.desc = f"Reassembly of {dst_reg} at {self.stop_addr:#x}"
                        ida_kernwin.take_database_snapshot(ss)
                    if shrink_fun:
                        self.patch_and_shrink_reassembly(addrs, asm_bytes)
                    else:
                        self.patch_reassembly(addrs, asm_bytes)
                else:
                    # Reassemble to instruction and show it in a View
                    insts = self.synth_ast.reassemble_to_insts(dst_reg, self.arch)
                    bb_viewer = BasicBlockViewer("Reassembly", insts)
                    bb_viewer.Show()
            except ReassemblyError as e:
                QtWidgets.QMessageBox.critical(self, "Reassembly error", f"Error: {e}")
        else:
            pass # Do nothing user click canceled on options

    def coallesce_addrs(self, addrs):
        blocks = []
        for addr in addrs:
            sz = ida_bytes.get_item_size(addr)
            if not blocks:
                blocks.append((addr, sz))
            else:
                b_addr, b_sz = blocks[-1]
                if b_addr + b_sz == addr:  # We are next to the previous item
                    blocks[-1] = (b_addr, b_sz+sz)
                else:  # The address is not contiguous
                    blocks.append((addr, sz))
        return blocks

    def patch_reassembly(self, addrs, asm) -> None:
        nop = self.arch.nop_instruction
        insts = self.arch.disasm(asm, 0x0)
        blocks = self.coallesce_addrs(addrs)
        if sum(x[1] for x in blocks) < len(asm):
            QtWidgets.QMessageBox.critical(self, "Reassembly error", f"No enough place to push back reassembled instructions")
            return

        block_addr, block_sz = blocks.pop()
        for i in insts[::-1]:
            data = i.bytes
            while len(data) > block_sz:  # While blocks are too small fill them with nop
                if block_sz != 0:
                    ida_bytes.patch_bytes(block_addr, nop * block_sz)  # Fill space of the block with NOPs
                if blocks:
                    block_addr, block_sz = blocks.pop()
                else:
                    QtWidgets.QMessageBox.critical(self, "Reassembly error", "No slot to place instruction remaining, abort")
                    return

            # Here there is meant to be enough space to place instruction
            p_addr = block_addr + block_sz - len(data)  # compute address where to patch within block
            self.safe_patch_instruction(p_addr, data)
            block_sz -= len(data)

        # All instructions have been put in blocks, fill the remaining ones with NOPs
        while blocks or block_sz:
            if block_sz:
                ida_bytes.patch_bytes(block_addr, nop * block_sz)  # Fill space of the block with NOPs
                block_sz = 0
            if blocks:
                block_addr, block_sz = blocks.pop()

    @staticmethod
    def safe_patch_instruction(ea, data):
        ida_bytes.patch_bytes(ea, data)  # Patch bytes
        if not ida_bytes.is_code(ida_bytes.get_flags(ea)):  # Check that the address is now a code instruction, if not:
            ida_bytes.del_items(ea, 0, len(data))   # Del all types
            ida_ua.create_insn(ea)                  # Redecode instruction
        if not ida_bytes.is_code(ida_bytes.get_flags(ea)):
            print("Really can't create instruction at that location")

    @staticmethod
    def safe_patch_instruction_block(ea, data):
        print(f"safe_patch_instruction_block {ea:#x}: {data}")
        ida_bytes.patch_bytes(ea, data)  # Patch bytes
        ida_bytes.del_items(ea, 0, len(data))
        ida_ua.create_insn(ea)

    def patch_and_shrink_reassembly(self, addrs, asm) -> None:
        init_addr = addrs[0]
        f = ida_funcs.get_func(init_addr)
        g = ida_gdl.FlowChart(f)
        low, high = init_addr, addrs[-1]
        block = None
        for bb in g:
            if bb.start_ea <= low < bb.end_ea and bb.start_ea <= high <= bb.end_ea:
                block = bb
        if block is None:
            QtWidgets.QMessageBox.critical(self, "Reassembly error", "Dependency slice have to be in the same basic block")
            return

        # Scan all items
        cur_addr = init_addr
        payload = b""
        while cur_addr < block.end_ea:  # Iterate the whole basicblock
            if cur_addr >= self.stop_addr and asm:  #
                payload += asm
                asm = None
            if cur_addr not in addrs:  # Instruction not in dependency so keep it
                sz = ida_bytes.get_item_size(cur_addr)
                payload += ida_bytes.get_bytes(cur_addr, sz)
            cur_addr = ida_bytes.next_head(cur_addr, block.end_ea)

        # Perform the final patching
        self.safe_patch_instruction_block(init_addr, payload)

        # If the block was the last of the function (adjust the end of the function)
        if f.end_ea == block.end_ea:
            print(f"adjust of the function to {init_addr+len(payload):#x}")
            ida_funcs.set_func_end(init_addr, init_addr+len(payload))

    def get_reassembly_options(self, ask_reg=False):
        dlg = QtWidgets.QDialog(parent=None)
        dlg.setWindowTitle('Reassembly options')

        verticalLayout = QtWidgets.QVBoxLayout(dlg)
        if ask_reg:
            h_layout = QtWidgets.QHBoxLayout()
            label_reg = QtWidgets.QLabel(dlg)
            label_reg.setText("Destination register:")
            h_layout.addWidget(label_reg)
            comboBox = QtWidgets.QComboBox(dlg)
            comboBox.addItems([x.name for x in ArchsManager.get_supported_regs(self.arch)])
            h_layout.addWidget(comboBox)
            verticalLayout.addLayout(h_layout)

        patch_fun = QtWidgets.QCheckBox(dlg)
        patch_fun.setText("patch function bytes")
        verticalLayout.addWidget(patch_fun)

        shrink_fun = QtWidgets.QCheckBox(dlg)
        shrink_fun.setText("shrink function\n move some instruction instead of filling with NOPs.\nCan break disassembly"
                           " for relative instructions. (Works only for linear blocks)")
        shrink_fun.setEnabled(False)
        verticalLayout.addWidget(shrink_fun)

        def patch_checked():
            if patch_fun.isChecked():
                shrink_fun.setEnabled(True)
            else:
                shrink_fun.setEnabled(False)
                shrink_fun.setChecked(False)

        patch_fun.stateChanged.connect(patch_checked)

        snapshot = QtWidgets.QCheckBox(dlg)
        snapshot.setText("Snapshot database before patching")
        verticalLayout.addWidget(snapshot)

        buttonBox = QtWidgets.QDialogButtonBox(dlg)
        buttonBox.setOrientation(QtCore.Qt.Horizontal)
        buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)
        verticalLayout.addWidget(buttonBox)
        buttonBox.accepted.connect(dlg.accept)
        buttonBox.rejected.connect(dlg.reject)

        dlg.exec()

        if dlg.result():
            reg = comboBox.currentText() if ask_reg else None
            return reg, patch_fun.isChecked(), shrink_fun.isChecked(), snapshot.isChecked()
        else:
            return None

    def selected_expr_register(self) -> Tuple[bool, Optional[str]]:
        if self.target_type == TargetType.REG:
            return True, self.register_box.currentText()
        elif self.target_type == TargetType.MEMORY:
            return False, None
        elif self.target_type == TargetType.OPERAND:
            opc = ida_bytes.get_bytes(self.stop_addr, ida_bytes.get_item_size(self.stop_addr))
            inst = self.arch.disasm_one(opc, self.stop_addr)
            op = inst.operands[self.op_num]
            if op.is_register():
                return True, op.register.name
            else:
                return False, None


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

