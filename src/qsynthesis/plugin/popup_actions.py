# third-party modules
from PyQt5 import QtWidgets

# qsynthesis deps
from qsynthesis.plugin.dependencies import ida_kernwin, ida_bytes, ida_ua, ida_lines
from qsynthesis.types import Addr


POPUP_PATH = "QSynthesis/"


class SynthetizeFromHere(ida_kernwin.action_handler_t):
    """
    Popup action that will be shown in the context-menu of the IDA-View (right-click).
    This action just takes the current line address and put it in the 'from' field
    of the Qsynthesis view.
    """

    def __init__(self, widget: 'SynthesizerView'):
        """
        Constructor. Take the QSynthesis main view as parameter.
        """
        ida_kernwin.action_handler_t.__init__(self)
        self.action_id = "Qsynthesis:from-here"
        self.text = "Synthesize from here"
        self.shortcut = "Ctrl+Shift+A"
        self.tooltip = "Start synthesizing from the current address"
        self.icon = 127 # GREEN_LIGHT  # 84  # BLOCK_DESCENT_FRM
        self.widget = widget

    def set_text_widget(self, ea: Addr) -> None:
        """
        Set the address given in parameter in the 'from' field of
        the Qsynthesis view.

        :param ea: address to set
        :return: Nones
        """
        self.widget.from_line.setText(f"{ea:#x}")

    def activate(self, ctx) -> bool:
        """
        Activation callback. Retrieve current address and set the from
        field of the QSynthesis view
        """
        ea = ida_kernwin.get_screen_ea()
        ea = ida_bytes.get_item_head(ea)  # Make sure we are on the head of the instruction
        if ida_bytes.is_code(ida_bytes.get_flags(ea)):
            self.set_text_widget(ea)
            return True
        else:
            QtWidgets.QMessageBox.critical(self.widget, "Invalid byte", f"Current address: {ea:#x} is not code")
            return False

    def update(self, _) -> int:
        """ Overridden method called on hook update """
        return ida_kernwin.AST_ENABLE_ALWAYS

    def register(self) -> bool:
        """
        Register the popup_action to the 'IDA View-A'.

        :return: True if the registration succeeded
        """
        idaview = ida_kernwin.find_widget("IDA View-A")
        if idaview is None:
            print("Can't find IDA View to attach action")
            return False
        else:
            action = ida_kernwin.action_desc_t(self.action_id, self.text, self, self.shortcut, self.tooltip, self.icon)
            res = ida_kernwin.register_action(action)
            res2 = ida_kernwin.attach_action_to_popup(idaview, None, self.action_id, POPUP_PATH)
            return res & res2

    def unregister(self) -> bool:
        """
        Remove popup action from the context menu of the view

        :return: True if unregistration succeeded
        """
        res = ida_kernwin.unregister_action(self.action_id)
        idaview = ida_kernwin.find_widget("IDA View-A")
        if idaview is None:
            print("Can't find IDA View to detach action")
            return False
        else:
            res2 = ida_kernwin.detach_action_from_popup(idaview, self.action_id)
        return res & res2


class SynthetizeToHere(SynthetizeFromHere):
    """
    Popup action that will be shown in the context-menu of the IDA-View (right-click).
    This action just takes the current line address and put it in the 'To' field
    of the Qsynthesis view.
    """

    def __init__(self, widget: 'SynthesizerView'):
        """
        Constructor. Take the QSynthesis main view as parameter.
        """
        super(SynthetizeToHere, self).__init__(widget)
        self.action_id = "Qsynthesis:to-here"
        self.text = "Synthesize up to here (included)"
        self.shortcut = "Ctrl+Shift+Z"
        self.tooltip = "Start synthesizing up to the current address (included)"
        self.icon = 120 # BREAKPOINT_PLUS

    def set_text_widget(self, ea: Addr) -> None:
        """
        Set the address given in parameter in the 'to' field of the Qsynthesis view.

        :param ea: address to set
        :return: None
        """
        self.widget.to_line.setText(f"{ea:#x}")


class SynthetizeOperand(SynthetizeFromHere):
    """
    Popup action that will be shown in the context-menu of the IDA-View
    and retrieve the current operand being selected. At the moment only
    full operand can be selected (e.g mov rax, [rbp + 4], 'rbp cannot
    be selected separately)
    """

    def __init__(self, widget: 'SynthesizerView'):
        """
        Constructor. Take the QSynthesis main view as parameter.
        """
        super(SynthetizeOperand, self).__init__(widget)
        self.action_id = "Qsynthesis:operand"
        self.text = "Synthesize operand"
        self.shortcut = "Ctrl+Shift+O"
        self.tooltip = "Synthesize the current operand at this address"
        self.icon = 12  # STAR_PLUS

    def activate(self, ctx) -> bool:
        """
        Upon clicking (or shortcut key typed) retrieve the current operand
        object and set various widget variable to be able to retrieve the
        information later on.

        :return: True if the operand is value and has successfully been retrieved
        """
        ea = ida_kernwin. get_screen_ea()
        ea = ida_bytes.get_item_head(ea)  # Make sure we are on the head of the instruction
        if ida_bytes.is_code(ida_bytes.get_flags(ea)):
            op_num = ida_kernwin.get_opnum()
            i = self.widget.get_instruction(ea)
            if i is None:
                QtWidgets.QMessageBox.critical(self.widget, "Invalid instruction", f"Cannot disassembl instruction at: {ea:#x}")
                return False
            op = i.operands[op_num]
            if op.is_register() or op.is_memory():
                self.widget.to_line.setText(f"{ea:#x}")
                self.widget.op_num = op_num
                self.widget.op_is_read = not op.is_written()

                # Switch to operand mode and set text
                self.widget.switch_to_target_operand()
                text = ida_lines.tag_remove(ida_ua.print_operand(ea, op_num))
                self.widget.operand_label.setText(f"{op_num}:{text} ({'write' if op.is_written() else 'read'})")
                return True
            else:
                QtWidgets.QMessageBox.critical(self.widget, "Invalid operand", f"Invalid operand type: {op.type}\nCannot synthesize such type")
                return False
        else:
            QtWidgets.QMessageBox.critical(self.widget, "Invalid byte", f"Current address: {ea:#x} is not code")
            return False
