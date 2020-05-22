from typing import List, NewType, TypeVar

Bit = int
Var = str


class Grammar(object):
    """
    Abstract Grammar class
    """

    def __init__(self):
        self.size: Bit = 0         # Has to be defined
        self.vars: List[Var] = []  # Has to be defined
        self.ops = None   # Has to be defined

    @property
    def non_terminal_operators(self) -> List:
        return self.ops

    def gen_test_inputs(self, inputs):
        raise NotImplementedError("gen_test_inputs abstract method, should be implemented by child class")

    def str_to_expr(self, v, *args):
        raise NotImplementedError("str_to_expr abstract method, should be implemented by child class")

    def to_dict(self):
        raise NotImplementedError("to_dict abstract method, should be implemented by child class")

    @staticmethod
    def from_dict(self):
        raise NotImplementedError("from_dict abstract method, should be implemented by child class")

    @staticmethod
    def dump_inputs(inputs):
        raise NotImplementedError("dump_inputs abstract method, should be implemented by child class")

    @staticmethod
    def load_inputs(dict_inputs):
        raise NotImplementedError("load_inputs abstract method, should be implemented by child class")
