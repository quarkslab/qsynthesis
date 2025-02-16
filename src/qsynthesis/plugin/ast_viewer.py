# built-in modules
from typing import List

# qsynthesis modules
from qsynthesis.plugin.dependencies import ida_graph, Instr
from qsynthesis.tritonast import TritonAst


class AstViewer(ida_graph.GraphViewer):
    """
    GraphViewer to show AST Triton expressions
    """

    def __init__(self, title: str, ast: TritonAst):
        """
        Constructor.

        :param title: string title of the windows
        :param ast: TritonAst object to show
        """
        ida_graph.GraphViewer.__init__(self, title)
        self._ast = ast

    def OnRefresh(self) -> bool:
        """
        GraphViewer callback called whenever refreshing the view (so at least once)
        """
        self.Clear()
        self.draw()
        return True

    def OnGetText(self, ida_node_id: int) -> str:
        """
        GraphViewer callback whenever it needs to get the text associated to a node.
        Just return the string pre-computed in :meth:`AstViewer.draw`.

        :param ida_node_id: GraphViewer node id
        :return: text of the node
        """
        return self[ida_node_id]

    def Show(self) -> bool:
        """
        Called when showing the view
        """
        # TODO: Checking if its really required ?
        if not ida_graph.GraphViewer.Show(self):
            return False
        return True

    def draw(self) -> None:
        """
        Create the graph by iterating the AST and creating
        associated node in the GraphViewer object. The text
        of each node is their 'symbol' attribute.
        """
        n_id = self.AddNode(self._ast.symbol)
        worklist = [(n_id, self._ast)]

        while worklist:
            node_id, node = worklist.pop(0)

            for c in node.get_children():
                child_id = self.AddNode(c.symbol)
                self.AddEdge(node_id, child_id)
                worklist.append((child_id, c))


class BasicBlockViewer(ida_graph.GraphViewer):
    """
    Class to show handcrafted basic blocks with
    a given set of instructions.
    """
    def __init__(self, title: str, insts: List[Instr]):
        """
        Constructor. Takes window title and a list of instruction
        to show.
        """
        ida_graph.GraphViewer.__init__(self, title)
        self.insts = insts
        # TODO: Using ida_lines stuff to precompute 'colored' lines

    def OnRefresh(self) -> bool:
        """
         GraphViewer callback called whenever refreshing the view (so at least once)
         """
        self.Clear()
        self.draw()
        return True

    def OnGetText(self, ida_node_id: int) -> str:
        """
        GraphViewer callback whenever it needs to get the text associated to a node.
        Just return the string pre-computed in :meth:`BasicBlockViewer.draw`.

        :param ida_node_id: GraphViewer node id
        :return: text of the node
        """
        return self[ida_node_id]

    def Show(self):
        """ Called when showing the view """
        return False if not ida_graph.GraphViewer.Show(self) else True

    def draw(self) -> None:
        """ Add a single basic block corresponding to the concatenation of string instructions """
        self.AddNode("\n".join(str(x) for x in self.insts))
