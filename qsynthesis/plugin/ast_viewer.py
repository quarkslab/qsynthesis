from qsynthesis.plugin.dependencies import ida_graph


class AstViewer(ida_graph.GraphViewer):

    def __init__(self, title, ast):
        ida_graph.GraphViewer.__init__(self, title)
        self._ast = ast

    def OnRefresh(self):
        self.Clear()
        self.draw()
        return True

    def OnGetText(self, ida_node_id):
        return self[ida_node_id]

    def Show(self):
        # TODO: Checking if its really required ?
        if not ida_graph.GraphViewer.Show(self):
            return False
        return True

    def draw(self):
        """
        Create the graph by iterating the AST
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
    def __init__(self, title, insts):
        ida_graph.GraphViewer.__init__(self, title)
        self.insts = insts
        # TODO: Using ida_lines stuff to precompute 'colored' lines

    def OnRefresh(self):
        self.Clear()
        self.draw()
        return True

    def OnGetText(self, ida_node_id):
        return self[ida_node_id]

    def Show(self):
        return False if not ida_graph.GraphViewer.Show(self) else True

    def draw(self):
        """ Add a single basic block corresponding to the instructions """
        self.AddNode("\n".join(str(x) for x in self.insts))
