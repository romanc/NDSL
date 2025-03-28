from dace.sdfg.analysis.schedule_tree import treenodes as tn


class ReorderKLoop(tn.ScheduleNodeVisitor):
    def _move_k_map_loop(self, k_map: tn.MapScope | tn.ForScope) -> None:
        """Move k-{map, loop} out, one level at a time, as far as possible."""
        while isinstance(k_map.parent, tn.MapScope):
            parent = k_map.parent
            grand_parent = parent.parent

            parent.children = k_map.children
            parent.parent = k_map

            k_map.parent = grand_parent
            k_map.children = [parent]

            if not isinstance(grand_parent, tn.MapScope):
                # Assumption: We are at the top (tree root) now
                #
                # insert k_map before parent, then delete parent
                # from the list of children in grand_parent
                parent_index = grand_parent.children.index(parent)
                grand_parent.children.insert(parent_index, k_map)
                grand_parent.children.remove(parent)
                return

            assert grand_parent is not None
            grand_parent.children = [k_map]

    def visit_MapScope(self, map_scope: tn.MapScope) -> None:
        map_parameter = map_scope.node.params
        # detect k-map
        if len(map_parameter) == 1 and map_parameter[0] == "__k":
            # attempt to move it out as far as possible
            return self._move_k_map_loop(map_scope)

        # visit children
        for child in map_scope.children:
            self.visit(child)

    def visit_ForScope(self, node: tn.ForScope) -> None:
        # detect k-loop
        if node.header.itervar == "__k":
            # attempt to move it out as far as possible
            return self._move_k_map_loop(node)

        # visit children
        for child in node.children:
            self.visit(child)
