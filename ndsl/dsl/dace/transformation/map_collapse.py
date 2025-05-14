from dace.sdfg.analysis.schedule_tree import treenodes as tn


class MapCollapse(tn.ScheduleNodeTransformer):
    """Collapses nested maps into multi-dimensional maps."""

    def visit_MapScope(self, scope: tn.MapScope) -> tn.MapScope:

        # Collapse map scopes as long as the only child is another MapScope
        while len(scope.children) == 1 and isinstance(scope.children[0], tn.MapScope):
            current_map = scope.node.map
            child = scope.children[0]
            child_map = child.node.map

            # merge maps
            current_map.params.append(*child_map.params)
            current_map.range.ranges.append(*child_map.range.ranges)
            current_map.range.tile_sizes.append(*child_map.range.tile_sizes)
            suffix = "".join([param for param in child_map.params])
            current_map.label = f"{current_map.label}{suffix}"

            # reconnect children and fix parents
            scope.children = child.children
            for child in scope.children:
                child.parent = scope

        return self.generic_visit(scope)
