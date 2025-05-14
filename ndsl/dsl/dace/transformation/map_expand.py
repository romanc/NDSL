from dace import dtypes, subsets
from dace.sdfg import nodes
from dace.sdfg.analysis.schedule_tree import treenodes as tn


class MapExpansion(tn.ScheduleNodeTransformer):
    """Expands N-dimensional maps into N one-dimensional maps."""

    def __init__(
        self, *, inner_schedule: dtypes.ScheduleType = dtypes.ScheduleType.Sequential
    ):
        self._inner_schedule = inner_schedule

    def visit_MapScope(self, scope: tn.MapScope) -> tn.MapScope:
        map = scope.node.map
        dimension = map.get_param_num()

        # Nothing to do if this is already a one-dimensional map.
        if dimension < 2:
            return self.generic_visit(scope)

        # Setup new Maps
        new_maps: list[nodes.Map] = []
        current_label = map.label
        for index, param, param_range, tile_size in zip(
            range(dimension), map.params, map.range.ranges, map.range.tile_sizes
        ):
            new_maps.append(
                nodes.Map(
                    f"{current_label}{param}",
                    [param],
                    subsets.Range([(*param_range, tile_size)]),
                    schedule=map.schedule if index == 0 else self._inner_schedule,
                )
            )

        # Create new scopes
        children = self.visit(scope.children)
        for new_map in reversed(new_maps):
            new_scope = tn.MapScope(children=children, node=nodes.MapEntry(new_map))
            for child in children:
                child.parent = new_scope
            children = [new_scope]

        new_scope.parent = scope.parent
        return new_scope
