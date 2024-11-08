from enum import Enum
from typing import Any, List
from dace.properties import CodeBlock
from dace.sdfg.analysis.schedule_tree.sdfg_to_tree import as_schedule_tree
from gt4py.cartesian.gtscript import (
    computation,
    interval,
    PARALLEL,
    FORWARD
)
from ndsl.boilerplate import get_factories_single_tile_orchestrated_cpu
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.typing import FloatField
from ndsl import StencilFactory, orchestrate
import numpy as np
import dace
import dace.sdfg.analysis.schedule_tree.treenodes as dace_stree

domain = (3, 3, 4)

stencil_factory, _ = get_factories_single_tile_orchestrated_cpu(
    domain[0], domain[1], domain[2], 0
)

# simple case (working)
def double_map(in_field: FloatField, out_field: FloatField):
    with computation(PARALLEL), interval(...):
        out_field = in_field

    with computation(PARALLEL), interval(...):
        out_field = in_field * 3

# simple case of force-mergeable intervals (working)
def double_map_with_different_intervals(in_field: FloatField, out_field: FloatField):
    with computation(PARALLEL), interval(1, None):
        out_field = in_field

    with computation(PARALLEL), interval(...):
        out_field = in_field * 3

# this should not merge (size doesn't match)
# somehow this throws an error
# def double_map(in_field: FloatField, out_field: FloatField):
#     with computation(PARALLEL), interval(...):
#         tmp = in_field * 3
# 
#     with computation(PARALLEL), interval(...):
#         out_field = tmp[1, 0, 0] + in_field

# we can't merge anything here (working)
def loop_and_map(in_field: FloatField, out_field: FloatField):
    with computation(FORWARD), interval(...):
        out_field = in_field

    with computation(PARALLEL), interval(...):
        out_field = in_field * 3

# TODO broken - weird shit is happening here ...
# merging K loops with over-computation
def mergeable_preserve_order(in_field: FloatField, out_field: FloatField):
    with computation(PARALLEL), interval(1, None):
        out_field = in_field

    with computation(PARALLEL), interval(...):
        out_field = in_field * 3

    with computation(PARALLEL), interval(1, None):
        out_field = in_field * 4

# TODO: borken - weird shit is happening here ...
# merging IJ - keeping K loops
def not_mergeable_k_dependency(in_field: FloatField, out_field: FloatField):
    with computation(PARALLEL), interval(1, None):
        out_field = in_field

    with computation(PARALLEL), interval(...):
        out_field = in_field * 3

    with computation(PARALLEL), interval(1, None):
        in_field = out_field

class DaCeGT4Py_Bridge:
    def __init__(self, stencil_factory: StencilFactory, function: Any):
        orchestrate(obj=self, config=stencil_factory.config.dace_config)
        self.stencil = stencil_factory.from_dims_halo(
            func=function,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )

    def __call__(self, in_field: FloatField, out_field: FloatField):
        self.stencil(in_field, out_field)

def is_k_map(node: dace_stree.MapScope) -> bool:
    """ Returns true iff node is a map over K. """
    map_parameter = node.node.params
    return len(map_parameter) == 1 and map_parameter[0] == "__k"

def execution_condition(range: dace.subsets.Range) -> CodeBlock:
    # NOTE range.ranges are inclusive, e.g.
    #      Range(0:4) -> ranges = (start=1, stop=3, step=1)
    start = range.ranges[0][0]
    stop = range.ranges[0][1]
    step = range.ranges[0][2]
    return CodeBlock(
        f"__k >= {start} and k <= {stop} and (__k - {start}) % {step} == 0"
    )

class MergeStrategy(Enum):
    none = 0
    trivial = 1
    force_K = 2

class MapMerge(dace_stree.ScheduleNodeTransformer):
    def __init__(self, merge_strategy: MergeStrategy = MergeStrategy.trivial) -> None:
        self.merge_strategy = merge_strategy

    def _merge_maps(self, children: List[dace_stree.ScheduleTreeNode]):
        # count number of maps in children
        map_scopes = [
            map_scope for map_scope in children if isinstance(map_scope, dace_stree.MapScope)
        ]

        if len(map_scopes) == 0:
            # stop the recursion
            return children

        if len(map_scopes) == 1:
            map_scope = map_scopes[0]
            map_index = children.index(map_scope)

            # recurse deeper, see if we can merge more maps
            children[map_index].children = self._merge_maps(map_scope.children)
            return children

        # We have at least two maps at this level. Attempt to merge consecutive maps
        i = 0
        while i < len(children):
            first_map = children[i]
            # skip all non-maps
            if not isinstance(first_map, dace_stree.MapScope):
                i += 1
                continue

            j = i + 1
            while j < len(children):
                second_map = children[j]

                # skip all non-maps
                if not isinstance(second_map, dace_stree.MapScope):
                    j += 1
                    continue

                equal_map_params = first_map.node.params == second_map.node.params
                equal_map_ranges = first_map.node.map.range == second_map.node.map.range
                trivial_merge = equal_map_params and equal_map_ranges
                if self.merge_strategy != MergeStrategy.none and trivial_merge:
                    # merge
                    print(f"trivial merge: {first_map.node.map.params} in {first_map.node.map.range}")
                    first_map.children.extend(second_map.children)
                    del children[j]

                    # TODO also merge containers and symbols (if applicable)

                    # recurse into children
                    first_map.children = self._merge_maps(first_map.children)
                elif self.merge_strategy == MergeStrategy.force_K and is_k_map(first_map) and is_k_map(second_map):
                    # Only for maps in K:
                    # force-merge by expanding the ranges
                    # then, guard children to only run in their respective range
                    first_range = first_map.node.map.range
                    second_range = second_map.node.map.range
                    merged_range = dace.subsets.Range([(
                        f"min({first_range.ranges[0][0]}, {second_range.ranges[0][0]})",
                        f"max({first_range.ranges[0][1]}, {second_range.ranges[0][1]})",
                        1, # NOTE: we can optimize this to gcd later
                    )])

                    # TODO also merge containers and symbols (if applicable)
                    merged_children: List[dace_stree.MapNode] = [
                        dace_stree.IfScope(
                            condition=execution_condition(first_range),
                            children=first_map.children
                        ),
                        dace_stree.IfScope(
                            condition=execution_condition(second_range),
                            children=second_map.children
                        ),
                    ]
                    first_map.children = merged_children

                    # TODO we also might need to merge other stuff in the map
                    first_map.node.map.range = merged_range

                    # delete now-merged second_map
                    del children[j]

                    # recurse into children
                    first_map.children = self._merge_maps(first_map.children)
                else:
                    # we couldn't merge, try the next consecutive pair
                    i += 1

                # break out of the inner while loop
                break

            # in case we merged everything ...
            if j >= len(children):
                i += 1

        return children

    def visit_ScheduleTreeScope(self, node: dace_stree.ScheduleTreeScope):
        node.children = self._merge_maps(node.children)


class KMapLoopFlip(dace_stree.ScheduleNodeVisitor):
    def _move_k_map_loop(self, k_map: dace_stree.MapScope | dace_stree.ForScope) -> None:
        """Move k-{map, loop} out, one level at a time, as far as possible."""
        while(isinstance(k_map.parent, dace_stree.MapScope)):
            parent = k_map.parent
            grand_parent = parent.parent

            parent.children = k_map.children
            parent.parent = k_map

            k_map.parent = grand_parent
            k_map.children = [parent]

            if not isinstance(grand_parent, dace_stree.MapScope):
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

    def visit_MapScope(self, map_scope: dace_stree.MapScope) -> None:
        map_parameter = map_scope.node.params
        # detect k-map
        if len(map_parameter) == 1 and map_parameter[0] == "__k":
            # attempt to move it out as far as possible
            return self._move_k_map_loop(map_scope)

        # visit children
        for child in map_scope.children:
            self.visit(child)

    def visit_ForScope(self, node: dace_stree.ForScope) -> None:
        # detect k-loop
        if node.header.itervar == "__k":
            # attempt to move it out as far as possible
            return self._move_k_map_loop(node)

        # visit children
        for child in node.children:
            self.visit(child)


if __name__ == "__main__":
    functions = [
        # double_map,
        # double_map_with_different_intervals,
        # loop_and_map,
        # mergeable_preserve_order,
        not_mergeable_k_dependency,
    ]

    for function in functions:
        I = np.arange(domain[0] * domain[1] * domain[2], dtype=np.float64).reshape(domain)
        O = np.zeros(domain)

        # # Trigger NDSL orchestration pipeline & grab cached SDFG
        bridge = DaCeGT4Py_Bridge(stencil_factory, function)
        bridge(I, O)
        sdfg: dace.sdfg.SDFG = bridge.__sdfg__(I, O).csdfg.sdfg
        schedule_tree = as_schedule_tree(sdfg)

        # Idea:
        # - first, push k loop out (optional - for CPU opt)
        # - second, merge maps (as before)

        KMapLoopFlip().visit(schedule_tree)
        print("\nFlipped k-map")
        print(f"{schedule_tree.as_string()}")

        MapMerge(MergeStrategy.force_K).visit(schedule_tree)
        print(f"\nMerged map ({function.__name__})")
        print(schedule_tree.as_string())
