from typing import Any, List
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
from dace.transformation.dataflow import MapFusion
from dace.transformation.interstate import StateFusion

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

# simple case of (currently) non-mergeable intervals (working)
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

# this is fine (working)
def loop_and_map(in_field: FloatField, out_field: FloatField):
    with computation(FORWARD), interval(...):
        out_field = in_field

    with computation(PARALLEL), interval(...):
        out_field = in_field * 3

# no overcomputation (yet) (working)
def overcomputation(in_field: FloatField, out_field: FloatField):
    with computation(PARALLEL), interval(1, None):
        out_field = in_field

    with computation(PARALLEL), interval(...):
        out_field = in_field * 3

# merging IJ - keeping K loops (working)
def not_mergeable_preserve_order(in_field: FloatField, out_field: FloatField):
    with computation(PARALLEL), interval(1, None):
        out_field = in_field

    with computation(PARALLEL), interval(...):
        out_field = in_field * 3

    with computation(PARALLEL), interval(1, None):
        out_field = in_field * 4

# merging IJ - keeping K loops (working)
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

# first (naive (and broken)) version
class MapMergeNaive(dace_stree.ScheduleNodeTransformer):
    def _merge_maps(self, children: dace_stree.ScheduleTreeScope | dace_stree.MapScope):
        map_scopes = [
            map_scope for map_scope in children if isinstance(map_scope, dace_stree.MapScope)
        ]

        if len(map_scopes) == 0:
            # stop the recursion if there's no more maps to consider
            return children
        
        if len(map_scopes) == 1:
            map_scope = map_scopes[0]
            map_index = children.index(map_scope)

            # recurse deeper, see if we can merge more maps
            children[map_index].children = self._merge_maps(map_scope.children)
            return children

        # We have at least two maps at this level. Attempt to merge them.
        i = 0
        while i < len(children):
            first_map = children[i]
            if not isinstance(first_map, dace_stree.MapScope):
                i = i + 1
                continue
            
            j = i + 1
            while j < len(children):
                second_map = children[j]

                if not isinstance(second_map, dace_stree.MapScope):
                    j = j + 1
                    continue

                if first_map.node.range == second_map.node.range:
                    print(f"merging {first_map.node.range} with {second_map.node.range}")
                    first_map.children.extend(second_map.children)
                    del children[j]
                    print(f"i={i}, j={j}, len(children)={len(children)}")
                else:
                    print(f"not merging {first_map.node.range} with {second_map.node.range}")

                j = j + 1
            i = i + 1

        return self._merge_maps(children)

    def visit_ScheduleTreeScope(self, node: dace_stree.ScheduleTreeScope):
        node.children = self._merge_maps(node.children)

class MapMerge(dace_stree.ScheduleNodeTransformer):
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

                mergeable = first_map.node.range == second_map.node.range
                if mergeable:
                    # merge
                    first_map.children.extend(second_map.children)
                    del children[j]

                    # TODO also merge containers and symbols (if applicable)

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


class KLoopFlip(dace_stree.ScheduleNodeVisitor):
    def _move_k_map(self, k_map: dace_stree.MapScope) -> None:
        """Move k loop out one level at a time (as far as possible)."""
        print("attempting to move k-loop")
        while(isinstance(k_map.parent, dace_stree.MapScope)):
            print("move k-loop out one level")
            parent = k_map.parent
            grand_parent = parent.parent

            parent.children = k_map.children
            parent.parent = k_map

            k_map.parent = grand_parent
            k_map.children = [parent]

            if not isinstance(grand_parent, dace_stree.MapScope):
                # insert k_map before parent, then delete parent
                # from the list of children in grand_parent
                parent_index = grand_parent.children.index(parent)
                grand_parent.children.insert(parent_index, k_map)
                grand_parent.children.remove(parent)
                # grand_parent.children.append(k_map)
            else:
                assert grand_parent is not None
                grand_parent.children = [k_map]    

            #if grand_parent is not None:
            #    grand_parent.children = [k_map]

            #if isinstance(parent, dace_stree.ScheduleTreeScope):
            #    parent.children.append(k_map)
            #else:
            #    assert grand_parent is not None
            #    grand_parent.children = [k_map]
        #
        #  return k_map

    def visit_MapScope(self, map_scope: dace_stree.MapScope) -> None:
        print(f"params: {map_scope.node.params}")

        map_parameter = map_scope.node.params
        if len(map_parameter) == 1 and map_parameter[0] == "__k":
            print("we should move this map out")
            self._move_k_map(map_scope)
            return
            # return self._move_k_map(map_scope)
            # print(f"{map_scope.as_string()}")
            # return new_scope

        for child in map_scope.children:
            self.visit(child)
        
        # return map_scope

#    def visit_ScheduleTreeScope(self, node: dace_stree.ScheduleTreeScope):
#        print("visiting scope")

        #is_map_scope = isinstance(node, dace_stree.MapScope)
        #is_k_map = is_map_scope and len(node.node.params) == 1 and node.node.params[0] == "__k"
        #if is_k_map:
        #    print("found k-map, attempting to move it out")
        #    return self._move_k_map(node)
#
        #for child in node.children:
        #    child = self.visit(child)
#
        #return node


if __name__ == "__main__":
    functions = [
        double_map,
        # double_map_with_different_intervals,
        # loop_and_map,
        # overcomputation,
        # not_mergeable_preserve_order,
        # not_mergeable_k_dependency,
    ]

    for function in functions:
        I = np.arange(domain[0] * domain[1] * domain[2], dtype=np.float64).reshape(domain)
        O = np.zeros(domain)

        # # Trigger NDSL orchestration pipeline & grab cached SDFG
        bridge = DaCeGT4Py_Bridge(stencil_factory, function)
        bridge(I, O)
        sdfg: dace.sdfg.SDFG = bridge.__sdfg__(I, O).csdfg.sdfg
        manual_tree = as_schedule_tree(sdfg)
        schedule_tree = as_schedule_tree(sdfg)

        # Idea:
        # - first, push k loop out (optional - for CPU opt)
        # - second, merge maps (as before)

        print(f"Before\n\n{manual_tree.as_string()}")

        #              tile i/j     i/j map     k-map
        manual_tree.children[2].children[0].children[0]
        k_map = manual_tree.children[2].children[0].children[0]

        ij_map = k_map.parent
        ij_parent = ij_map.parent

        ij_map.children = k_map.children
        ij_map.parent = k_map
        
        k_map.parent = ij_parent
        k_map.children = [ij_map]
        
        ij_parent.children = [k_map]

        print(f"After\n\n{manual_tree.as_string()}")

        KLoopFlip().visit(schedule_tree)
        print(f"Flipped loops\n\n{schedule_tree.as_string()}")

        # MapMerge().visit(schedule_tree)
        # print(f"Schedule Tree: {function.__name__}")
        # print(schedule_tree.as_string())
