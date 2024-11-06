from typing import List
from dace.sdfg.analysis.schedule_tree.sdfg_to_tree import as_schedule_tree
from gt4py.cartesian.gtscript import (
    computation,
    interval,
    PARALLEL,
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

stencil_factory, ijk_quantity_factory = get_factories_single_tile_orchestrated_cpu(
    domain[0], domain[1], domain[2], 0
)


def double_map(in_field: FloatField, out_field: FloatField):
    with computation(PARALLEL), interval(...):
        out_field = in_field

    with computation(PARALLEL), interval(...):
        out_field = in_field * 3


class DaCeGT4Py_Bridge:
    def __init__(self, stencil_factory: StencilFactory):
        orchestrate(obj=self, config=stencil_factory.config.dace_config)
        self.stencil = stencil_factory.from_dims_halo(
            func=double_map,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )

    def __call__(self, in_field: FloatField, out_field: FloatField):
        self.stencil(in_field, out_field)


class MapMerge(dace_stree.ScheduleNodeTransformer):
    def _merge_maps(self, children: dace_stree.ScheduleTreeScope | dace_stree.MapScope):
        map_scopes = [
            map_scope for map_scope in children if isinstance(map_scope, dace_stree.MapScope)
        ]

        if len(map_scopes) == 0:
            # stop the recursion if there's no more maps to consider
            # return
            return children
        
        if len(map_scopes) == 1:
            map_scope = map_scopes[0]
            map_index = children.index(map_scope)

            # recurse deeper, see if we can merge more maps
            # return self._merge_maps(children[map_index].children)
            children[map_index].children = self._merge_maps(map_scope.children)
            return children

        # TODO
        # We have at least two maps at this level. Attempt to merge them.
        i = 0
        while i < len(children):
            # first_map = children[i]
            if not isinstance(children[i], dace_stree.MapScope):
                i = i + 1
                continue
            
            j = i + 1
            while j < len(children):
                # second_map = children[j]

                if not isinstance(children[j], dace_stree.MapScope):
                    j = j + 1
                    continue

                if children[i].node.range == children[j].node.range:
                    print(f"merging {children[i].node.range} with {children[j].node.range}")
                    children[i].children.extend(children[j].children)
                    del children[j]
                    # children[i].parent.children.remove(children[j])
                    print(f"i={i}, j={j}, len(children)={len(children)}")
                else:
                    print(f"not merging {children[i].node.range} with {children[j].node.range}")

                j = j + 1
            i = i + 1

        return self._merge_maps(children)

    def visit_ScheduleTreeScope(self, node: dace_stree.ScheduleTreeScope):
        node.children = self._merge_maps(node.children)

if __name__ == "__main__":
    I = np.arange(domain[0] * domain[1] * domain[2], dtype=np.float64).reshape(domain)
    O = np.zeros(domain)

    # # Trigger NDSL orchestration pipeline & grab cached SDFG
    bridge = DaCeGT4Py_Bridge(stencil_factory)
    bridge(I, O)
    sdfg: dace.sdfg.SDFG = bridge.__sdfg__(I, O).csdfg.sdfg
    sdfg.save("orig.sdfg")
    other_schedule_tree = as_schedule_tree(sdfg)
    schedule_tree = as_schedule_tree(sdfg)

    #### Classic SDFG transform
    # Strategy: fuse states out of the way, then merge maps

    # State fusion
    # No fusion occurs because of a potential write race since
    # it would be putting both maps under one input
    r = sdfg.apply_transformations_repeated(
        StateFusion,
        print_report=True,
        validate_all=True,
    )
    print(f"\nState fusion\n - Fused {r} states")
    sdfg.save("state_fusion.sdfg")
    
    # No fusion occurs because maps are in different states
    # (previous failure to merge)
    r = sdfg.apply_transformations_repeated(
        MapFusion,
        print_report=True,
        validate_all=True,
    )
    print(f"\nMap fusion\n - Fused {r} maps")
    sdfg.save("map_fusion.sdfg")

    #### Schedule Tree transform
    # We demonstrate here a very basic usage of Schedule Tree to merge
    # maps that have the same range, which should be the first pass
    # we write
    print("\nSchedule tree\n Before merge")
    print(schedule_tree.as_string())
    first_map: dace_stree.MapScope = schedule_tree.children[2]
    second_map: dace_stree.MapScope = schedule_tree.children[3]
    if first_map.node.range == second_map.node.range:
        first_map.children.extend(second_map.children)
        first_map.parent.children.remove(second_map)
    print(" After merge")
    print(schedule_tree.as_string())

    print(" Other tree")
    MapMerge().visit(other_schedule_tree)
    print(other_schedule_tree.as_string())
