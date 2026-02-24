from dataclasses import dataclass

from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.subsets import Range

from ndsl.dsl.dace.stree.optimizations import AxisIterator


# A word about the runtime errors
#
# These can become AnnotationErrors, which then can be used as control flow, e.g. in calling code
# if AnnotationError happens during Annotation
#  - log an error (with the NDSL logger),
#  - skip Markup & Action stages,
#  - and possibly clean up the tree of any annotations that this pass has added


@dataclass(kw_only=True)
class AxisMergeAnnotation:
    axis: AxisIterator
    range: Range


class AxisMergeAnnotate(tn.ScheduleNodeTransformer):
    def __init__(self, axis: AxisIterator, *, max_recursion_depth: int = 1024) -> None:
        self.axis = axis
        self._max_recursion_depth = max_recursion_depth
        self._dict_key = "AxisMerge"

    def __str__(self) -> str:
        return f"AxisMergeAnnotate_{self.axis.name}"

    def visit_MapScope(self, node: tn.MapScope) -> tn.MapScope:
        map = node.node.map
        if len(map.params) != 1:
            raise RuntimeError(
                f"AxisMergeAnnotate expects maps with only one parameter, got {len(map.params)} instead: {map.params}"
            )

        map_axis = map.params[0]
        if not map_axis.startswith(self.axis.as_str()):
            node.children = self.visit(node.children)
            return node

        # determine annotation info
        annotation = AxisMergeAnnotation(
            axis=map_axis[: len(self.axis.as_str())],
            range=map.range,
        )

        # add annotation to this node
        self._annotate_scope(node, annotation)

        # recursively add annotation to all parents
        parent = node.parent
        counter = 0
        while not isinstance(parent, tn.ScheduleTreeRoot):
            if not (counter < self._max_recursion_depth):
                raise RuntimeError(f"Max recursion depth reached in {self}.")

            self._annotate_scope(parent, annotation)
            parent = parent.parent
            counter += 1

        return node

    def _annotate_scope(
        self, node: tn.ScheduleTreeScope, annotation: AxisMergeAnnotation
    ) -> None:
        if not hasattr(node, "ndsl"):
            node.ndsl = dict()
        if self._dict_key not in node.ndsl:
            node.ndsl[self._dict_key] = dict()

        node.ndsl[self._dict_key][self.axis] = annotation
