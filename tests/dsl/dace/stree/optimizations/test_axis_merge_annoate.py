import pytest
from dace import nodes, subsets
from dace.properties import CodeBlock
from dace.sdfg.analysis.schedule_tree import treenodes as tn

from ndsl.dsl.dace.stree.optimizations import AxisIterator, AxisMergeAnnotate


@pytest.mark.parametrize("axis", (AxisIterator._I, AxisIterator._J, AxisIterator._K))
def test_AnnotationTransformer(axis: AxisIterator) -> None:
    transformer = AxisMergeAnnotate(axis)
    assert str(transformer) == f"AxisMergeAnnotate_{axis.name}"


def test_raise_if_more_than_one_map_param() -> None:
    stree = tn.ScheduleTreeRoot(
        name="test",
        containers={},
        children=[
            tn.MapScope(
                node=nodes.MapEntry(
                    nodes.Map(
                        "map",
                        [AxisIterator._I.as_str(), AxisIterator._J.as_str()],
                        subsets.Range([(0, 2, 1), (0, 3, 1)]),
                    )
                ),
                children=[],
            ),
        ],
    )

    transformer = AxisMergeAnnotate(AxisIterator._I)
    with pytest.raises(
        RuntimeError, match="AxisMergeAnnotate expects maps with only one parameter"
    ):
        transformer.visit(stree)


def test_raise_max_recursion_reached() -> None:
    stree = tn.ScheduleTreeRoot(
        name="test",
        containers={},
        children=[
            tn.IfScope(
                condition=CodeBlock("True"),
                children=[
                    tn.MapScope(
                        node=nodes.MapEntry(
                            nodes.Map(
                                "map",
                                [AxisIterator._I.as_str()],
                                subsets.Range([(0, 2, 1)]),
                            )
                        ),
                        children=[],
                    ),
                ],
            ),
        ],
    )

    transformer = AxisMergeAnnotate(AxisIterator._I, max_recursion_depth=0)
    with pytest.raises(RuntimeError, match="Max recursion depth reached"):
        transformer.visit(stree)


def test_trivial_mergeable() -> None:
    stree = tn.ScheduleTreeRoot(
        name="test",
        containers={},
        children=[
            tn.MapScope(
                node=nodes.MapEntry(
                    nodes.Map(
                        "map", [AxisIterator._I.as_str()], subsets.Range([(0, 2, 1)])
                    )
                ),
                children=[
                    tn.TaskletNode(nodes.Tasklet("noop_1", {}, {}, "pass"), {}, {}),
                ],
            ),
            tn.MapScope(
                node=nodes.MapEntry(
                    nodes.Map(
                        "map", [AxisIterator._J.as_str()], subsets.Range([(0, 2, 1)])
                    )
                ),
                children=[
                    tn.TaskletNode(nodes.Tasklet("noop_2", {}, {}, "..."), {}, {}),
                ],
            ),
            tn.IfScope(
                condition=CodeBlock(True),
                children=[
                    tn.MapScope(
                        node=nodes.MapEntry(
                            nodes.Map(
                                "map",
                                [AxisIterator._I.as_str()],
                                subsets.Range([(0, 2, 1)]),
                            )
                        ),
                        children=[
                            tn.TaskletNode(
                                nodes.Tasklet("noop_1", {}, {}, "pass"), {}, {}
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

    transformer = AxisMergeAnnotate(AxisIterator._I)
    stree = transformer.visit(stree)

    # validate annotation on i_map
    i_map = stree.children[0]
    assert (
        isinstance(i_map, tn.MapScope)
        and i_map.node.params[0] == AxisIterator._I.as_str()
    )
    assert hasattr(i_map, "ndsl") and isinstance(i_map.ndsl, dict)
    assert "AxisMerge" in i_map.ndsl
    assert AxisIterator._I in i_map.ndsl["AxisMerge"]

    # validate no annotation on j_map
    j_map = stree.children[1]
    assert (
        isinstance(j_map, tn.MapScope)
        and j_map.node.params[0] == AxisIterator._J.as_str()
    )
    assert not hasattr(j_map, "ndsl")

    # validate annotation on if with i_map inside
    if_scope = stree.children[2]
    assert isinstance(if_scope, tn.IfScope)
    assert hasattr(if_scope, "ndsl") and isinstance(if_scope.ndsl, dict)
    assert "AxisMerge" in if_scope.ndsl
    assert AxisIterator._I in if_scope.ndsl["AxisMerge"]
