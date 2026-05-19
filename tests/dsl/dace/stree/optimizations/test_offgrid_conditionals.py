from typing import TypeAlias

import pytest

from ndsl import (
    Backend,
    NDSLRuntime,
    QuantityFactory,
    StencilFactory,
    orchestrate,
    stencils,
)
from ndsl.boilerplate import get_factories_single_tile
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.typing import FloatField
from tests.dsl.dace.stree import StreeOptimization, get_SDFG_and_purge


class OrchestratedCode(NDSLRuntime):
    def __init__(self, stencil_factory: StencilFactory, *, flag: bool = True) -> None:
        super().__init__(stencil_factory)

        methods_to_orchestrate = [
            "happy_case",
            "happy_case_2",
            "blocked_by_else",
            "blocked_by_other_nodes",
        ]

        for method in methods_to_orchestrate:
            orchestrate(
                obj=self,
                config=stencil_factory.config.dace_config,
                method_to_orchestrate=method,
            )

        self._copy_stencil = stencil_factory.from_dims_halo(
            func=stencils.copy, compute_dims=[I_DIM, J_DIM, K_DIM]
        )
        self.some_flag = flag

    def happy_case(self, in_field: FloatField, out_field: FloatField) -> None:
        if in_field[0, 0, 0] > 0:
            self._copy_stencil(in_field, out_field)

    def happy_case_2(self, in_field: FloatField, out_field: FloatField) -> None:
        if not self.some_flag:
            self._copy_stencil(in_field, out_field)

    def blocked_by_else(self, in_field: FloatField, out_field: FloatField) -> None:
        if self.some_flag:
            self._copy_stencil(in_field, out_field)
        else:
            self._copy_stencil(out_field, in_field)

    def blocked_by_other_nodes(
        self, in_field: FloatField, out_field: FloatField
    ) -> None:
        if self.some_flag:
            in_field[:] = 42.0
            self._copy_stencil(in_field, out_field)


Factories: TypeAlias = tuple[StencilFactory, QuantityFactory]


class TestStreeInlineOffgridConditionals:
    @pytest.fixture(params=["orch:dace:cpu:IJK", "orch:dace:cpu:KJI"])
    def factories(self, request) -> Factories:

        domain = (3, 3, 4)
        return get_factories_single_tile(
            domain[0], domain[1], domain[2], 0, backend=Backend(request.param)
        )

    @pytest.fixture
    def code(self, factories: Factories) -> OrchestratedCode:
        stencil_factory, _ = factories
        flag_value = 42 > 0
        return OrchestratedCode(stencil_factory, flag=flag_value)

    def test_happy_case(self, code: OrchestratedCode, factories: Factories) -> None:
        stencil_factory, quantity_factory = factories
        in_quantity = quantity_factory.ones([I_DIM, J_DIM, K_DIM], "")
        out_quantity = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "")

        with StreeOptimization():
            code.happy_case(in_quantity, out_quantity)

        precompiled_sdfg = get_SDFG_and_purge(stencil_factory)
        assert precompiled_sdfg.sdfg
