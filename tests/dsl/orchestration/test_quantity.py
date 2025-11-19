import pytest

from ndsl import NDSLRuntime, Quantity, StencilFactory
from ndsl.boilerplate import (
    get_factories_single_tile,
    get_factories_single_tile_orchestrated,
)
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.gt4py import PARALLEL, computation, interval
from ndsl.dsl.typing import FloatField


class QuantityDataAndFieldAccess(NDSLRuntime):
    def __init__(self, stencil_factory: StencilFactory) -> None:
        super().__init__(stencil_factory.config.dace_config)

    def __call__(self, quantity: Quantity) -> None:
        quantity.data[:] = 20
        quantity.field[:] = 10


@pytest.mark.parametrize("backend", ["dace:cpu", "dace:cpu_kfirst"])
def test_quantity_data_and_field_access(backend) -> None:
    domain = (2, 3, 4)
    halo_size = 1

    stencil_factory, quantity_factory = get_factories_single_tile_orchestrated(
        nx=domain[0], ny=domain[1], nz=domain[2], nhalo=halo_size, backend=backend
    )

    code = QuantityDataAndFieldAccess(stencil_factory)
    quantity = quantity_factory.zeros(dims=[X_DIM, Y_DIM, Z_DIM], units="n/a")
    code(quantity)

    assert (quantity.data[0] == 20).all()  # check a part of the halo
    assert (quantity.field[:] == 10).all()  # check the full compute domain


def test_quantity_stencil() -> None:
    domain = (2, 3, 4)
    halo_size = 1

    stencil_factory, quantity_factory = get_factories_single_tile(
        domain[0], domain[1], domain[2], halo_size
    )

    code = QuantityDataAndFieldAccess(stencil_factory)
    quantity = quantity_factory.full(dims=[X_DIM, Y_DIM, Z_DIM], units="n/a", value=20)

    def my_stencil(q_in: FloatField):
        with computation(PARALLEL), interval(...):
            q_in = 10

    stencil = stencil_factory.from_dims_halo(my_stencil, [X_DIM, Y_DIM, Z_DIM])

    stencil(quantity)

    assert (quantity.data[0] == 20).all()  # check a part of the halo
    assert (quantity.field[:] == 10).all()  # check the compute domain
