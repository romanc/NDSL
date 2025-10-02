from ndsl import StencilFactory, orchestrate
from ndsl.boilerplate import DaceConfig, get_factories_single_tile_orchestrated
from ndsl.constants import X_DIM, X_INTERFACE_DIM, Y_DIM, Y_INTERFACE_DIM, Z_DIM
from ndsl.dsl.gt4py import PARALLEL, computation, horizontal, interval, region
from ndsl.dsl.typing import Float, FloatField


def fill_corners_dgrid_defn(
    x_in: FloatField,  # type: ignore
    x_out: FloatField,  # type: ignore
    y_in: FloatField,  # type: ignore
    y_out: FloatField,  # type: ignore
    mysign: float,
):
    from __externals__ import i_start, j_start

    with computation(PARALLEL), interval(...):
        # part of sw corner
        with horizontal(region[i_start - 1, j_start - 1]):
            x_out[0, 0, 0] = mysign * y_in[0, 1, 0]
            y_out[0, 0, 0] = mysign * x_in[1, 0, 0]


class OrchestratedCorner:
    def __init__(self, stencil_factory: StencilFactory) -> None:
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config
            or DaceConfig(communicator=None, backend=stencil_factory.backend),
        )
        origin, domain = stencil_factory.grid_indexing.get_origin_domain(
            dims=[X_DIM, Y_DIM, Z_DIM]
        )
        axes_offsets = stencil_factory.grid_indexing.axis_offsets(origin, domain)

        self.corner_stencil = stencil_factory.from_origin_domain(
            fill_corners_dgrid_defn,
            externals=axes_offsets,
            origin=origin,
            domain=domain,
        )

    def __call__(self, x, y):
        self.corner_stencil(x, x, y, y, 1.0)


def test_empty_corners():
    stencil_factory, quantity_factory = get_factories_single_tile_orchestrated(
        nx=12, ny=12, nz=5, nhalo=0
    )
    # Make sure we are "in the middle" and don't trigger any corner code
    stencil_factory.grid_indexing.south_edge = False
    stencil_factory.grid_indexing.north_edge = False
    stencil_factory.grid_indexing.west_edge = False
    stencil_factory.grid_indexing.east_edge = False
    stencil_factory.grid_indexing.axis_offsets

    x = quantity_factory.empty(dims=[X_INTERFACE_DIM, Y_DIM, Z_DIM], units="n/a")
    y = quantity_factory.empty(dims=[X_DIM, Y_INTERFACE_DIM, Z_DIM], units="n/a")

    orchestrated_corner = OrchestratedCorner(stencil_factory)
    orchestrated_corner(x, y)


# Note
# `other_field` is unused and that is on purpose.
def unused_field_stencil(
    field: FloatField, other_field: FloatField, result: FloatField  # type: ignore
):
    with computation(PARALLEL), interval(...):
        result = field[1, 0, 0] + field[0, 1, 0] + field[-1, 0, 0] + field[0, -1, 0]


class OrchestratedUnusedField:
    def __init__(self, stencil_factory: StencilFactory):
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config
            or DaceConfig(communicator=None, backend=stencil_factory.backend),
        )
        origin, domain = stencil_factory.grid_indexing.get_origin_domain(
            dims=[X_DIM, Y_DIM, Z_DIM]
        )

        self.unused_stencil = stencil_factory.from_origin_domain(
            unused_field_stencil,
            origin=origin,
            domain=domain,
        )

    def __call__(self, x, unused_field, y):
        self.unused_stencil(x, unused_field, y)


def test_unused_field():
    stencil_factory, quantity_factory = get_factories_single_tile_orchestrated(
        nx=12, ny=12, nz=5, nhalo=2
    )

    x = quantity_factory.empty(dims=[X_INTERFACE_DIM, Y_DIM, Z_DIM], units="n/a")
    x_unused = quantity_factory.empty(dims=[X_INTERFACE_DIM, Y_DIM, Z_DIM], units="n/a")
    y = quantity_factory.empty(dims=[X_DIM, Y_INTERFACE_DIM, Z_DIM], units="n/a")

    unused_field_orchestrated = OrchestratedUnusedField(stencil_factory)
    unused_field_orchestrated(x, x_unused, y)


# Note
# `weight` is unused an this is on purpose.
def unused_parameter_stencil(field: FloatField, result: FloatField, weight: Float):  # type: ignore
    with computation(PARALLEL), interval(...):
        result = field[1, 0, 0] + field[0, 1, 0] + field[-1, 0, 0] + field[0, -1, 0]


class OrchestratedUnusedParameter:
    def __init__(self, stencil_factory: StencilFactory):
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config
            or DaceConfig(communicator=None, backend=stencil_factory.backend),
        )
        origin, domain = stencil_factory.grid_indexing.get_origin_domain(
            dims=[X_DIM, Y_DIM, Z_DIM]
        )

        self.unused_stencil = stencil_factory.from_origin_domain(
            unused_parameter_stencil,
            origin=origin,
            domain=domain,
        )

    def __call__(self, x, y):
        value = 42
        self.unused_stencil(x, y, 1.0)
        self.unused_stencil(x, y, 3 + 5)
        self.unused_stencil(x, y, value)
        self.unused_stencil(x, y, value * 2.0)


def test_unused_parameter():
    stencil_factory, quantity_factory = get_factories_single_tile_orchestrated(
        nx=12, ny=12, nz=5, nhalo=2
    )

    x = quantity_factory.empty(dims=[X_INTERFACE_DIM, Y_DIM, Z_DIM], units="n/a")
    y = quantity_factory.empty(dims=[X_DIM, Y_INTERFACE_DIM, Z_DIM], units="n/a")

    unused_parameter_orchestrated = OrchestratedUnusedParameter(stencil_factory)

    unused_parameter_orchestrated(x, y)
