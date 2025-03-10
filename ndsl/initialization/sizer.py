import dataclasses
from typing import Dict, Iterable, Sequence, Tuple

import ndsl.constants as constants
from ndsl.comm.partitioner import TilePartitioner
from ndsl.constants import N_HALO_DEFAULT


@dataclasses.dataclass
class GridSizer:
    nx: int
    """length of the x compute dimension for produced arrays"""
    ny: int
    """length of the y compute dimension for produced arrays"""
    nz: int
    """length of the z compute dimension for produced arrays"""
    n_halo: int
    """number of horizontal halo points for produced arrays"""
    extra_dim_lengths: Dict[str, int]
    """lengths of any non-x/y/z dimensions, such as land or radiation dimensions"""

    def get_origin(self, dims: Sequence[str]) -> Tuple[int, ...]:
        raise NotImplementedError()

    def get_extent(self, dims: Sequence[str]) -> Tuple[int, ...]:
        raise NotImplementedError()

    def get_shape(self, dims: Sequence[str]) -> Tuple[int, ...]:
        raise NotImplementedError()


class SubtileGridSizer(GridSizer):
    @classmethod
    def from_tile_params(
        cls,
        nx_tile: int,
        ny_tile: int,
        nz: int,
        n_halo: int,
        extra_dim_lengths: Dict[str, int],
        layout: Tuple[int, int],
        tile_partitioner: TilePartitioner = None,
        tile_rank: int = 0,
    ):
        """Create a SubtileGridSizer from parameters about the full tile.

        Args:
            nx_tile: number of x cell centers on the tile
            ny_tile: number of y cell centers on the tile
            nz: number of vertical levels
            n_halo: number of halo points
            extra_dim_lengths: lengths of any non-x/y/z dimensions,
                such as land or radiation dimensions
            layout: (y, x) number of ranks along tile edges
            tile_partitioner (optional): partitioner object for the tile. By default, a
                TilePartitioner is created with the given layout
            tile_rank (optional): rank of this subtile.
        """
        if tile_partitioner is None:
            tile_partitioner = TilePartitioner(layout)
        y_slice, x_slice = tile_partitioner.subtile_slice(
            tile_rank,
            [constants.Y_DIM, constants.X_DIM],
            [ny_tile, nx_tile],
            overlap=True,
        )
        nx = x_slice.stop - x_slice.start
        ny = y_slice.stop - y_slice.start

        # TODO: Remove after vector halo update issue resolved
        if nx <= n_halo:
            raise Exception(
                "SubtileGridSizer::from_tile_params: Compute domain extent must be greater than halo size"
            )
        if ny <= n_halo:
            raise Exception(
                "SubtileGridSizer::from_tile_params: Compute domain extent must be greater than halo size"
            )

        return cls(nx, ny, nz, n_halo, extra_dim_lengths)

    @classmethod
    def from_namelist(
        cls,
        namelist: dict,
        tile_partitioner: TilePartitioner = None,
        tile_rank: int = 0,
    ):
        """Create a SubtileGridSizer from a Fortran namelist.

        Args:
            namelist: A namelist for the fv3gfs fortran model
            tile_partitioner (optional): a partitioner to use for segmenting the tile.
                By default, a TilePartitioner is used.
            tile_rank (optional): current rank on tile. Default is 0. Only matters if
                different ranks have different domain shapes. If tile_partitioner
                is a TilePartitioner, this argument does not matter.
        """
        if "fv_core_nml" in namelist.keys():
            layout = namelist["fv_core_nml"]["layout"]
            # npx and npy in the namelist are cell centers, but npz is mid levels
            nx_tile = namelist["fv_core_nml"]["npx"] - 1
            ny_tile = namelist["fv_core_nml"]["npy"] - 1
            nz = namelist["fv_core_nml"]["npz"]
        elif "nx_tile" in namelist.keys():
            layout = namelist["layout"]
            # everything is cell centered in this format
            nx_tile = namelist["nx_tile"]
            ny_tile = namelist["nx_tile"]
            nz = namelist["nz"]
        else:
            raise KeyError(
                "Namelist format is unrecognized, "
                "expected to find nx_tile or fv_core_nml"
            )
        return cls.from_tile_params(
            nx_tile,
            ny_tile,
            nz,
            N_HALO_DEFAULT,
            {},
            layout,
            tile_partitioner,
            tile_rank,
        )

    @property
    def dim_extents(self) -> Dict[str, int]:
        return_dict = self.extra_dim_lengths.copy()
        return_dict.update(
            {
                constants.X_DIM: self.nx,
                constants.X_INTERFACE_DIM: self.nx + 1,
                constants.Y_DIM: self.ny,
                constants.Y_INTERFACE_DIM: self.ny + 1,
                constants.Z_DIM: self.nz,
                constants.Z_INTERFACE_DIM: self.nz + 1,
            }
        )
        return return_dict

    def get_origin(self, dims: Iterable[str]) -> Tuple[int, ...]:
        return_list = [
            self.n_halo if dim in constants.HORIZONTAL_DIMS else 0 for dim in dims
        ]
        return tuple(return_list)

    def get_extent(self, dims: Iterable[str]) -> Tuple[int, ...]:
        extents = self.dim_extents
        return tuple(extents[dim] for dim in dims)

    def get_shape(self, dims: Iterable[str]) -> Tuple[int, ...]:
        shape_dict = self.extra_dim_lengths.copy()
        # must pad non-interface variables to have the same shape as interface variables
        shape_dict.update(
            {
                constants.X_DIM: self.nx + 1 + 2 * self.n_halo,
                constants.X_INTERFACE_DIM: self.nx + 1 + 2 * self.n_halo,
                constants.Y_DIM: self.ny + 1 + 2 * self.n_halo,
                constants.Y_INTERFACE_DIM: self.ny + 1 + 2 * self.n_halo,
                constants.Z_DIM: self.nz + 1,
                constants.Z_INTERFACE_DIM: self.nz + 1,
            }
        )
        return tuple(shape_dict[dim] for dim in dims)
