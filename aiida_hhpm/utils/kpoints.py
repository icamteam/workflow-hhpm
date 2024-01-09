import typing
from pprint import pprint

import numpy as np
from aiida import orm
from ase.cell import Cell


def get_kpoints_by_density(
        structure: orm.StructureData,
        density: float,
        pbc: typing.Sequence[bool] = None,
) -> orm.KpointsData:
    if pbc is None:
        pbc = structure.pbc

    kpoints = orm.KpointsData()
    kpoints.set_cell(structure.cell, pbc=pbc)
    kpoints.set_kpoints_mesh_from_density(distance=1 / density)

    return kpoints


def get_kpoints_line(
        structure: orm.StructureData,
        nkpoints: int,
        dimension: int,
) -> orm.KpointsData:
    if dimension != 2:
        raise NotImplementedError("Only support 2D system.")

    atoms = structure.get_ase()
    bp = atoms.cell.bandpath(npoints=nkpoints, pbc=[True, True, False])
    kpts = bp.kpts
    labels = []
    for i, kpt in enumerate(kpts):
        for label, kpt_label in bp.special_points.items():
            if np.allclose(kpt, kpt_label):
                labels.append([i, label])
                break
        else:
            labels.append([i, ""])

    kpoints = orm.KpointsData()
    kpoints.set_kpoints(kpts, cartesian=False, labels=labels)
    return kpoints
