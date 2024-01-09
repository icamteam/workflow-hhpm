import tempfile
from pathlib import Path

import cclib
import numpy as np
import scipy.optimize
from aiida import orm
from aiida.engine import WorkChain, calcfunction
from aiida.plugins import WorkflowFactory
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import PointGroupAnalyzer

from aiida_hhpm.calculations.syva import HhpmSyvaCalculation


class HhpmCationWorkChain(WorkChain):
    _nwchen_workchain_label = "nwchem.base"

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("A", valid_type=orm.StructureData)
        spec.input("nwchem.metadata", non_db=True)
        spec.input("nwchem.code", valid_type=orm.InstalledCode)
        spec.input("syva.metadata", non_db=True)
        spec.input("syva.code", valid_type=orm.InstalledCode)

        spec.output("A", valid_type=orm.StructureData)

        spec.outline(
            cls.setup,
            cls.symmetrize,
            cls.parse_symmetrize,
            cls.relax,
            cls.parse_relax,
            cls.symmetrize,
            cls.parse_symmetrize,
            cls.dipole,
            cls.parse_dipole,
            cls.rotate,
            cls.parse,
        )

    def setup(self):
        self.ctx.structure = self.inputs.A

    def symmetrize(self):
        metadata = self.inputs.syva.metadata.copy()
        metadata["label"] = "symmetrize"
        inputs = {
            "metadata": metadata,
            "code": self.inputs.syva.code,
            "structure": self.ctx.structure,
            "parameters": orm.Dict(
                {
                    "tol": 0.1,
                }
            ),
        }
        self.to_context(calculation_symmetrize=self.submit(HhpmSyvaCalculation, **inputs))

    def parse_symmetrize(self):
        self.ctx.structure = self.ctx.calculation_symmetrize.outputs.output_structure

    def relax(self):
        inputs = {
            "metadata": {
                "label": "workchain_relax",
            },
            "nwchem": {
                "metadata": self.inputs.nwchem.metadata,
                "code": self.inputs.nwchem.code,
                "structure": self.ctx.structure,
                "parameters": orm.Dict(
                    {
                        "basis": {
                            "*": "library 6-311g**",
                        },
                        "charge": 1,
                        "dft": {
                            "xc": "r2scan",
                            "disp": "vdw 4",
                        },
                        "driver": {
                            "tight": "",
                            "maxiter": 500,
                        },
                        "task": "dft optimize",
                    }
                ),
            },
        }
        self.to_context(workchain_relax=self.submit(WorkflowFactory(self._nwchen_workchain_label), **inputs))

    def parse_relax(self):
        self.ctx.structure = self.ctx.workchain_relax.outputs.output_structure

    def dipole(self):
        inputs = {
            "metadata": {
                "label": "dipole",
            },
            "nwchem": {
                "metadata": self.inputs.nwchem.metadata,
                "code": self.inputs.nwchem.code,
                "structure": self.ctx.structure,
                "parameters": orm.Dict(
                    {
                        "basis": {
                            "*": "library def2-tzvpd",
                        },
                        "charge": 1,
                        "dft": {
                            "xc": "b3lyp",
                            "disp": "vdw 4",
                        },
                        "property": {
                            "dipole": "",
                        },
                        "task": "dft property",
                    }
                ),
            },
        }
        self.to_context(dipole=self.submit(WorkflowFactory(self._nwchen_workchain_label), **inputs))

    def parse_dipole(self):
        results = _parse_dipole(self.ctx.dipole.outputs.retrieved)
        self.ctx.structure = results["output_structure"]
        self.ctx.dipole = results["dipole"]

    def rotate(self):
        structure = _rotate_molecule(self.ctx.structure, self.ctx.dipole)
        self.ctx.structure = structure

    def parse(self):
        self.out("A", self.ctx.structure)


@calcfunction
def _parse_dipole(folder):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        filename = tmp_dir / "nwchem.out"
        filename.write_bytes(folder.get_object_content("aiida.out", mode="rb"))

        data = cclib.io.ccread(filename)
        dipole = data.moments[-1]
        dipole = [-i for i in dipole]
        atoms = Atoms(positions=data.atomcoords[-1], numbers=data.atomnos)

        return {
            "dipole": orm.List(dipole),
            "output_structure": orm.StructureData(ase=atoms),
        }


@calcfunction
def _rotate_molecule(structure: orm.StructureData, dipole: orm.List):
    dipole = np.array(dipole.get_list())
    molecule = structure.get_pymatgen_molecule()
    atoms = structure.get_ase()

    pga = PointGroupAnalyzer(molecule)
    point_group = pga.sch_symbol

    if point_group in ["C1", "Cs", "C2v", "C3v"]:
        is_polar = True
    elif point_group in ["Td", "D3h"]:
        is_polar = False
    else:
        raise NotImplementedError(f"Point group {point_group} is not implemented.")

    if is_polar:
        atoms = _rotate_polar_molecule(atoms, point_group, dipole)

    else:
        atoms = _rotate_nonpolar_molecule(atoms, point_group)

    return orm.StructureData(ase=atoms)


def _rotate_polar_molecule(atoms, point_group, dipole):
    dipole /= np.linalg.norm(dipole)

    if point_group == "C1":
        atoms.rotate(dipole, "z", center="COM")
        direction = _get_widest_direction(atoms)
        angle = np.rad2deg(np.arctan2(direction[1], direction[0]))
        atoms.rotate(-angle, "z", center="COM")

    elif point_group == "Cs":
        n = _get_mirror_plane_direction(atoms)
        dipole_in_plane = dipole - dipole @ n * n
        atoms.rotate(dipole_in_plane, "z", center="COM")

        n = _get_mirror_plane_direction(atoms)
        n[2] = 0
        atoms.rotate(n, "y", center="COM")

    elif point_group == "C2v":
        axis = _get_rotation_axis_direction(atoms, 2 * np.pi / 2)
        dipole_on_axis = dipole @ axis * axis
        atoms.rotate(dipole_on_axis, "z", center="COM")

        n = _get_mirror_plane_direction(atoms)
        n[2] = 0
        atoms.rotate(n, "y", center="COM")

    elif point_group == "C3v":
        axis = _get_rotation_axis_direction(atoms, 2 * np.pi / 3)
        dipole_on_axis = dipole @ axis * axis
        atoms.rotate(dipole_on_axis, "z", center="COM")

        n = _get_mirror_plane_direction(atoms)
        n[2] = 0
        atoms.rotate(n, "y", center="COM")

    else:
        raise NotImplementedError(f"Point group {point_group} not implemented.")

    return atoms


def _rotate_nonpolar_molecule(atoms, point_group):
    if point_group == "Td":
        axis = _get_rotation_axis_direction(atoms, 2 * np.pi / 3)
        atoms.rotate(axis, "z", center="COM")

        pos = atoms.positions - atoms.get_center_of_mass()
        pos[:, :2] = 0
        direction = pos[np.argmax(np.linalg.norm(pos, axis=1))]
        if direction[2] > 0:
            atoms.rotate(180, "x", center="COM")

        direction = _get_vertical_mirror_plane_direction(atoms)
        angle = np.rad2deg(np.arctan2(direction[1], direction[0]))
        atoms.rotate(-angle + 90, "z", center="COM")

        pos = atoms.positions - atoms.get_center_of_mass()
        pos[:, 1:] = 0
        direction = pos[np.argmax(np.linalg.norm(pos, axis=1))]
        if direction[0] < 0:
            atoms.rotate(180, "z", center="COM")

    elif point_group == "D3h":
        axis = _get_rotation_axis_direction(atoms, 2 * np.pi / 3)
        atoms.rotate(axis, "y", center="COM")

        axis = _get_rotation_axis_direction(atoms, 2 * np.pi / 2)
        angle = np.rad2deg(np.arctan2(axis[0], axis[2]))
        atoms.rotate(-angle, "y", center="COM")

        pos = atoms.positions - atoms.get_center_of_mass()
        pos[:, :2] = 0
        direction = pos[np.argmax(np.linalg.norm(pos, axis=1))]
        if direction[2] > 0:
            atoms.rotate(180, "x", center="COM")

    else:
        raise NotImplementedError(f"Point group {point_group} not implemented.")

    return atoms


def _get_mirror_plane_direction(atoms):
    ops = PointGroupAnalyzer(AseAtomsAdaptor.get_molecule(atoms, charge_spin_check=False)).get_symmetry_operations()
    for op in ops:
        eigs = np.sort(np.linalg.eigvals(op.rotation_matrix))
        if np.allclose(eigs, [-1, 1, 1], atol=1e-2):
            break
    else:
        raise RuntimeError("No mirror plane found!")

    def func(x):
        a, b, c = x
        return (
            np.array(
                [
                    [1 - 2 * a**2, -2 * a * b, -2 * a * c],
                    [-2 * a * b, 1 - 2 * b**2, -2 * b * c],
                    [-2 * a * c, -2 * b * c, 1 - 2 * c**2],
                ]
            )
            - op.rotation_matrix
        ).ravel()

    res = scipy.optimize.least_squares(func, [1, 1, 1])
    n = res.x / np.linalg.norm(res.x)
    return n


def _get_vertical_mirror_plane_direction(atoms):
    ops = PointGroupAnalyzer(AseAtomsAdaptor.get_molecule(atoms, charge_spin_check=False)).get_symmetry_operations()

    filter_ops = []
    for op in ops:
        eigs = np.sort(np.linalg.eigvals(op.rotation_matrix))
        if np.allclose(eigs, [-1, 1, 1], atol=1e-2):
            filter_ops.append(op)

    if len(filter_ops) == 0:
        raise RuntimeError("No mirror plane found.")

    def get_func(op_):
        def func(x):
            a, b, c = x
            return (
                np.array(
                    [
                        [1 - 2 * a**2, -2 * a * b, -2 * a * c],
                        [-2 * a * b, 1 - 2 * b**2, -2 * b * c],
                        [-2 * a * c, -2 * b * c, 1 - 2 * c**2],
                    ]
                )
                - op_.rotation_matrix
            ).ravel()

        return func

    res_list = []
    for op in filter_ops:
        res = scipy.optimize.least_squares(get_func(op), [1, 1, 1])
        if np.isclose(res.x[2] / np.linalg.norm(res.x), 0, atol=1e-5):
            res_list.append(res)

    res = min(res_list, key=lambda x: np.linalg.norm(x.fun))

    n = res.x / np.linalg.norm(res.x)
    return n


def _get_rotation_axis_direction(atoms, t):
    ops = PointGroupAnalyzer(AseAtomsAdaptor.get_molecule(atoms, charge_spin_check=False)).get_symmetry_operations()

    filter_ops = []

    for op in ops:
        if np.all(np.eye(3) == op.rotation_matrix):
            continue
        elif np.allclose(np.linalg.det(op.rotation_matrix), 1, atol=1e-2):
            filter_ops.append(op)

    if len(filter_ops) == 0:
        raise RuntimeError("No rotation axis found.")

    def get_func(op_):
        def func(q):
            x, y, z = q
            x, y, z = np.r_[x, y, z] / np.linalg.norm(np.r_[x, y, z])
            return (
                np.array(
                    [
                        [
                            x * x * (1 - np.cos(t)) + np.cos(t),
                            x * y * (1 - np.cos(t)) - z * np.sin(t),
                            x * z * (1 - np.cos(t)) + y * np.sin(t),
                        ],
                        [
                            x * y * (1 - np.cos(t)) + z * np.sin(t),
                            y * y * (1 - np.cos(t)) + np.cos(t),
                            y * z * (1 - np.cos(t)) - x * np.sin(t),
                        ],
                        [
                            x * z * (1 - np.cos(t)) - y * np.sin(t),
                            y * z * (1 - np.cos(t)) + x * np.sin(t),
                            z * z * (1 - np.cos(t)) + np.cos(t),
                        ],
                    ]
                )
                - op_.rotation_matrix
            ).ravel()

        return func

    res_list = []
    for op in filter_ops:
        res = scipy.optimize.least_squares(get_func(op), [0.1, 0.1, 0.1])
        res_list.append(res)

    res = min(res_list, key=lambda x: np.linalg.norm(x.fun))

    axis = res.x / np.linalg.norm(res.x)
    return axis


def _get_widest_direction(atoms):
    def func(x):
        t = x
        d = np.r_[np.cos(t), np.sin(t)]
        positions = atoms.get_positions()[:, :2]
        pos = positions @ d
        return -(pos.max() - pos.min())

    res_list = []
    for theta in np.linspace(0, np.pi, 180):
        res = scipy.optimize.minimize(func, theta)
        res_list.append(res)

    res = min(res_list, key=lambda x: x.fun)

    direction = np.r_[np.cos(res.x), np.sin(res.x), 0]

    return direction
