import itertools
import re

import mendeleev
import numpy as np
from aiida import orm
from aiida.engine import WorkChain, calcfunction
from aiida.plugins import DataFactory, CalculationFactory
from ase import Atoms
from ase.build import sort
from ase.geometry import permute_axes
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure, Molecule
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import PointGroupAnalyzer, SpacegroupAnalyzer
from spglib import standardize_cell


class HhpmRelaxWorkChain(WorkChain):
    k_density = 0.25

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("siesta.metadata", non_db=True)
        spec.input("siesta.code", valid_type=orm.InstalledCode)

        spec.input("A", valid_type=orm.StructureData)
        spec.input("B", valid_type=orm.Str)
        spec.input("X", valid_type=orm.Str)

        spec.output("structure", valid_type=orm.StructureData)
        spec.output("start_structures", valid_type=orm.Dict)

        spec.outline(
            cls.setup,
            cls.build,
            cls.coarse_relax,
            cls.coarse_filter,
            cls.fine_relax,
            cls.fine_filter,
            cls.standardize,
            cls.result,
        )

    def setup(self):
        self.ctx.A = self.inputs.A
        self.ctx.B = self.inputs.B
        self.ctx.X = self.inputs.X

    def build(self):
        structures = build_A2BX4(self.ctx.A, self.ctx.B, self.ctx.X)
        self.ctx.structures = structures

        self.out("start_structures", structures)

    def coarse_relax(self):
        for name, dict_ in self.ctx.structures.items():
            structure = DataFactory("hhpm.jsonable_structure").from_dict(dict_)

            family = orm.Group.get(label="PseudoDojo/0.4/PBE/SR/standard/psml")
            pseudos = family.get_pseudos(structure=structure)

            parameters = {
                "xc-functional": "GGA",
                "xc-authors": "PBE",
                "dftd3": True,
                "dftd3-periodic": "[1 1 0]",
                "mesh_cutoff": "300 Ry",
                "md-type-of-run": "FIRE",
                "md-variable-cell": True,
                "md-max-force-tol": "0.05 eV/Ang",
                "md-fire-time-step": "0.5 fs",
                "md-steps": 2000,
                "max-scf-iterations": 200,
                "write-coor-step": True,
                "scf-mixer-history": 8,
                "%block geometry-constraints": f"""
cell-vector c
%endblock geometry-constraints""",
            }

            basis = {
                "pao-basis-size": "DZP",
            }

            inputs = {
                "code": self.inputs.siesta.code,
                "structure": structure,
                "parameters": orm.Dict(parameters),
                "basis": orm.Dict(basis),
                "kpoints": self._get_kpoints(structure),
                "pseudos": pseudos,
                "metadata": self.inputs.siesta.metadata,
            }

            self.to_context(**{f"coarse_relax_{name}": self.submit(CalculationFactory("siesta.siesta"), **inputs)})

    def coarse_filter(self):
        structures = {}
        energies = {}

        for name, dict_ in self.ctx.structures.items():
            calc = self.ctx[f"coarse_relax_{name}"]
            if calc.exit_status == 451:
                structure, energy = self._get_final_structure_energy(calc.outputs.retrieved)
            else:
                structure = calc.outputs.output_structure
                energy = calc.outputs.output_parameters["E_KS"]
            structures[name] = DataFactory("hhpm.jsonable_structure")(ase=structure.get_ase()).as_dict()
            energies[name] = energy

        structures = self._filter_structure_by_energy(structures, energies, 1e-3)
        self.ctx.structures = structures

    def fine_relax(self):
        for name, dict_ in self.ctx.structures.items():
            structure = DataFactory("hhpm.jsonable_structure").from_dict(dict_)

            family = orm.Group.get(label="PseudoDojo/0.4/PBE/SR/standard/psml")
            pseudos = family.get_pseudos(structure=structure)

            parameters = {
                "xc-functional": "GGA",
                "xc-authors": "PBE",
                "dftd3": True,
                "dftd3-periodic": "[1 1 0]",
                "mesh_cutoff": "300 Ry",
                "md-type-of-run": "FIRE",
                "md-variable-cell": True,
                "md-max-force-tol": "0.02 eV/Ang",
                "md-fire-time-step": "0.5 fs",
                "md-steps": 2000,
                "max-scf-iterations": 200,
                "write-coor-step": True,
                "scf-mixer-history": 8,
                "%block geometry-constraints": f"""
    cell-vector c
    %endblock geometry-constraints""",
            }

            basis = {
                "pao-basis-size": "DZP",
            }

            inputs = {
                "code": self.inputs.siesta.code,
                "structure": structure,
                "parameters": orm.Dict(parameters),
                "basis": orm.Dict(basis),
                "kpoints": self._get_kpoints(structure),
                "pseudos": pseudos,
                "metadata": self.inputs.siesta.metadata,
            }

            self.to_context(**{f"fine_relax_{name}": self.submit(CalculationFactory("siesta.siesta"), **inputs)})

    def fine_filter(self):
        structures = {}
        energies = {}

        for name, dict_ in self.ctx.structures.items():
            calc = self.ctx[f"fine_relax_{name}"]
            if calc.exit_status == 451:
                structure, energy = self._get_final_structure_energy(calc.outputs.retrieved)
            else:
                structure = calc.outputs.output_structure
                energy = calc.outputs.output_parameters["E_KS"]
            structures[name] = DataFactory("hhpm.jsonable_structure")(ase=structure.get_ase()).as_dict()
            energies[name] = energy

        structure = self._filter_structure_by_energy(structures, energies, 0)
        self.ctx.structure = structure

    def standardize(self):
        self.ctx.structure = _standardize(self.ctx.structure)

    def result(self):
        self.out("structure", self.ctx.structure)

    @staticmethod
    def _get_final_structure_energy(folder):
        text = folder.get_object_content("aiida.out", mode="r")
        coords_text = re.findall(
            r"outcoor: Atomic coordinates \(Ang\):\s+\n((?:\s*\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s*\n)+)", text
        )
        cell_text = re.findall(r"outcell: Unit cell vectors \(Ang\):\s*\n((?:\s*\S+\s+\S+\s+\S+\s*\n)+)", text)
        force_text = re.findall(r"Max\s+(\S+)\s+constrained", text)
        energy_text = re.findall(r"siesta: E_KS\(eV\)\s*=\s*(\S+)", text)

        forces = []
        energies = []
        atoms_list = []

        assert len(coords_text) == len(cell_text) == len(force_text) == len(energy_text)

        for text1, text2, text3, text4 in zip(coords_text, cell_text, force_text, energy_text):
            arr = np.array(text1.split(), dtype=str).reshape((-1, 6))
            symbols = arr[:, -1]
            positions = arr[:, :3].astype(float)
            cell = np.array(text2.split(), dtype=float).reshape((3, 3))
            force = float(text3)
            energy = float(text4)

            atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
            atoms_list.append(atoms)
            forces.append(force)
            energies.append(energy)

        forces = np.array(forces)

        index = forces.argmin()
        atoms = atoms_list[index]
        energy = energies[index]

        return orm.StructureData(ase=atoms), energy

    @staticmethod
    @calcfunction
    def _filter_structure_by_energy(structures, energies, threshold):
        ave_energies = {}

        for name in structures.keys():
            structure = DataFactory("hhpm.jsonable_structure").from_dict(structures[name])
            energy = energies[name] / len(structure.sites)
            ave_energies[name] = energy

        min_energy = min(ave_energies.values())

        filtered_structures = {}
        for name in structures.keys():
            if ave_energies[name] - min_energy <= threshold:
                filtered_structures[name] = structures[name]

        if threshold == 0:
            structure = DataFactory("hhpm.jsonable_structure").from_dict(list(filtered_structures.values())[0])
            atoms = structure.get_ase()
            return orm.StructureData(ase=atoms)
        else:
            return orm.Dict(filtered_structures)

    def _get_kpoints(self, structure):
        kpoints = orm.KpointsData()
        kpoints.set_cell(structure.cell, [True, True, False])
        kpoints.set_kpoints_mesh_from_density(self.k_density)
        return kpoints


@calcfunction
def _standardize(structure):
    atoms = structure.get_ase()

    cell, scaled_positions, numbers = standardize_cell(
        (atoms.get_cell().array, atoms.get_scaled_positions(), atoms.get_atomic_numbers()),
        to_primitive=True,
        symprec=0.01,
    )
    atoms = Atoms(cell=cell, scaled_positions=scaled_positions, numbers=numbers, pbc=True)

    pbc = _get_pbc(atoms)
    assert np.count_nonzero(np.array(pbc)) == 2, "Structure is not 2D"

    if not pbc[0]:
        permute = [1, 2, 0]
    elif not pbc[1]:
        permute = [2, 0, 1]
    else:
        permute = [0, 1, 2]

    atoms = permute_axes(atoms, permute)

    c_length = atoms.cell.lengths()[2]
    c_direction = np.cross(atoms.cell.array[0], atoms.cell.array[1])
    c_direction = c_direction / np.linalg.norm(c_direction)
    atoms.cell.array[2] = c_direction * c_length

    zs = atoms.positions[:, 2]
    sorted_zs = np.sort(zs)
    sorted_zs = np.append(sorted_zs, sorted_zs[0] + np.linalg.norm(atoms.cell[2]))
    arg_max_space = np.argmax(np.diff(sorted_zs)) + 1
    z_max = sorted_zs[arg_max_space]
    atoms.translate([0, 0, -z_max + 0.1])
    atoms.wrap()

    atoms.center(axis=2, vacuum=20 / 2)

    atoms = sort(atoms)

    structure = AseAtomsAdaptor.get_structure(atoms)

    return orm.StructureData(pymatgen_structure=structure)


def _get_pbc(atoms):
    vacuum = 10
    pbc = [True, True, True]

    for i in range(3):
        if atoms.cell.lengths()[i] == 0:
            pbc[i] = False
            continue

        norm = np.cross(atoms.cell[(i + 1) % 3], atoms.cell[(i + 2) % 3])
        norm = norm / np.linalg.norm(norm)
        vector = atoms.cell[i] / np.linalg.norm(atoms.cell[i])
        projected_vacuum = vacuum * (norm @ vector)
        positions = vector @ atoms.positions.T
        sorted_positions = np.sort(positions)
        sorted_positions = np.append(sorted_positions, sorted_positions[0] + np.linalg.norm(atoms.cell[i]))
        arg_max_space = np.argmax(np.diff(sorted_positions)) + 1
        max_space = sorted_positions[arg_max_space] - sorted_positions[arg_max_space - 1]

        if max_space > projected_vacuum:
            pbc[i] = False

    return pbc


@calcfunction
def build_A2BX4(A, B, X):
    import scipy.constants as C

    A = A.get_ase()
    B = B.value
    X = X.value

    b = mendeleev.element(B).atomic_radius + mendeleev.element(X).atomic_radius
    b = b * C.pico / C.angstrom  # bond length
    b *= 1.1  # add 10%
    a = b * 2 * 2 ** 0.5  # lattice constant

    atoms = Atoms(pbc=True, cell=(a, a, 200 * 2))

    # B atoms
    positions = [[0, 0, 0], [a / 2, a / 2, 0]]
    atoms.extend(Atoms([B] * 2, positions=positions))

    # X atoms
    positions = [
        [0, 0, b],
        [0, 0, -b],
        [a / 4, a / 4, 0],
        [3 / 4 * a, 1 / 4 * a, 0],
        [1 / 4 * a, 3 / 4 * a, 0],
        [3 / 4 * a, 3 / 4 * a, 0],
        [a / 2, a / 2, b],
        [a / 2, a / 2, -b],
    ]
    atoms.extend(Atoms([X] * 8, positions=positions))

    A1, A2, A3, A4 = A.copy(), A.copy(), A.copy(), A.copy()  # A1, A2 top; A3, A4 bottom

    A1.translate(np.r_[a / 2, 0, 0] - A1.get_center_of_mass())
    A1.translate(np.r_[0, 0, b / 2] - np.r_[0, 0, A1.positions.min(axis=0)[2]])

    A2.translate(np.r_[0, a / 2, 0] - A2.get_center_of_mass())
    A2.translate(np.r_[0, 0, b / 2] - np.r_[0, 0, A2.positions.min(axis=0)[2]])

    A3.positions = -A3.positions
    A3.translate(np.r_[a / 2, 0, 0] - A3.get_center_of_mass())
    # A3.rotate(180, "x", center="com")
    A3.translate(np.r_[0, 0, -b / 2] - np.r_[0, 0, A3.positions.max(axis=0)[2]])

    A4.positions = -A4.positions
    A4.translate(np.r_[0, a / 2, 0] - A4.get_center_of_mass())
    # A4.rotate(180, "x", center="com")
    A4.translate(np.r_[0, 0, -b / 2] - np.r_[0, 0, A4.positions.max(axis=0)[2]])

    molecule = Molecule(A.get_chemical_symbols(), A.get_positions(), charge_spin_check=False)
    point_group_name = PointGroupAnalyzer(molecule).sch_symbol

    # angle_step = 9 if "5" in point_group_name else 15
    angle_step = 45

    atoms_set = {}
    comp = StructureMatcher()
    for angles in itertools.product(*[np.arange(0, 360, angle_step)] * 4):
        if abs(angles[0] - angles[1]) not in (90, 270):
            continue
        if abs(angles[2] - angles[3]) not in (90, 270):
            continue
        if abs(angles[0] - angles[2]) not in (0,) or abs(angles[1] - angles[3]) not in (0,):
            continue
        atoms_tmp = join_atoms(atoms, [A1, A2, A3, A4], angles)
        number = SpacegroupAnalyzer(AseAtomsAdaptor.get_structure(atoms_tmp), symprec=0.2).get_space_group_number()
        if filter_spacegroup(point_group_name, number):
            atoms_set.setdefault(number, [])
            if is_ok(atoms_tmp, number, atoms_set, comp):
                atoms_set[number].append(atoms_tmp)

    atoms_list = []
    for number_, structure_list_ in sorted(atoms_set.items(), key=lambda x: x[0]):
        atoms_list.extend(structure_list_)

    structure_list = [DataFactory("hhpm.jsonable_structure")(ase=atoms) for atoms in atoms_list]
    for structure in structure_list:
        structure.set_pbc([True, True, False])

    data = {f"structure_{i}": structure.as_dict() for i, structure in enumerate(structure_list, 1)}

    return orm.Dict(data)


def join_atoms(atoms, As, angles):
    atoms = atoms.copy()
    A1, A2, A3, A4 = [A.copy() for A in As]
    A1.rotate(angles[0], "z", center="com")
    A2.rotate(angles[1], "z", center="com")
    A3.rotate(angles[2], "z", center="com")
    A4.rotate(angles[3], "z", center="com")

    atoms += A1 + A2 + A3 + A4
    atoms.translate(np.r_[0, 0, 200])
    atoms.center(axis=2, vacuum=20 / 2)
    atoms.wrap()
    return atoms


def filter_spacegroup(point_group_name, space_group_number):
    if point_group_name == "Cs":
        return True if space_group_number >= 2 else False
    elif point_group_name in ["C2", "C3", "C5", "C2v", "C3v", "C4v", "C5v", "C*v"]:
        return True if space_group_number >= 2 else False
    elif point_group_name == "C1":
        return True if space_group_number >= 2 else False
    else:
        return True if space_group_number >= 2 else False


def is_ok(atoms, number, atoms_set, comp):
    for atoms_ in atoms_set[number]:
        if comp.fit(AseAtomsAdaptor.get_structure(atoms), AseAtomsAdaptor.get_structure(atoms_)):
            return False
    return True
