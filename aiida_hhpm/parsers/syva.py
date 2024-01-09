import re

import numpy as np
from aiida import orm
from aiida.parsers import Parser
from aiida.plugins import CalculationFactory
from ase import Atoms


class HhpmSyvaParser(Parser):
    def parse(self, **kwargs):
        self._parse_structure()
        self._parse_parameters()

    def _parse_structure(self):
        filenames = self.retrieved.base.repository.list_object_names()
        filename = next(x for x in filenames if x.endswith(".gjf"))
        content = self.retrieved.base.repository.get_object_content(filename)

        match = re.search(r"(?m)(^[ \t]*\d+[ \t]+\S+[ \t]+\S+[ \t]+\S+[ \t]*$\n)+", content).group(0)
        arr = np.array(match.split(), dtype=str).reshape((-1, 4))
        numbers = arr[:, 0].astype(int)
        positions = arr[:, 1:].astype(float)

        atoms = Atoms(numbers=numbers, positions=positions)
        output_structure = orm.StructureData(ase=atoms)

        self.out("output_structure", output_structure)

    def _parse_parameters(self):
        content = self.retrieved.base.repository.get_object_content(CalculationFactory("hhpm.syva")._stdout_name)
        try:
            point_group = re.search("The structure should belong to the (.*) point group.", content).group(1).strip()
        except AttributeError:
            point_group = "C1"
            is_polar = True
            is_chiral = True
        else:
            polar_chiral_raw = re.search("The molecule is (not polar|polar) and (not chiral|chiral).", content).groups()
            is_polar = True if polar_chiral_raw[0].strip() == "polar" else False
            is_chiral = True if polar_chiral_raw[1].strip() == "chiral" else False

        parameters = {
            "point_group": point_group,
            "is_polar": is_polar,
            "is_chiral": is_chiral,
        }

        self.out("output_parameters", orm.Dict(parameters))
