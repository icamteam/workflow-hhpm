from aiida import orm
from aiida.engine import WorkChain

from aiida_hhpm.workflows.cation import HhpmCationWorkChain
from aiida_hhpm.workflows.property import HhpmPropertyWorkChain
from aiida_hhpm.workflows.relax import HhpmRelaxWorkChain


class HhpmMainWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("nwchem.code", valid_type=orm.InstalledCode)
        spec.input("nwchem.metadata", non_db=True)

        spec.input("syva.code", valid_type=orm.InstalledCode)
        spec.input("syva.metadata", non_db=True)

        spec.input("siesta.code", valid_type=orm.InstalledCode)
        spec.input("siesta.metadata", non_db=True)

        spec.input("vasp.code", valid_type=orm.InstalledCode)
        spec.input("vasp.metadata", non_db=True)
        spec.input("vasp.ncore", non_db=True)

        spec.input("shell.metadata", non_db=True)

        spec.input("A", valid_type=orm.StructureData)
        spec.input("B", valid_type=orm.Str)
        spec.input("X", valid_type=orm.Str)

        spec.outline(
            cls.setup,
            cls.cation,
            cls.parse_cation,
            cls.relax,
            cls.parse_relax,
            cls.property,
            cls.parse_property,
        )

    def setup(self):
        pass

    def cation(self):
        inputs = {
            "metadata": {
                "label": "workchain_cation",
            },
            "A": self.inputs.A,
            "nwchem": {
                "code": self.inputs.nwchem.code,
                "metadata": self.inputs.nwchem.metadata,
            },
            "syva": {
                "code": self.inputs.syva.code,
                "metadata": self.inputs.syva.metadata,
            },
        }
        future = self.submit(HhpmCationWorkChain, **inputs)
        self.to_context(workchain_cation=future)

    def parse_cation(self):
        self.ctx.A = self.ctx.workchain_cation.outputs.A

    def relax(self):
        inputs = {
            "siesta": {
                "code": self.inputs.siesta.code,
                "metadata": self.inputs.siesta.metadata,
            },
            "A": self.ctx.A,
            "B": self.inputs.B,
            "X": self.inputs.X,
        }
        future = self.submit(HhpmRelaxWorkChain, **inputs)
        self.to_context(workchain_relax=future)

    def parse_relax(self):
        self.ctx.structure = self.ctx.workchain_relax.outputs.structure

    def property(self):
        inputs = {
            "metadata": {
                "label": "property",
            },
            "vasp": {
                "code": self.inputs.vasp.code,
                "metadata": self.inputs.vasp.metadata,
                "ncore": self.inputs.vasp.ncore,
            },
            "shell": {
                "metadata": self.inputs.shell.metadata,
            },
            "structure": self.ctx.structure,
            "kdensity": orm.Float(4.0),
            "nkpoints_line": orm.Int(100),
        }
        future = self.submit(HhpmPropertyWorkChain, **inputs)
        self.to_context(workchain_property=future)

    def parse_property(self):
        assert self.ctx.workchain_property.is_finished_ok
