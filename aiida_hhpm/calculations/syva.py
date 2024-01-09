from aiida import orm
from aiida.common import CalcInfo, CodeInfo
from aiida.common.folders import Folder
from aiida.engine import CalcJob


class HhpmSyvaCalculation(CalcJob):
    _input_name = "syva.in"
    _stdout_name = "syva.out"

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("code", valid_type=orm.InstalledCode)
        spec.input("structure", valid_type=orm.StructureData)
        spec.input("parameters", valid_type=orm.Dict)

        spec.output("output_structure", valid_type=orm.StructureData)
        spec.output("output_parameters", valid_type=orm.Dict)

        spec.inputs["metadata"]["options"]["parser_name"].default = "hhpm.syva"
        spec.inputs["metadata"]["options"]["withmpi"].default = False

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        self._write_input_file(folder)

        code_info = CodeInfo()
        code_info.cmdline_params = [f"--tol={self.inputs.parameters['tol']}", "--gaussian", self._input_name]
        code_info.stdout_name = self._stdout_name
        code_info.join_files = True
        code_info.withmpi = False
        code_info.code_uuid = self.inputs.code.uuid

        calc_info = CalcInfo()
        calc_info.codes_info = [code_info]
        calc_info.retrieve_list = ["*.gjf", self._stdout_name]

        return calc_info

    def _write_input_file(self, folder):
        atoms = self.inputs.structure.get_ase()

        with folder.open(self._input_name, "w") as f:
            f.write("title\n")
            f.write(f"{len(atoms)}\n")
            for atom in atoms:
                f.write("{} {} {} {}\n".format(atom.number, *atom.position))
