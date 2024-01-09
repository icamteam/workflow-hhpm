import re
import tempfile
from pathlib import Path

from aiida import orm
from aiida.engine import WorkChain, calcfunction, if_
from aiida.plugins import WorkflowFactory
from aiida_shell import launch_shell_job
from pymatgen.io.vasp import Vasprun

from aiida_hhpm.utils.incar import get_incar_magmom, get_incar_ldau, get_incar_spin, get_incar_dipole
from aiida_hhpm.utils.kpoints import get_kpoints_by_density, get_kpoints_line
from aiida_hhpm.utils.potcar import get_recommend_pp_family, get_recommend_pp_map


class HhpmMomentWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("vasp.code", valid_type=orm.InstalledCode)
        spec.input("vasp.metadata", non_db=True)
        spec.input("vasp.ncore", non_db=True)

        spec.input("structure", valid_type=orm.StructureData)
        spec.input("kdensity", valid_type=orm.Float)

        spec.output("magnetic_moment", valid_type=orm.Float)
        spec.output("dipole_moment", valid_type=orm.Float)

        spec.outline(
            cls.calc_moment,
            cls.parse,
        )

    def calc_moment(self):
        parameters = orm.Dict(
            {
                "incar": {
                    "LWAVE": False,
                    "LCHARG": False,
                    "GGA": "PE",
                    "ENCUT": 600,
                    "LASPH": True,
                    "ADDGRID": True,
                    "PREC": "Accurate",
                    "LREAL": "Auto",
                    "NCORE": self.inputs.vasp.ncore,
                    "IVDW": 12,
                    "ALGO": "All",
                    "ISMEAR": 0,
                    "SIGMA": 0.05,
                    "NELM": 200,
                    "EDIFF": 1e-3,
                    "EFERMI": "MIDGAP",
                    "ISPIN": 2,
                    "LDIPOL": True,
                    "IDIPOL": 3,
                    "DIPOL": self.inputs.structure.get_ase().get_center_of_mass(scaled=True),
                }
            }
        )
        parameters["incar"].update(get_incar_magmom(self.inputs.structure))
        parameters["incar"].update(get_incar_ldau(self.inputs.structure))

        settings = orm.Dict(
            {
                "parser_settings": {
                    "add_misc": True,
                }
            }
        )
        options = orm.Dict(self.inputs.vasp.metadata["options"].copy())

        inputs = {
            "code": self.inputs.vasp.code,
            "structure": self.inputs.structure,
            "kpoints": get_kpoints_by_density(self.inputs.structure, self.inputs.kdensity.value, [True, True, False]),
            "parameters": parameters,
            "potential_family": get_recommend_pp_family(),
            "potential_mapping": get_recommend_pp_map(self.inputs.structure),
            "options": options,
            "settings": settings,
            "clean_workdir": orm.Bool(False),
            "max_iterations": orm.Int(1),
            "metadata": {
                "label": "moment",
            },
        }

        self.to_context(process_moment=self.submit(WorkflowFactory("vasp.vasp"), **inputs))

    def parse(self):
        results = self.parse_moment(self.ctx.process_moment.outputs.retrieved)
        self.out_many(results)

    @staticmethod
    @calcfunction
    def parse_moment(retrieved: orm.FolderData):
        results = {}

        with tempfile.TemporaryDirectory() as tmp_dir:
            outcar_filename = Path(tmp_dir) / "OUTCAR"
            outcar_filename.write_bytes(retrieved.get_object_content("OUTCAR", mode="rb"))
            text = outcar_filename.read_text()
            magnetic_moment = float(re.findall(r"number of electron\s+\S+\s+magnetization\s+(\S+)", text)[-1])
            results["magnetic_moment"] = orm.Float(magnetic_moment)

            vasprun_filename = Path(tmp_dir) / "vasprun.xml"
            vasprun_filename.write_bytes(retrieved.get_object_content("vasprun.xml", mode="rb"))
            text = vasprun_filename.read_text()
            dipole_moment = float(re.findall(r'(?s)<dipole>.*?<v name="dipole">\s*\S+\s+\S+\s+(\S+)\s*</v>', text)[-1])
            results["dipole_moment"] = orm.Float(dipole_moment)

        return results


class HhpmBandOpticsWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("vasp.code", valid_type=orm.InstalledCode)
        spec.input("vasp.metadata", non_db=True)
        spec.input("vasp.ncore", non_db=True)

        spec.input("structure", valid_type=orm.StructureData)
        spec.input("kdensity", valid_type=orm.Float)
        spec.input("nkpoints_line", valid_type=orm.Int)
        spec.input("magnetic_moment", valid_type=orm.Float)
        spec.input("restart_folder", valid_type=orm.RemoteData)
        spec.input("nbands", valid_type=orm.Int)
        spec.input("xc", valid_type=orm.Str)

        spec.outline(
            cls.calc_band,
            cls.parse_band,
            if_(cls.enable_optics)(
                cls.calc_optics,
            ),
            cls.parse,
        )

    def parse(self):
        pass

    def calc_band(self):
        parameters = orm.Dict(
            {
                "incar": {
                    "LWAVE": False,
                    "LCHARG": False,
                    "ICHARG": 11,
                    "ENCUT": 600,
                    "LASPH": True,
                    "ADDGRID": True,
                    "PREC": "Accurate",
                    "LREAL": False,
                    "NCORE": self.inputs.vasp.ncore,
                    "ISMEAR": 0,
                    "SIGMA": 0.05,
                    "NELM": 200,
                    "EDIFF": 1e-5,
                    "EFERMI": "MIDGAP",
                }
            }
        )
        parameters["incar"].update(get_incar_spin(self.inputs.structure, self.inputs.magnetic_moment))
        parameters["incar"].update(get_incar_ldau(self.inputs.structure))
        if self.inputs.xc.value == "PBE":
            parameters["incar"]["GGA"] = "PE"
        elif self.inputs.xc.value == "HLE16":
            parameters["incar"]["GGA"] = "LIBXC"
            parameters["incar"]["LIBXC1"] = "GGA_XC_HLE16"

        settings = orm.Dict(
            {
                "parser_settings": {
                    "add_misc": True,
                },
                "unsupported_parameters": {
                    "LIBXC1": {
                        "default": "GGA_XC_HLE16",
                        "description": "GGA_XC_HLE16",
                        "type": "str",
                        "values": ["GGA_XC_HLE16"],
                    },
                },
            }
        )
        options = orm.Dict(self.inputs.vasp.metadata["options"].copy())

        inputs = {
            "code": self.inputs.vasp.code,
            "structure": self.inputs.structure,
            "kpoints": get_kpoints_line(self.inputs.structure, self.inputs.nkpoints_line.value, 2),
            "parameters": parameters,
            "potential_family": get_recommend_pp_family(),
            "potential_mapping": get_recommend_pp_map(self.inputs.structure),
            "options": options,
            "settings": settings,
            "clean_workdir": orm.Bool(False),
            "max_iterations": orm.Int(1),
            "restart_folder": self.inputs.restart_folder,
            "metadata": {
                "label": "band",
            },
        }

        future = self.submit(WorkflowFactory("vasp.vasp"), **inputs)
        self.to_context(process_band=future)

    def parse_band(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            vasprun_filename = Path(tmp_dir) / "vasprun.xml"
            vasprun_filename.write_bytes(
                self.ctx.process_band.outputs.retrieved.get_object_content("vasprun.xml", mode="rb")
            )
            vasprun = Vasprun(str(vasprun_filename), parse_potcar_file=False)
            bs = vasprun.get_band_structure(efermi="smart")
            band_gap = bs.get_band_gap()["energy"]
            self.ctx.band_gap = band_gap

    def enable_optics(self):
        return self.ctx.band_gap > 0.1

    def calc_optics(self):
        parameters = orm.Dict(
            {
                "incar": {
                    "LWAVE": False,
                    "LCHARG": False,
                    # "GGA": "PE",
                    "ENCUT": 600,
                    "LASPH": True,
                    "ADDGRID": True,
                    "PREC": "Accurate",
                    "LREAL": False,
                    "NCORE": self.inputs.vasp.ncore,
                    "ISMEAR": 0,
                    "SIGMA": 0.01,
                    "NELM": 200,
                    "EDIFF": 1e-5,
                    "EFERMI": "MIDGAP",
                    "NBANDS": self.inputs.nbands.value,
                    "LOPTICS": True,
                    "CSHIFT": 0.1,
                    # "NOMEGA": 200,
                    "OMEGAMAX": 5,
                }
            }
        )
        parameters["incar"].update(get_incar_spin(self.inputs.structure, self.inputs.magnetic_moment))
        parameters["incar"].update(get_incar_ldau(self.inputs.structure))
        if self.inputs.xc.value == "PBE":
            parameters["incar"]["GGA"] = "PE"
        elif self.inputs.xc.value == "HLE16":
            parameters["incar"]["GGA"] = "LIBXC"
            parameters["incar"]["LIBXC1"] = "GGA_XC_HLE16"

        options = orm.Dict(self.inputs.vasp.metadata["options"].copy())

        settings = orm.Dict(
            {
                "parser_settings": {
                    "add_misc": True,
                },
                "unsupported_parameters": {
                    "LIBXC1": {
                        "default": "GGA_XC_HLE16",
                        "description": "GGA_XC_HLE16",
                        "type": "str",
                        "values": ["GGA_XC_HLE16"],
                    },
                },
            }
        )
        inputs = {
            "code": self.inputs.vasp.code,
            "structure": self.inputs.structure,
            "kpoints": get_kpoints_by_density(self.inputs.structure, self.inputs.kdensity.value, [True, True, False]),
            "parameters": parameters,
            "potential_family": get_recommend_pp_family(),
            "potential_mapping": get_recommend_pp_map(self.inputs.structure),
            "options": options,
            "settings": settings,
            "restart_folder": self.inputs.restart_folder,
            "clean_workdir": orm.Bool(False),
            "max_iterations": orm.Int(1),
            "metadata": {
                "label": "optics",
            },
        }

        future = self.submit(WorkflowFactory("vasp.vasp"), **inputs)
        self.to_context(process_optics=future)


class HhpmPropertyPbeWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("vasp.code", valid_type=orm.InstalledCode)
        spec.input("vasp.metadata", non_db=True)
        spec.input("vasp.ncore", non_db=True)
        spec.input("shell.metadata", non_db=True)

        spec.input("structure", valid_type=orm.StructureData)
        spec.input("kdensity", valid_type=orm.Float)
        spec.input("nkpoints_line", valid_type=orm.Int)
        spec.input("magnetic_moment", valid_type=orm.Float)
        spec.input("dipole_moment", valid_type=orm.Float)

        spec.expose_inputs(HhpmBandOpticsWorkChain, exclude=["restart_folder", "nbands", "xc"])

        spec.outline(
            cls.call_scf,
            cls.parse_scf,
            cls.call_others,
            cls.parse,
        )

    def call_scf(self):
        parameters = orm.Dict(
            {
                "incar": {
                    "LWAVE": True,
                    "LCHARG": True,
                    "LAECHG": True,
                    "LVHAR": True,
                    "GGA": "PE",
                    "ENCUT": 600,
                    "LASPH": True,
                    "ADDGRID": True,
                    "PREC": "Accurate",
                    "LREAL": False,
                    "NCORE": self.inputs.vasp.ncore,
                    "IVDW": 12,
                    "ISMEAR": 0,
                    "SIGMA": 0.05,
                    "NELM": 200,
                    "EDIFF": 1e-5,
                    "EFERMI": "MIDGAP",
                }
            }
        )
        parameters["incar"].update(get_incar_spin(self.inputs.structure, self.inputs.magnetic_moment))
        parameters["incar"].update(get_incar_dipole(self.inputs.structure, self.inputs.dipole_moment))
        parameters["incar"].update(get_incar_ldau(self.inputs.structure))

        settings = orm.Dict(
            {
                "parser_settings": {
                    "add_misc": True,
                }
            }
        )
        options = orm.Dict(self.inputs.vasp.metadata["options"].copy())

        inputs = {
            "code": (self.inputs.vasp.code),
            "structure": self.inputs.structure,
            "kpoints": get_kpoints_by_density(self.inputs.structure, self.inputs.kdensity.value, [True, True, False]),
            "parameters": parameters,
            "potential_family": get_recommend_pp_family(),
            "potential_mapping": get_recommend_pp_map(self.inputs.structure),
            "options": options,
            "settings": settings,
            "clean_workdir": orm.Bool(False),
            "max_iterations": orm.Int(1),
            "metadata": {
                "label": "scf",
            },
        }

        self.to_context(process_scf=self.submit(WorkflowFactory("vasp.vasp"), **inputs))

    def parse_scf(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            vasprun_filename = Path(tmp_dir) / "vasprun.xml"
            vasprun_filename.write_bytes(
                self.ctx.process_scf.outputs.retrieved.get_object_content("vasprun.xml", mode="rb")
            )
            vasprun = Vasprun(str(vasprun_filename), parse_potcar_file=False)
            self.ctx.nbands = vasprun.get_band_structure().nb_bands

    def call_others(self):
        future_dos = self.calc_dos()
        self.to_context(process_dos=future_dos)
        future_bader = self.calc_bader()
        self.to_context(process_bader=future_bader)
        future_work_function = self.calc_work_function()
        self.to_context(process_work_function=future_work_function)
        future_band_optics = self.call_band_optics()
        self.to_context(process_band_optics=future_band_optics)

    def call_band_optics(self):
        inputs = self.exposed_inputs(HhpmBandOpticsWorkChain)
        inputs.update(
            {
                "nbands": orm.Int(self.ctx.nbands * 2),
                "restart_folder": self.ctx.process_scf.outputs.remote_folder,
                "xc": orm.Str("PBE"),
                "metadata": {
                    "label": "band_optics",
                }
            }
        )
        future = self.submit(HhpmBandOpticsWorkChain, **inputs)
        return future

    def calc_work_function(self):
        code = self.inputs.vasp.code
        remote_folder = self.ctx.process_scf.outputs.remote_folder
        results, future = launch_shell_job(
            "$HOME/bin/pymatgen_work_function.py",
            submit=True,
            nodes={"remote_data": remote_folder},
            outputs=["work_function.json"],
            metadata={
                "options": {
                    "computer": code.computer,
                    "submit_script_filename": "_aiidasubmit_shell.sh",
                    "resources": self.inputs.shell.metadata["options"]["resources"],
                },
                "label": "work_function",
            },
        )
        return future

    def calc_bader(self):
        code = self.inputs.vasp.code
        remote_folder = self.ctx.process_scf.outputs.remote_folder
        results, future = launch_shell_job(
            "$HOME/bin/pymatgen_bader.py",
            submit=True,
            nodes={"remote_data": remote_folder},
            outputs=["bader.json"],
            metadata={
                "options": {
                    "computer": code.computer,
                    "submit_script_filename": "_aiidasubmit_shell.sh",
                    "resources": self.inputs.shell.metadata["options"]["resources"],
                },
                "label": "bader",
            },
        )
        return future

    def calc_dos(self):
        code = self.inputs.vasp.code
        metadata = self.inputs.vasp.metadata
        ncore = self.inputs.vasp.ncore

        parameters = orm.Dict(
            {
                "incar": {
                    "LWAVE": False,
                    "LCHARG": False,
                    "ICHARG": 11,
                    "GGA": "PE",
                    "ENCUT": 600,
                    "LASPH": True,
                    "ADDGRID": True,
                    "PREC": "Accurate",
                    "LREAL": False,
                    "NCORE": ncore,
                    "ISMEAR": -5,
                    "SIGMA": 0.05,
                    "NELM": 200,
                    "EDIFF": 1e-5,
                    "EFERMI": "MIDGAP",
                    "NEDOS": 2001,
                }
            }
        )
        parameters["incar"].update(get_incar_spin(self.inputs.structure, self.inputs.magnetic_moment))
        parameters["incar"].update(get_incar_ldau(self.inputs.structure))

        settings = orm.Dict(
            {
                "parser_settings": {
                    "add_misc": True,
                }
            }
        )
        options = orm.Dict(metadata["options"].copy())
        restart_folder = self.ctx.process_scf.outputs.remote_folder

        inputs = {
            "code": code,
            "structure": self.inputs.structure,
            "kpoints": get_kpoints_by_density(self.inputs.structure, self.inputs.kdensity.value, [True, True, False]),
            "parameters": parameters,
            "potential_family": get_recommend_pp_family(),
            "potential_mapping": get_recommend_pp_map(self.inputs.structure),
            "options": options,
            "settings": settings,
            "clean_workdir": orm.Bool(False),
            "max_iterations": orm.Int(1),
            "restart_folder": restart_folder,
            "metadata": {
                "label": "dos",
            },
        }

        future = self.submit(WorkflowFactory("vasp.vasp"), **inputs)
        return future

    def parse(self):
        pass


class HhpmPropertyHle16WorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("vasp.code", valid_type=orm.InstalledCode)
        spec.input("vasp.metadata", non_db=True)
        spec.input("vasp.ncore", non_db=True)
        spec.input("shell.metadata", non_db=True)

        spec.input("structure", valid_type=orm.StructureData)
        spec.input("kdensity", valid_type=orm.Float)
        spec.input("nkpoints_line", valid_type=orm.Int)
        spec.input("magnetic_moment", valid_type=orm.Float)
        spec.input("dipole_moment", valid_type=orm.Float)

        spec.expose_inputs(HhpmBandOpticsWorkChain, exclude=["restart_folder", "nbands", "xc"])

        spec.outline(
            cls.call_scf,
            cls.parse_scf,
            cls.call_others,
            cls.parse,
        )

    def call_scf(self):
        parameters = orm.Dict(
            {
                "incar": {
                    "LWAVE": True,
                    "LCHARG": True,
                    "GGA": "LIBXC",
                    "LIBXC1": "GGA_XC_HLE16",
                    "ENCUT": 600,
                    "LASPH": True,
                    "ADDGRID": True,
                    "PREC": "Accurate",
                    "LREAL": False,
                    "NCORE": self.inputs.vasp.ncore,
                    "ISMEAR": 0,
                    "SIGMA": 0.05,
                    "NELM": 200,
                    "EDIFF": 1e-5,
                    "EFERMI": "MIDGAP",
                }
            }
        )
        parameters["incar"].update(get_incar_spin(self.inputs.structure, self.inputs.magnetic_moment))
        parameters["incar"].update(get_incar_dipole(self.inputs.structure, self.inputs.dipole_moment))
        parameters["incar"].update(get_incar_ldau(self.inputs.structure))

        settings = orm.Dict(
            {
                "parser_settings": {
                    "add_misc": True,
                },
                "unsupported_parameters": {
                    "LIBXC1": {
                        "default": "GGA_XC_HLE16",
                        "description": "GGA_XC_HLE16",
                        "type": "str",
                        "values": ["GGA_XC_HLE16"],
                    },
                },
            }
        )
        options = orm.Dict(self.inputs.vasp.metadata["options"].copy())

        inputs = {
            "code": (self.inputs.vasp.code),
            "structure": self.inputs.structure,
            "kpoints": get_kpoints_by_density(self.inputs.structure, self.inputs.kdensity.value, [True, True, False]),
            "parameters": parameters,
            "potential_family": get_recommend_pp_family(),
            "potential_mapping": get_recommend_pp_map(self.inputs.structure),
            "options": options,
            "settings": settings,
            "clean_workdir": orm.Bool(False),
            "max_iterations": orm.Int(1),
            "metadata": {
                "label": "scf",
            },
        }

        self.to_context(process_scf=self.submit(WorkflowFactory("vasp.vasp"), **inputs))

    def parse_scf(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            vasprun_filename = Path(tmp_dir) / "vasprun.xml"
            vasprun_filename.write_bytes(
                self.ctx.process_scf.outputs.retrieved.get_object_content("vasprun.xml", mode="rb")
            )
            vasprun = Vasprun(str(vasprun_filename), parse_potcar_file=False)
            self.ctx.nbands = vasprun.get_band_structure().nb_bands

    def call_others(self):
        future_dos = self.calc_dos()
        self.to_context(process_dos=future_dos)
        future_band_optics = self.call_band_optics()
        self.to_context(process_band_optics=future_band_optics)

    def call_band_optics(self):
        inputs = self.exposed_inputs(HhpmBandOpticsWorkChain)
        inputs.update(
            {
                "nbands": orm.Int(self.ctx.nbands * 2),
                "restart_folder": self.ctx.process_scf.outputs.remote_folder,
                "xc": orm.Str("HLE16"),
                "metadata": {
                    "label": "band_optics",
                }
            }
        )
        future = self.submit(HhpmBandOpticsWorkChain, **inputs)
        return future

    def calc_dos(self):
        code = self.inputs.vasp.code
        metadata = self.inputs.vasp.metadata
        ncore = self.inputs.vasp.ncore

        parameters = orm.Dict(
            {
                "incar": {
                    "LWAVE": False,
                    "LCHARG": False,
                    "ICHARG": 11,
                    "GGA": "LIBXC",
                    "LIBXC1": "GGA_XC_HLE16",
                    "ENCUT": 600,
                    "LASPH": True,
                    "ADDGRID": True,
                    "PREC": "Accurate",
                    "LREAL": False,
                    "NCORE": ncore,
                    "ISMEAR": -5,
                    "SIGMA": 0.05,
                    "NELM": 200,
                    "EDIFF": 1e-5,
                    "EFERMI": "MIDGAP",
                    "NEDOS": 2001,
                }
            }
        )
        parameters["incar"].update(get_incar_spin(self.inputs.structure, self.inputs.magnetic_moment))
        parameters["incar"].update(get_incar_ldau(self.inputs.structure))

        settings = orm.Dict(
            {
                "parser_settings": {
                    "add_misc": True,
                },
                "unsupported_parameters": {
                    "LIBXC1": {
                        "default": "GGA_XC_HLE16",
                        "description": "GGA_XC_HLE16",
                        "type": "str",
                        "values": ["GGA_XC_HLE16"],
                    },
                },
            }
        )
        options = orm.Dict(metadata["options"].copy())
        restart_folder = self.ctx.process_scf.outputs.remote_folder

        inputs = {
            "code": code,
            "structure": self.inputs.structure,
            "kpoints": get_kpoints_by_density(self.inputs.structure, self.inputs.kdensity.value, [True, True, False]),
            "parameters": parameters,
            "potential_family": get_recommend_pp_family(),
            "potential_mapping": get_recommend_pp_map(self.inputs.structure),
            "options": options,
            "settings": settings,
            "clean_workdir": orm.Bool(False),
            "max_iterations": orm.Int(1),
            "restart_folder": restart_folder,
            "metadata": {
                "label": "dos",
            },
        }

        future = self.submit(WorkflowFactory("vasp.vasp"), **inputs)
        return future

    def parse(self):
        pass


class HhpmPropertyWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("vasp.code", valid_type=orm.InstalledCode)
        spec.input("vasp.metadata", non_db=True)
        spec.input("vasp.ncore", non_db=True)
        spec.input("shell.metadata", non_db=True)

        spec.input("structure", valid_type=orm.StructureData)
        spec.input("kdensity", valid_type=orm.Float)
        spec.input("nkpoints_line", valid_type=orm.Int)

        spec.expose_inputs(HhpmMomentWorkChain)
        spec.expose_inputs(HhpmPropertyPbeWorkChain, exclude=["magnetic_moment", "dipole_moment"])
        spec.expose_inputs(HhpmPropertyHle16WorkChain, exclude=["magnetic_moment", "dipole_moment"])

        spec.expose_outputs(HhpmMomentWorkChain)

        spec.outline(
            cls.call_moment,
            cls.call_others,
            cls.parse,
        )

    def call_moment(self):
        inputs = self.exposed_inputs(HhpmMomentWorkChain)
        inputs["metadata"] = {"label": "moment"}
        self.to_context(process_moment=self.submit(HhpmMomentWorkChain, **inputs))

    def call_others(self):
        inputs = self.exposed_inputs(HhpmPropertyPbeWorkChain)
        inputs.update(
            {
                "magnetic_moment": self.ctx.process_moment.outputs.magnetic_moment,
                "dipole_moment": self.ctx.process_moment.outputs.dipole_moment,
                "metadata": {"label": "property_pbe"},
            }
        )
        self.to_context(process_property_pbe=self.submit(HhpmPropertyPbeWorkChain, **inputs))

        inputs = self.exposed_inputs(HhpmPropertyHle16WorkChain)
        inputs.update(
            {
                "magnetic_moment": self.ctx.process_moment.outputs.magnetic_moment,
                "dipole_moment": self.ctx.process_moment.outputs.dipole_moment,
                "metadata": {"label": "property_hle16"},
            }
        )
        self.to_context(process_property_hle16=self.submit(HhpmPropertyHle16WorkChain, **inputs))

    def parse(self):
        self.out_many(self.exposed_outputs(self.ctx.process_moment, HhpmMomentWorkChain))
