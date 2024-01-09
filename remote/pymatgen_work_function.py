#!/usr/bin/env python

import json
from pathlib import Path

from pymatgen.analysis.surface_analysis import WorkFunctionAnalyzer

path = str(Path.cwd())

wfa = WorkFunctionAnalyzer.from_files(
    poscar_filename=f"{path}/POSCAR",
    locpot_filename=f"{path}/LOCPOT",
    outcar_filename=f"{path}/OUTCAR",
)

data = {}

data["work_function"] = wfa.work_function
data["is_converged"] = wfa.is_converged()
data["efermi"] = wfa.efermi
data["vacuum_locpot"] = wfa.vacuum_locpot
data["along_c"] = list(wfa.along_c)
data["locpot_along_c"] = list(wfa.locpot_along_c)

with open("work_function.json", "w") as f:
    json.dump(data, f, indent=2)
