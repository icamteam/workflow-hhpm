#!/usr/bin/env python

import json
import subprocess
from pathlib import Path

from pymatgen.command_line.bader_caller import BaderAnalysis

subprocess.run("/path/to/chgsum.pl AECCAR0 AECCAR2", shell=True)

path = str(Path.cwd())

ba = BaderAnalysis(
    f"{path}/CHGCAR",
    f"{path}/POTCAR",
    f"{path}/CHGCAR_sum",
    bader_exe_path="/path/to/bader",
)

with open("bader.json", "w") as f:
    json.dump(ba.summary, f, indent=2)
